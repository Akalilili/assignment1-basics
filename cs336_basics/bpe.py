from multiprocessing import Pool
import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def worker(args):
        """
        Input
            args: 
                input_path: str
                PAT: str GPT-2 regex pattern
                start, end: int, int

        Output
            frequency_table: dict[int, bytes]
        """
        input_path, PAT, special_tokens, start, end = args
        parital_frequency_table: Counter[tuple[bytes, ...]] = Counter()

        # mp内部自行读取文件而不是统一读取分发给process：
        # 1. multiprocessing 在 Windows 上用 spawn，会把任务参数序列化（pickle）后传给子进程。而 open() 返回的文件对象无法 pickle
        # 2. multi processes同时seek导致指针混乱
        with open(input_path, "rb") as file:
            file.seek(start)
            text = file.read(end-start).decode("utf-8", errors="ignore")
            # 换行转换
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            # remove special tokens
            if special_tokens:
                # generater (re.escape(token)): lazy iterable
                escaped_tokens = (re.escape(token) for token in special_tokens)
                chunks = re.split("|".join(escaped_tokens), text)
            else:
                chunks = [text]

            for chunk in chunks:
                # finditer() instead of findall(), avoiding store all the matches in one list  
                for match in re.finditer(PAT, chunk):
                    byte_tuple = tuple(
                        bytes([byte]) for byte in match.group().encode("utf-8")
                    )
                    parital_frequency_table[byte_tuple] += 1
        
        return parital_frequency_table

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
):
    """
    Input
        input_path: str  Path to a text file with BPE tokenizer training data.
        vocab_size: int  A positive integer that defines the maximum final vocabulary size (including
        the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str]  A list of strings to add to the vocabulary. During training, treat
        them as hard boundaries that prevent merges across their spans, but do not include them when
        computing merge statistics.
    Output
        vocab: dict[int, bytes]  The tokenizer vocabulary, a mapping from int (token ID in the
        vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]]  A list of BPE merges produced from training. Each list
        item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
        <token2>. The merges should be ordered by order of creation.
    """

    
    # Init Vocab
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    for j in range(len(special_tokens)):
        vocab[j+256] = special_tokens[j].encode("utf-8")


    # Pre-Tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    frequency_table: Counter[tuple[bytes, ...]] = Counter()

    # # serial version
    # with open(input_path, "r", encoding="utf-8") as f:
    #     text = f.read()

        # # remove special tokens
        # if special_tokens:
        #     # generater (re.escape(token)): lazy iterable
        #     escaped_tokens = (re.escape(token) for token in special_tokens)
        #     chunks = re.split("|".join(escaped_tokens), text)
        # else:
        #     chunks = [text]

        # for chunk in chunks:
        #     # finditer() instead of findall(), avoiding store all the matches in one list  
        #     for match in re.finditer(PAT, chunk):
        #         byte_tuple = tuple(
        #             bytes([byte]) for byte in match.group().encode("utf-8")
        #         )
        #         frequency_table[byte_tuple] += 1
                    
    # parallel version
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    with Pool(num_processes) as p:
        # imap: lazy iterable
        for partial in p.imap(
            worker, 
            ((input_path, PAT, special_tokens, start, end) for (start, end) in zip(boundaries[:-1], boundaries[1:]))
        ):
            frequency_table += partial

    # Merge
    merges: list[tuple[bytes, bytes]] = []
    pairs_count: Counter[tuple[bytes, bytes]] = Counter()
    # to find affected pair efficiently 
    pair_index_in_table: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    iter_togo = vocab_size - len(vocab)

    # count frequency of pairs
    for key, frequency in frequency_table.items():
        for pair in zip(key, key[1:]):
            pairs_count[pair] += frequency
            pair_index_in_table[pair].add(key)

    # merge and update
    for _ in range(iter_togo):
        # the most frequent and lexicographically greatest pair
        if not pairs_count:
            break
        best = max(
                    pairs_count,
                    key=lambda pair: (pairs_count[pair], pair)
                )

        left, right = best
        merged_token = left + right
        # update merge
        merges.append(best)
        # add new token into vocab
        vocab[len(vocab)] = merged_token

        # avoid editing pair_index_in_table while iterating it
        affected_token_tuples = pair_index_in_table[best].copy()
        # update pairs_count smartly
        for token_tuple in affected_token_tuples:
            frequency = frequency_table[token_tuple]
            # list for the convience of replacing
            new_token_list = list(token_tuple)

            i=0
            while i+1 < len(new_token_list):
                if (new_token_list[i], new_token_list[i+1]) == best:
                    # update pairs_count
                    # if any left neighbour
                    if i > 0:
                        left_neibour = (new_token_list[i-1], new_token_list[i])
                        update_pairs_count(pairs_count, left_neibour, -frequency)

                        new_left_neibour = (new_token_list[i-1], merged_token)
                        update_pairs_count(pairs_count, new_left_neibour, frequency)
                    # if any right neighbour
                    if i+2 < len(new_token_list):
                        right_neibour = (new_token_list[i+1], new_token_list[i+2])
                        update_pairs_count(pairs_count, right_neibour, -frequency)

                        new_right_neibour = (merged_token, new_token_list[i+2])
                        update_pairs_count(pairs_count, new_right_neibour, frequency)

                    # save token change
                    new_token_list.pop(i)
                    new_token_list[i] = merged_token
                i+=1

                    
            # update frequency_table.keys()
            new_token_tuple = tuple(new_token_list)
            del frequency_table[token_tuple]  
            # case: ab+c and a+bc -> += instead of =
            frequency_table[new_token_tuple] += frequency

            for pair in set(zip(token_tuple, token_tuple[1:])):
                pair_index_in_table[pair].remove(token_tuple)
            for pair in set(zip(new_token_tuple, new_token_tuple[1:])):
                pair_index_in_table[pair].add(new_token_tuple)

        del pair_index_in_table[best] 
        del pairs_count[best] 

    return vocab, merges

def update_pairs_count(
    counter: Counter[tuple[bytes, bytes]],
    pair: tuple[bytes, bytes],
    change: int
):
    counter[pair] += change
    if counter[pair] == 0:
        del counter[pair]
        return 
    return 