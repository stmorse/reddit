"""
Implements data load and sentence encoding distributed across multiple nodes
"""

import os
import json
import bz2
import time

import torch
import numpy as np
# from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from sentence_transformers import SentenceTransformer

LOAD_PATH = '/sciclone/data10/twford/reddit/reddit/'
SAVE_PATH = '/sciclone/geograd/stmorse/reddit/test/'

DATA_YEAR = 2007
DATA_MONTH = '01'

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f'cuda:{local_rank}'

    print(f'Process {local_rank}: starting... (world size: {world_size})')
    print(f'{local_rank}: CUDA available? {torch.cuda.is_available()}')
    t0 = time.time()

    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")

    # initialize model on each GPU
    print(f'{local_rank}: Loading model... ({time.time()-t0:.2f})')
    model = SentenceTransformer('all-MiniLM-L6-v2',
                                model_kwargs={'torch_dtype': 'float16'})\
                                    .to(device)
    
    # TODO: need to eliminate / fix this, we're loading the entire file into memory to count lines
    # determine chunk sizes; last rank gets any leftover
    print(f'{local_rank}: Computing chunk sizes... ({time.time()-t0:.2f})')
    file_path = f'{LOAD_PATH}comments/RC_{DATA_YEAR}-{DATA_MONTH}.bz2'
    with bz2.BZ2File(file_path, 'rb') as f:
        total_lines = sum(1 for line in f)
    chunk_size = total_lines // world_size
    start_idx = local_rank * chunk_size
    end_idx = total_lines if local_rank == world_size - 1 else (local_rank + 1) * chunk_size

    # pull sentences
    print(f'{local_rank}: Loading data ... ({time.time()-t0:.2f})')
    sentences = []
    with bz2.BZ2File(file_path, 'rb') as f:
        for i, line in enumerate(f):
            if i < start_idx or i > end_idx:
                continue

            if len(line) < 10:
                continue

            entry = json.loads(line)
            if entry['author'] == '[deleted]':
                continue

            sentences.append(entry['body'])

    print(f'{local_rank}: (data size: {len(sentences)})')

    # Embed sentences for the assigned subset and save
    print(f'{local_rank}: Embedding... ({time.time()-t0:.2f})')
    # embeddings = embed_sentences(data_loader, model, device)
    embeddings = model.encode(sentences)
    
    # save this embedding
    print(f'{local_rank}: Saving to file... ({time.time()-t0:.2f})')
    with open(f'{SAVE_PATH}test/embeddings_{DATA_YEAR}-{DATA_MONTH}-{local_rank}.npz', 'wb') as f:
        np.savez_compressed(f, embeddings=embeddings, allow_pickle=False)

    print(f'{local_rank}: Complete, exiting... ({time.time()-t0:.2f})')
    destroy_process_group()


if __name__ == "__main__":
    main()
