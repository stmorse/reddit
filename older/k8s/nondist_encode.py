"""
Input date range
Output embedded comments
"""

import json
import bz2
import gc
import time
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
# from sentence_transformer_hf import SentenceTransformerHF

LOAD_PATH = '/home/projects/reddit/'
SAVE_PATH = '/home/projects/reddit/'

# load data
def load_sentences_bz2(filename):
    with bz2.BZ2File(filename, 'rb') as f:
        for i, line in enumerate(f):
            if len(line) < 10:
                continue

            entry = json.loads(line)
            if entry['author'] == '[deleted]':
                continue

            yield entry['body']


if __name__=='__main__':
    print(f'GPU enabled? {torch.cuda.is_available()}')

    # embed_model = SentenceTransformerHF()
    embed_model = SentenceTransformer('all-MiniLM-L6-v1',
                                      device='cuda',
                                      model_kwargs={'torch_dtype': 'float16'})
    
    years = [2007]
    months = ['01']

    t0 = time.time()
    for year in years:
        for month in months:
            print(f'\nProcessing {year}-{month}... ({time.time()-t0:.2f})')

            # load this month comments and users
            print(f'> Loading users and comments... ({time.time()-t0:.2f})')
            file_path = f'{LOAD_PATH}RC_{year}-{month}.bz2'
            sentences = []  # will store all comment bodies
            for sentence in load_sentences_bz2(file_path):
                sentences.append(sentence)

            # embed comments
            print(f'> Embedding {len(sentences)} sentences... ({time.time()-t0:.2f})')
            embeddings = embed_model.encode(sentences)

            # save embeddings
            print(f'> Saving... ({time.time()-t0:.2f})')
            # with open(f'{SAVE_PATH}embeddings_{year}-{month}.npz', 'wb') as f:
            #     np.savez_compressed(f, embeddings=embeddings, allow_pickle=False)
            with open(f'{SAVE_PATH}test.pt', 'wb') as f:
                torch.save(embeddings, f)

            # clear memory
            print(f'> Garbage collection... ({time.time()-t0:.2f})')
            del embeddings
            del sentences
            gc.collect()

    print(f'\n\nCOMPLETE. ({time.time()-t0:.2f})  Exiting...\n')
            

    