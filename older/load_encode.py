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

LOAD_PATH = '/sciclone/data10/twford/reddit/reddit/'
SAVE_PATH = '/sciclone/geograd/stmorse/reddit/'

# load data
def load_file(year, month):
    filename = f'RC_{year}-{month}.bz2'
    
    try:
        with bz2.BZ2File(LOAD_PATH + 'comments/' + filename, 'rb') as file:
            for line in file:
                if len(line) > 10:
                    yield json.loads(line)
    except FileNotFoundError:
        print(f'Error: {filename} not found')

if __name__=='__main__':
    print(f'GPU enabled? {torch.cuda.is_available()}')

    embed_model = SentenceTransformer('all-MiniLM-L6-v2', 
                                      device='cuda',
                                      model_kwargs={'torch_dtype': 'float16'})
    
    years = [2010]
    # months = ['01', '02', '03', '04', '05', '06', 
    #           '07', '08', '09', '10', '11', '12']
    months = ['01']

    t0 = time.time()
    for year in years:
        for month in months:
            print(f'\nProcessing {year}-{month}... ({time.time()-t0:.2f})')

            # load this month comments and users
            print(f'> Loading users and comments... ({time.time()-t0:.2f})')
            sentences = []  # will store all comment bodies
            df = []         # will store metadata
            k = 0           # index common between sentences and df
            for entry in load_file(year, month):
                author = entry['author']
                if author != '[deleted]':
                    sentences.append(entry['body'])
                    df.append([k, author, entry['id'], entry['created_utc']])
                    k += 1

            # embed comments
            print(f'> Embedding {len(sentences)} sentences... ({time.time()-t0:.2f})')
            embeddings = embed_model.encode(sentences, show_progress_bar=True)

            # save embeddings
            print(f'> Saving... ({time.time()-t0:.2f})')
            with open(f'{SAVE_PATH}embeddings/embeddings_{year}-{month}.npz', 'wb') as f:
                np.savez_compressed(f, embeddings=embeddings, allow_pickle=False)

            # save metadata
            with open(f'{SAVE_PATH}metadata/metadata_{year}-{month}.pkl', 'wb') as f:
                pickle.dump(df, f)

            # clear memory
            print(f'> Garbage collection... ({time.time()-t0:.2f})')
            del embeddings
            del sentences
            del df
            gc.collect()

    print(f'\n\nCOMPLETE. ({time.time()-t0:.2f})  Exiting...\n')
            

    