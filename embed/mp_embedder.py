"""
Streams entirety of a month into CPU Mem, 
then batches into multiple GPU for encoding,
and saves to file
"""

import bz2
import json
import time
import pickle
import gc

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

LOAD_PATH = '/sciclone/data10/twford/reddit/reddit/comments/'
SAVE_PATH = '/sciclone/geograd/stmorse/reddit/embeddings/'
META_PATH = '/sciclone/geograd/stmorse/reddit/metadata/'
BATCH_SIZE = 1000000  # approx 125-250 Mb (?)
SHOW_PROGRESS = False
HF_MODEL = 'all-MiniLM-L6-v2'

YEARS = [2010, 2011]
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# YEARS = [2008]
# MONTHS = ['01']

def main():
    t0 = time.time()
    print(f'GPU enabled? {torch.cuda.is_available()}')
    print(f'Loading model ... ({time.time()-t0:.2f})')

    model = SentenceTransformer(HF_MODEL,
                                device='cuda',
                                model_kwargs={'torch_dtype': 'float16'})  

    print(f'Starting multiprocessing pool ... ({time.time()-t0:.2f})')  

    pool = model.start_multi_process_pool()
    
    for year, month in [(yr, mo) for yr in YEARS for mo in MONTHS]:
        print(f'Processing {year}-{month} ... ({time.time()-t0:.2f})')

        # path to bz2 compressed file
        file_path = f'{LOAD_PATH}RC_{year}-{month}.bz2'

        # initialize an array to store embeddings
        embeddings = []

        # will hold metadata
        df = []

        # Open the bz2 compressed file
        print(f'> Loading comments (Batch size: {BATCH_SIZE})... ({time.time()-t0:.2f})')
        j, k = 0, 0   # line count, batch count
        with bz2.open(file_path, 'rb') as f:
            batch = []
            for line in f:
                entry = json.loads(line)
                j += 1

                if 'body' not in entry or entry['author'] == '[deleted]':
                    continue
                
                # quick and dirty to keep entries closer to SBERT token limit (256)
                body = entry['body']
                if len(body) > 2000:
                    body = body[:2000]

                batch.append(body)

                # continue building metadata
                df.append([k, entry['author'], entry['id'], entry['created_utc']])
                        
                # when the batch is full, process it
                if len(batch) == BATCH_SIZE:
                    print(f'> Processing batch {k} ... ({time.time()-t0:.2f})')

                    # encode the batch of sentences on the GPU
                    # then move them back to CPU and conver to ndarray
                    # batch_embeddings = (model.encode(
                    #     batch, show_progress_bar=SHOW_PROGRESS, convert_to_tensor=True)
                    #     .cpu()
                    #     .numpy()
                    # )
                    batch_embeddings = model.encode_multi_process(
                        batch, pool, show_progress_bar=SHOW_PROGRESS)
                    embeddings.append(batch_embeddings)
                    
                    batch = []  # Clear batch after processing
                    k += 1
            
            # Process any remaining sentences in the final batch
            if len(batch) > 0:
                print(f'> Leftovers ... ({time.time()-t0:.2f})')
                # batch_embeddings = (model.encode(
                #     batch, show_progress_bar=SHOW_PROGRESS, convert_to_tensor=True)
                #     .cpu()
                #     .numpy()
                # )
                batch_embeddings = model.encode_multi_process(
                        batch, pool, show_progress_bar=SHOW_PROGRESS)
                embeddings.append(batch_embeddings)
                batch = []
                k += 1

        # Convert list of arrays into a single array
        embeddings = np.vstack(embeddings)

        # Save the embeddings to disk
        print(f'> Total lines: {j}  Total embeddings: {len(embeddings)}  Total batches: {k}')
        print(f'> Saving to disk ... ({time.time()-t0:.2f})')
        with open(f'{SAVE_PATH}embeddings_{year}-{month}.npz', 'wb') as f:
            np.savez_compressed(f, embeddings=embeddings, allow_pickle=False)

        # save metadata to disk
        print(f'> Saving metadata to disk ... ({time.time()-t0:.2f})')
        with open(f'{META_PATH}metadata_{year}-{month}.pkl', 'wb') as f:
            pickle.dump(df, f)

        print(f'> Garbage collection ... ({time.time()-t0:.2f})')
        del embeddings
        del df
        gc.collect()

    print(f'Stopping multiprocess pool ...')
    model.stop_multi_process_pool(pool)

    print('\nCOMPLETE.\n\n')

if __name__=="__main__":
    main()

