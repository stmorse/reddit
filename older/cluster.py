"""
Input date range
Output cluster model
"""

import gc
import time

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from joblib import dump

LOAD_PATH = '/sciclone/data10/twford/reddit/reddit/'
SAVE_PATH = '/sciclone/geograd/stmorse/reddit/'

# load data
def load_embeddings(year, month):
    filename = f'embeddings_{year}-{month}.npz'
    
    try:
        with open(SAVE_PATH + 'embeddings/' + filename, 'rb') as f:
            embeddings = np.load(f)['embeddings']
            f.close()
            return embeddings
    except FileNotFoundError:
        print(f'Error: {filename} not found')

if __name__=='__main__':
    print(f'GPU enabled? {torch.cuda.is_available()}')

    cluster_model = MiniBatchKMeans(n_clusters=15)
    
    years = [2007, 2008, 2009]
    months = ['01', '02', '03', '04', '05', '06', 
              '07', '08', '09', '10', '11', '12']
    
    t0 = time.time()
    for year in years:
        for month in months:
            print(f'\nProcessing {year}-{month}... ({time.time()-t0:.2f})')

            # load this month comments
            print(f'> Loading comments... ({time.time()-t0:.2f})')
            embeddings = load_embeddings(year, month)

            # fit model
            print(f'> Partial fit ({len(embeddings)})... ({time.time()-t0:.2f})')
            cluster_model.partial_fit(embeddings)

            # clear memory
            print(f'> Garbage collection... ({time.time()-t0:.2f})')
            del embeddings
            gc.collect()

    print('\n\n---\nTRAINING COMPLETE.')

    print('SAVING MODEL...')
    with open(f'{SAVE_PATH}cm_mbkm_20241108.joblib', 'wb') as f:
        fname = dump(cluster_model, f, compress=3)
        print(f'> Saved to: {fname}')

    print('PREDICTING LABELS.\n---\n')
    
    # model is fit to all batches
    # now cycle back through data and predict labels
    for year in years:
        for month in months:
            print(f'\nProcessing {year}-{month}... ({time.time()-t0:.2f})')

            # load comments again
            print(f'> Loading comments... ({time.time()-t0:.2f})')
            embeddings = load_embeddings(year, month)

            # predict labels
            print(f'> Predicting labels ({len(embeddings)})... ({time.time()-t0:.2f})')
            labels = cluster_model.predict(embeddings)

            # save labels
            print(f'> Saving labels... ({time.time()-t0:.2f})')
            with open(f'{SAVE_PATH}labels/labels_{year}-{month}.npz', 'wb') as f:
                np.savez_compressed(f, labels=labels, allow_pickle=False)

            # clear memory
            print(f'> Garbage collection... ({time.time()-t0:.2f})')
            del embeddings
            del labels
            gc.collect()

    print(f'COMPLETE. ({time.time()-t0:.2f})  Exiting...\n')
    