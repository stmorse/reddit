"""
Train KMeans distributed
"""

import gc
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
import joblib

MODEL_PATH = '/sciclone/geograd/stmorse/reddit/td2/'
EMBED_PATH = '/sciclone/geograd/stmorse/reddit/embeddings/'
CHUNK_SIZE = 1000000  # approx 125-250 Mb (?)
BATCH_SIZE = 256 * joblib.cpu_count()   # for partial_fit this may not do anything
LOAD_MODEL = None
N_CLUSTERS = 40
N_SAMPLES  = 300000  # for score function, do over 3 random samples of this size
DO_SCORING = False

# SAVE_MODEL = f'mbkm_{N_CLUSTERS}_2008_td2'

YEARS = [2009, 2010, 2011]
MONTHS = [f'{m:02}' for m in range(1,13)]
# YEARS = [2009]
# MONTHS = ['07']

def main():
    print(f'Starting train.  CPU: {joblib.cpu_count()}')

    t0 = time.time()
    print(f'Loading model ... ({time.time()-t0:.2f})')
    if LOAD_MODEL is None:
        # since we're using partial_fit on chunks, this batch_size is (I think) irrelevant
        print(f'(initiating: n_clusters: {N_CLUSTERS}, batch_size: {BATCH_SIZE})')
        model = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=BATCH_SIZE)
    else:
        print(f'(loading from file: {LOAD_MODEL})')
        with open(f'{MODEL_PATH}LOAD_MODEL', 'rb') as f:
            model = joblib.load(f)

    for year in YEARS:
        for month in MONTHS:
            print(f'\nProcessing {year}-{month} ... ({time.time()-t0:.2f})')

            print(f'> Loading embeddings ... ({time.time()-t0:.2f})')
            file_path = f'{EMBED_PATH}embeddings_{year}-{month}.npz'
            with open(file_path, 'rb') as f:
                embeddings = np.load(f)['embeddings']
                
                L = len(embeddings)
                M = L // CHUNK_SIZE   # num chunks
                print(f'> Fitting (size: {L}) (chunks: {M+1}) ...')
                for i in range(0, M):
                    j = i * CHUNK_SIZE
                    chunk = embeddings[j:j + CHUNK_SIZE]

                    print(f'> Fitting chunk {i} ({len(chunk)}) ... ({time.time()-t0:.2f})')
                    model.partial_fit(chunk)
                    del chunk
                    gc.collect()

                # fit on "leftovers"
                fidx = M * CHUNK_SIZE
                if fidx < L:
                    leftovers = embeddings[fidx:]
                    model.partial_fit(leftovers)
                    del leftovers
                    gc.collect()

                # give a sense of score in this month
                # this won't necessarily converge but should generally go down (?)
                # if DO_SCORING:
                #     print(f'> Scoring: ... ({time.time()-t0:.2f})')
                #     for k in range(3):
                #         idx = np.random.choice(L, size=min(N_SAMPLES, L), replace=False)
                #         labels = model.predict(embeddings[idx])
                #         score = davies_bouldin_score(embeddings[idx], labels)
                #         print(f'> Iter {k}: {score}')
                
        # end for : month

        SAVE_MODEL = f'mbkm_{N_CLUSTERS}_{year}_td2'
        
        # Save trained model 
        print(f'> Saving model ... ({time.time()-t0:.2f})')
        with open(f'{MODEL_PATH}{SAVE_MODEL}.joblib', 'wb') as f:
            joblib.dump(model, f, compress=0)

        # this in case of compatibility issues between sklearn/joblib versions
        print(f'Saving final model cluster centers ...')
        with open(f'{MODEL_PATH}{SAVE_MODEL}_cc.npz', 'wb') as f:
            np.savez_compressed(f, cc=model.cluster_centers_, allow_pickle=False)

    # end for : year

    print(f'COMPLETE')

if __name__ == "__main__":
    main()