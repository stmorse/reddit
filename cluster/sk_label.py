"""
Train KMeans distributed
"""

import gc
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
import joblib

MODEL_PATH = '/sciclone/geograd/stmorse/reddit/td/'
EMBED_PATH = '/sciclone/geograd/stmorse/reddit/embeddings/'
# LABEL_PATH = '/sciclone/geograd/stmorse/reddit/labels/'
LABEL_PATH = '/sciclone/geograd/stmorse/reddit/td/labels/'
CHUNK_SIZE = 1000000  # approx 125-250 Mb (?)
LOAD_MODEL = 'mbkm_20_2011_td'
N_CLUSTERS = 20
N_SAMPLES  = 300000  # for score function, do over 3 random samples of this size

# YEARS = [2007, 2008, 2009, 2010, 2011]
MONTHS = [f'{m:02}' for m in range(1,13)]
YEARS = [2011]
# MONTHS = ['07']

def main():
    print(f'Starting prediction.  CPU: {joblib.cpu_count()}')

    t0 = time.time()
    print(f'Loading model from file: {LOAD_MODEL} ... ({time.time()-t0:.2f})')
    with open(f'{MODEL_PATH}{LOAD_MODEL}.joblib', 'rb') as f:
        model = joblib.load(f)
    print(f'Model: ({model.cluster_centers_.shape})')

    for year, month in [(yr, mo) for yr in YEARS for mo in MONTHS]:
        print(f'\nProcessing {year}-{month} ... ({time.time()-t0:.2f})')

        print(f'> Loading embeddings ... ({time.time()-t0:.2f})')
        file_path = f'{EMBED_PATH}embeddings_{year}-{month}.npz'
        with open(file_path, 'rb') as f:
            embeddings = np.load(f)['embeddings']
            labels = []
            
            L = len(embeddings)
            M = L // CHUNK_SIZE   # num chunks
            print(f'> Labeling (size: {L}) (chunks: {M+1}) ...')
            for i in range(0, M):
                j = i * CHUNK_SIZE
                chunk = embeddings[j:j + CHUNK_SIZE]

                print(f'> Labeling chunk {i} ({len(chunk)}) ... ({time.time()-t0:.2f})')
                chunk_labels = model.predict(chunk)
                labels.append(chunk_labels)
                del chunk
                gc.collect()

            # label "leftovers"
            fidx = M * CHUNK_SIZE
            if fidx < L:
                leftovers = embeddings[fidx:]
                chunk_labels = model.predict(leftovers)
                labels.append(chunk_labels)
                del leftovers
                gc.collect()

            labels = np.concatenate(labels)
            print(f'> Labeling complete ({labels.shape}) ... ({time.time()-t0:.2f})')

            # give a sense of score in this month
            # this won't necessarily converge but should generally go down (?)
            # print(f'> Scoring: ... ({time.time()-t0:.2f})')
            # for k in range(3):
            #     idx = np.random.choice(L, size=min(N_SAMPLES, L), replace=False)
            #     score = davies_bouldin_score(embeddings[idx], labels[idx])
            #     print(f'> Iter {k}: {score}')
    
        # Save labels
        print(f'> Saving labels ... ({time.time()-t0:.2f})')
        with open(f'{LABEL_PATH}labels_{year}-{month}.npz', 'wb') as f:
            np.savez_compressed(f, labels=labels, allow_pickle=False)

if __name__ == "__main__":
    main()