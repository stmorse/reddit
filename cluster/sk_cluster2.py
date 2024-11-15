import argparse
import gc
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
import joblib

def main(args):
    LOAD_MODEL = args.load_model_path
    

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

    for year, month in [(yr, mo) for yr in YEARS for mo in MONTHS]:
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
            print(f'> Scoring: ... ({time.time()-t0:.2f})')
            for k in range(3):
                idx = np.random.choice(L, size=min(N_SAMPLES, L), replace=False)
                labels = model.predict(embeddings[idx])
                score = davies_bouldin_score(embeddings[idx], labels)
                print(f'> Iter {k}: {score}')
    
        # Save partially trained model checkpoint
        print(f'> Saving checkpoint ... ({time.time()-t0:.2f})')
        with open(f'{MODEL_PATH}{SAVE_MODEL}_ckpt.joblib', 'wb') as f:
            joblib.dump(model, f, compress=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--start_month", type=int, required=True)
    parser.add_argument("--end_month", type=int, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    args = parser.parse_args()

    main(args)