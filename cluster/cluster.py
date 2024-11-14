"""
Train KMeans distributed
"""

import os
import argparse

import numpy as np
from dask.distributed import Client
# from dask_ml.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def load_embeddings(path, year, month):
    filename = f'{path}embeddings_{year}-{month}.npz'
    
    with open(filename, 'rb') as f:
        embeddings = np.load(f)['embeddings']
        return embeddings
    
def main(args):
    # Connect to Dask cluster
    client = Client()
    print(client)

    # Initialize MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, batch_size=args.batch_size)
    
    # List all data files to process
    files = [os.path.join(args.load_filepath, f) for f in os.listdir(args.load_filepath) if f.endswith('.npy')]

    for file_path in files:
        # Load data in chunks to avoid memory overload
        data = da.from_array(np.load(file_path), chunks=(args.batch_size, 312))
        
        # Incrementally fit the model with each chunk
        for i in range(0, data.shape[0], args.batch_size):
            batch = data[i:i + args.batch_size].compute()  # Load and process batch
            kmeans.partial_fit(batch)
            print(f"Processed batch {i // args.batch_size + 1} from {file_path}")


    # Save labels
    labels = kmeans.labels_
    with open(args.save_filepath, 'wb') as f:
        np.save(f, labels)

    client.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Distributed KMeans with Dask")
    parser.add_argument('--start_year', type=int, required=True, help="Start year")
    parser.add_argument('--end_year', type=int, required=True, help="End year (inclusive)")
    parser.add_argument('--n_clusters', type=int, default=10, help="Number k-means clusters")
    parser.add_argument('--load_filepath', type=str, required=True, help="Path to data")
    parser.add_argument('--save_filepath', type=str, required=True, help="Path to save labels")

    args = parser.parse_args()
    main(args)