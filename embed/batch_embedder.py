import bz2
import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# for SLURM
LOAD_PATH = '/sciclone/data10/twford/reddit/reddit/'
SAVE_PATH = '/sciclone/geograd/stmorse/reddit/test/'

def main():
    t0 = time.time()
    print(f'GPU enabled? {torch.cuda.is_available()}')
    print(f'Loading model ... ({time.time()-t0:.2f})')

    model = SentenceTransformer('all-MiniLM-L6-v1',
                                device='cuda',
                                model_kwargs={'torch_dtype': 'float16'})
    
    # Path to your bz2 compressed file
    file_path = f'{LOAD_PATH}comments/RC_2007-01.bz2'
    batch_size = 1024  

    # Initialize an array to store embeddings incrementally
    embeddings = []

    # Open the bz2 compressed file
    print(f'> Loading comments... ({time.time()-t0:.2f})')
    k = 0
    with bz2.open(file_path, 'rb') as f:
        batch = []
        for line in f:
            entry = json.loads(line)
            
            if 'body' in entry:
                batch.append(entry['body'])
                    
            # when the batch is full, process it
            if len(batch) == batch_size:
                print(f'> Processing batch {k} ...')

                # encode the batch of sentences on the GPU
                # then move them back to CPU and conver to ndarray
                batch_embeddings = model.encode(batch, convert_to_tensor=True).cpu().numpy()
                embeddings.append(batch_embeddings)
                
                batch = []  # Clear batch after processing
                k += 1
        
        # Process any remaining sentences in the final batch
        if len(batch) > 0:
            print(f'> Leftovers ...')
            batch_embeddings = model.encode(batch, convert_to_tensor=True).cpu().numpy()
            embeddings.append(batch_embeddings)
            batch = []

    # Convert list of arrays into a single array
    embeddings_array = np.vstack(embeddings)

    # Save the embeddings to disk (e.g., as a numpy file)
    with open(f'{SAVE_PATH}test.npy', 'wb') as f:
        np.savez_compressed(f, embeddings=embeddings, allow_pickle=False)

if __name__=="__main__":
    main()

