import time
import bz2
import json
import pickle
import gc

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def load_sentences_bz2(path, year, month):
    sentences = []
    with bz2.BZ2File(f'{path}RC_{year}-{month}.bz2', 'rb') as f:
        for line in f:
            entry = json.loads(line)
            if 'body' not in entry or entry['author']=='[deleted]':
                continue
            sentences.append(entry['body'])
    return sentences

# load random sample from file given ids
def load_sample_from_file(path, year, month, ids):
    # make ids a set for quicker lookup
    ids = set(ids)

    # loop through file and save comments with id=id with probability p
    sample = []
    with bz2.BZ2File(f'{path}RC_{year}-{month}.bz2', 'rb') as f:
        for line in f:
            entry = json.loads(line)
            if entry['id'] in ids:
                sample.append(entry['body'])
    
    return sample

def load_npz(path, year, month, key):
    with open(f'{path}{key}/{key}_{year}-{month}.npz', 'rb') as f:
        array = np.load(f)[key]
    return array

def load_cc(name, base_path='/sciclone/geograd/stmorse/reddit/'):
    with open(f'{base_path}{name}.npz', 'rb') as f:
        cluster_centers = np.load(f)['cc']
    return cluster_centers

def load_metadata(path, year, month):
    with open(f'{path}metadata/metadata_{year}-{month}.pkl', 'rb') as f:
        metadata = pickle.load(f)
        metadata = pd.DataFrame(metadata, columns=['k', 'author', 'id', 'utc'])
    return metadata

def sample_cluster_sentences_and_embeddings(
        coi,            # cluster of interest
        sample_rate, 
        years=[2007], 
        months=['01'], 
        base_path='/sciclone/geograd/stmorse/reddit/',
        label_path='/sciclone/geograd/stmorse/reddit/',
        data_path='/sciclone/data10/twford/reddit/reddit/comments/',
        seed=123
    ):

    # load sample of sentence + author from across all files
    sample_sentences = []
    sample_embeddings = []
    t0 = time.time()
    print(f'Loading ... ')
    for year, month in [(yr, mo) for yr in years for mo in months]:    
        # get indices of this label
        labels = load_npz(label_path, year, month, 'labels')
        idx = np.where(labels==coi)[0]
        del labels

        # get corresponding metadata
        metadata = load_metadata(base_path, year, month)
        ids = metadata.iloc[idx, 2]
        
        print(f'> {year}-{month}: cluster size {len(ids)}, sampling ... ({time.time()-t0:.2f})')

        # choosing sample idx

        # create RNG with seed
        rng = np.random.default_rng(seed=seed)

        # sample_idx is indices to idx (sorted)
        sample_idx = rng.choice(len(idx), size=int(sample_rate * len(idx)), replace=False)
        sample_idx = np.sort(sample_idx)

        # sampled ids within ids
        sample_ids = ids.iloc[sample_idx]

        # sample sentences from these ids
        sample = load_sample_from_file(data_path, year, month, sample_ids)
        sample_sentences.extend(sample)
        del sample

        # load corresponding embeddings
        # note we need to use total idx, but only sampled ones
        embeddings = load_npz(base_path, year, month, 'embeddings')[idx[sample_idx]]
        sample_embeddings.append(embeddings)
        print(f'> Sampled (s: {len(sample_sentences)}) (e: {embeddings.shape}) ({time.time()-t0:.2f})')
        del embeddings

        gc.collect()

    sample_embeddings = np.vstack(sample_embeddings)
    
    print(f'COMPLETE: {len(sample_sentences)}, {sample_embeddings.shape}')

    return sample_sentences, sample_embeddings

def sample_cluster_embeddings(
        coi,            # cluster of interest
        sample_rate, 
        years=[2007], 
        months=['01'], 
        base_path='/sciclone/geograd/stmorse/reddit/',
        label_path='/sciclone/geograd/stmorse/reddit/',
        seed=123
    ):

    # load sample of embeddings for cluster `coi` from across all files
    sample_embeddings = []
    sample_indices = []
    t0 = time.time()
    print(f'Loading ...')
    for year, month in [(yr, mo) for yr in years for mo in months]:
        # get indices of this label
        labels = load_npz(label_path, year, month, 'labels')
        idx = np.where(labels==coi)[0]
        L = len(idx)
        del labels

        # load embeddings
        embeddings = load_npz(base_path, year, month, 'embeddings')

        print(f'> {year}-{month}: Sampling {int(sample_rate * L)} from {L} ... ({time.time()-t0:.2f})')

        # get sample indices

        # create RNG with seed
        rng = np.random.default_rng(seed=seed)

        # sample_idx is indices to idx (sorted)
        sample_idx = rng.choice(L, size=int(sample_rate * L), replace=False)
        sample_idx = np.sort(sample_idx)

        embeddings = load_npz(base_path, year, month, 'embeddings')[idx[sample_idx]]
        sample_embeddings.append(embeddings)
        sample_indices.append(idx[sample_idx])
        print(f'> Sampled (e: {embeddings.shape}) ({time.time()-t0:.2f})')
        del embeddings

        gc.collect()

    sample_embeddings = np.vstack(sample_embeddings)
    sample_indices = np.concatenate(sample_indices)
    
    print(f'COMPLETE: {sample_embeddings.shape}, {sample_indices.shape}')

    return sample_embeddings, sample_indices

def sample_embeddings_and_labels(
        sample_rate,
        years=[2007],
        months=['01'],
        base_path='/sciclone/geograd/stmorse/reddit/',
        label_path='/sciclone/geograd/stmorse/reddit/',
        seed=123
    ):

    # load sample of sentence + author from across all files
    sample_embeddings = []
    sample_labels = []
    t0 = time.time()
    print(f'Loading ...')
    for year, month in [(yr, mo) for yr in years for mo in months]:
        # load labels
        labels = load_npz(label_path, year, month, 'labels')
        L = labels.shape[0]

        # load embeddings
        embeddings = load_npz(base_path, year, month, 'embeddings')

        print(f'> {year}-{month}: Sampling {int(sample_rate * L)} from {L} ... ({time.time()-t0:.2f})')

        # get sample indices

        # create RNG with seed
        rng = np.random.default_rng(seed=seed)

        # sample_idx is indices to idx (sorted)
        sample_idx = rng.choice(L, size=int(sample_rate * L), replace=False)
        sample_idx = np.sort(sample_idx)

        labels = labels[sample_idx]
        embeddings = embeddings[sample_idx]

        sample_embeddings.append(embeddings)
        sample_labels.append(labels)

        del labels
        del embeddings
        gc.collect()

    sample_embeddings = np.vstack(sample_embeddings)
    sample_labels = np.hstack(sample_labels).flatten()

    print(f'Complete.  Labels: {sample_labels.shape}, Embeds: {sample_embeddings.shape}')

    return sample_embeddings, sample_labels

def get_closest_vectors(query_vector, embeddings, top_k=10):
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    closest_points = np.argsort(distances)[:top_k]  # args to top distances
    return closest_points

def get_closest_vectors_from_subset(query_vector, embeddings, idx, top_k=10):
    distances = np.linalg.norm(embeddings[idx] - query_vector, axis=1)
    closest_points = np.argsort(distances)[:top_k]  # args to top distances
    return idx[closest_points]

def get_clusters_for_term(
        term,
        top_k=2,
        years=[2007],
        month='03',   # will just check a given month as a hacky speedup
        label_path='/sciclone/geograd/stmorse/reddit/',
        data_path='/sciclone/data10/twford/reddit/reddit/comments/',
        model_name='mbkm_cc_20_2007_2011'
    ):

    sel_clusters = []           # will hold cluster # corresponding to centroids
    sel_counts = []             # will hold counts in that cluster (doc freq)
    centroids = []              # will hold all centroids for these clusters
    year_key = []               # keeps track of year corresponding to centroid

    for year in years:
        print(f'Processing {year} ...')

        # grab indices of this term in a given year
        with bz2.BZ2File(f'{data_path}RC_{year}-{month}.bz2', 'rb') as f:
            idx = []
            k = 0
            for line in f:
                entry = json.loads(line)
                if 'body' not in entry or entry['author'] == '[deleted]':
                    continue
                
                if term in entry['body']:
                    idx.append(k)

                k += 1

        idx = np.array(idx)
        print(idx.shape)

        # grab labels for this term
        labs = load_npz(label_path, year, month, 'labels')

        # look at cluster (aka document) frequency
        cluster, count = np.unique(labs[idx], return_counts=True)
        top_c_idx = np.argsort(count)[::-1][:top_k]
        # print(f'All: {[(cl, co) for cl, co in zip(cluster, count)]}')
        print(f'Top: ', cluster[top_c_idx], count[top_c_idx])
        sel_clusters.append(cluster[top_c_idx])
        sel_counts.append(count[top_c_idx])

        # grab centroids corresponding to these top clusters
        cluster_centers = load_cc(model_name, base_path=label_path)
        for i in range(top_k):
            t = cluster[top_c_idx[i]]
            centroids.append(cluster_centers[t])
            year_key.append(year)

    centroids = np.vstack(centroids)
    sel_counts = np.vstack(sel_counts)
    sel_clusters = np.vstack(sel_clusters)

    return centroids, sel_counts, sel_clusters, year_key

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

