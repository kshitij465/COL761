from __future__ import print_function
import timeit
import time
import math
from scipy.spatial import distance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import sys
def partc_lsg(file_path,d):
    print(d)
    k=5
    number_of_queries = 100
    # we build only 50 tables, increasing this quantity will improve the query time
    # at a cost of slower preprocessing and larger memory footprint, feel free to
    # play with this number


    print('Reading the dataset')

    # Load data into a NumPy array
    dataset = np.loadtxt(file_path)
    dataset =dataset.astype(np.float32)
    pca = PCA(n_components=d, random_state=42)
    dataset=pca.fit_transform(dataset)
    print('Done')

    # It's important not to use doubles, unless they are strictly necessary.
    # If your dataset consists of doubles, convert it to floats using `astype`.
    assert dataset.dtype == np.float32

    # Normalize all the lenghts, since we care about the cosine similarity.
    print('Normalizing the dataset')
    dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    print('Done')

    # Choose random data points to be queries.
    print('Generating queries')
    np.random.seed(4057218)
    np.random.shuffle(dataset)
    queries = dataset[len(dataset) - number_of_queries:]
    dataset = dataset[:len(dataset) - number_of_queries]
    print('Done')
    times=[]
    # Perform linear scan using NumPy to get answers to the queries.
    print('Solving queries using linear scan')
    for query in queries:
        start_time = time.time()
        distances = distance.cdist(dataset, query.reshape(1, -1), 'euclidean').flatten()
        # Get the indices of the k smallest distances
        nearest_neighbor_indices = np.argsort(distances)[:k]
        ttaken = time.time() - start_time
        times.append(ttaken)
    print('Done')


  
    return (np.mean(times)),( np.std(times))
dimensions=[2,4,10,20]
mean=[]
accuracy=[]
st=[]
for d in dimensions:
    a,b=partc_lsg(sys.argv[1],d)
    mean.append(a)
    st.append(b)
plt.figure(figsize=(12, 6))
plt.errorbar(dimensions, mean, yerr=st, label='Mean Running Time', fmt='-o', capsize=5)
plt.xlabel('Dimension')
plt.ylabel('Average Running Time (seconds)')
plt.title('Average Running Time of 5-NN Query with Standard Deviation')
plt.legend()
plt.grid(True)
plt.savefig('brute_force_runtime.png')  # Save the figure