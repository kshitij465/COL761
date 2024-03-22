from __future__ import print_function
import numpy as np
import falconn
import timeit
import time
import math
import sys
from scipy.spatial import distance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def partc_lsg(file_path,d):
    print(d)
    k=5
    number_of_queries = 100
    # we build only 50 tables, increasing this quantity will improve the query time
    # at a cost of slower preprocessing and larger memory footprint, feel free to
    # play with this number
    number_of_tables = 400

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

    # Perform linear scan using NumPy to get answers to the queries.
    print('Solving queries using linear scan')
    t1 = timeit.default_timer()
    answers = []
    for query in queries:
        distances = distance.cdist(dataset, query.reshape(1, -1), 'euclidean').flatten()
        
        # Get the indices of the k smallest distances
        nearest_neighbor_indices = np.argsort(distances)[:k]
        answers.append(nearest_neighbor_indices)
    t2 = timeit.default_timer()
    print('Done')
    print('Linear scan time: {} per query'.format((t2 - t1) / float(
        len(queries))))

    # Center the dataset and the queries: this improves the performance of LSH quite a bit.
    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')

    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(dataset[0])
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    params_cp.l = number_of_tables
    # we set one rotation, since the data is dense enough,
    # for sparse data set it to 2
    params_cp.num_rotations = 2
    params_cp.seed = 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 12
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    # we build 18-bit hashes so that each table has
    # 2^18 bins; this is a good choise since 2^18 is of the same
    # order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(20, params_cp)

    print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset)
    t2 = timeit.default_timer()
    print('Done')
    print('Construction time: {}'.format(t2 - t1))

    query_object = table.construct_query_object()

    # find the smallest number of probes to achieve accuracy 0.9
    # using the binary search
    print('Choosing number of probes')
    number_of_probes = number_of_tables



    # final evaluation
    times=[]
    score = 0
    for (i, query) in enumerate(queries):
        start_time = time.time()
        arr=query_object.find_k_nearest_neighbors(query,k)
        ttaken = time.time() - start_time
        times.append(ttaken)
        flg=1
        for aa in arr:
            if( aa not in answers[i]):
                flg=0
                break
        score+=flg

    return (np.mean(times)),( np.std(times)),(float(score) / len(queries))
def partd_lsg(file_path,k, d):
    number_of_queries = 100
    # we build only 50 tables, increasing this quantity will improve the query time
    # at a cost of slower preprocessing and larger memory footprint, feel free to
    # play with this number
    number_of_tables = 400

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

    # Perform linear scan using NumPy to get answers to the queries.
    print('Solving queries using linear scan')
    t1 = timeit.default_timer()
    answers = []
    for query in queries:
        distances = distance.cdist(dataset, query.reshape(1, -1), 'euclidean').flatten()
        
        # Get the indices of the k smallest distances
        nearest_neighbor_indices = np.argsort(distances)[:k]
        answers.append(nearest_neighbor_indices)
    t2 = timeit.default_timer()
    print('Done')
    print('Linear scan time: {} per query'.format((t2 - t1) / float(
        len(queries))))

    # Center the dataset and the queries: this improves the performance of LSH quite a bit.
    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')

    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(dataset[0])
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    params_cp.l = number_of_tables
    # we set one rotation, since the data is dense enough,
    # for sparse data set it to 2
    params_cp.num_rotations = 1
    params_cp.seed = 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 12
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    # we build 18-bit hashes so that each table has
    # 2^18 bins; this is a good choise since 2^18 is of the same
    # order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(20, params_cp)

    print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset)
    t2 = timeit.default_timer()
    print('Done')
    print('Construction time: {}'.format(t2 - t1))

    query_object = table.construct_query_object()

    # find the smallest number of probes to achieve accuracy 0.9
    # using the binary search
    print('Choosing number of probes')
    number_of_probes = number_of_tables



    # final evaluation
    times=[]
    score = 0
    for (i, query) in enumerate(queries):
        start_time = time.time()
        arr=query_object.find_k_nearest_neighbors(query,k)
        ttaken = time.time() - start_time
        times.append(ttaken)
        flg=1
        for aa in arr:
            if( aa not in answers[i]):
                flg=0
                break
        score+=flg

    return (np.mean(times)),( np.std(times)),(float(score) / len(queries))
data_path = sys.argv[1]

if(sys.argv[2]=="c"):
    dimensions=[2,4,10,20]
    mean=[]
    accuracy=[]
    st=[]
    for d in dimensions:
        a,b,c=partc_lsg(data_path,d)
        mean.append(a)
        st.append(b)
        accuracy.append(c)
    print(mean)
    print(st)
    print(accuracy)
    plt.figure(figsize=(12, 6))
    plt.errorbar(dimensions, mean, yerr=st, label='Mean Running Time', fmt='-o', capsize=5)
    plt.xlabel('Dimension')
    plt.ylabel('Average Running Time (seconds)')
    plt.title('Average Running Time of 5-NN Query with Standard Deviation')
    plt.legend()
    plt.grid(True)
    plt.savefig('LSH_runtime.png')  # Save the figure

    plt.figure(figsize=(12, 6))
    plt.plot(dimensions, accuracy, '-o', label='LSH Accuracy (Jaccard)')
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy (Jaccard Index)')
    plt.title('Jaccard Accuracy of LSH against Dimension')
    plt.legend()
    plt.grid(True)
    plt.savefig('lsh_accuracy_against_dimension.png')  # Save the figure



if(sys.argv[2]=="d"):

    dims = [2, 4, 10, 20]
    for dim in dims:
        mean=[]
        accuracy=[]
        st=[]
        k=[1, 5, 10, 50, 100,500]
        for a in k:
            a,b,c=partd_lsg(data_path,a, dim)
            mean.append(a)
            st.append(b)
            accuracy.append(c)
        print(mean)
        print(st)
        print(accuracy)


        plt.figure(figsize=(12, 6))
        plt.plot(k, accuracy, '-o', label='LSH Accuracy (Jaccard)')
        plt.xlabel('K value')
        plt.ylabel('Accuracy (Jaccard Index)')
        plt.title('Jaccard Accuracy of LSH against K value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'lsh_accuracy_against_k_for_dim_{dim}.png')  # Save the figure