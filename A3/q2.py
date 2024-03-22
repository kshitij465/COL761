import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from sklearn.neighbors import KDTree
from mtree import MTree
from scipy.spatial.distance import euclidean
# from falconn import LSHIndex

from sklearn.decomposition import PCA

def reduce_dimensions(data, target_dimensions):
    pca = PCA(n_components=target_dimensions, random_state=42)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def build_kdtree(data):
    return KDTree(data, leaf_size = 50)

def build_mtree(data):
    mtree = MTree(euclidean, max_node_size=50)
    for dat in data:
        mtree.add(tuple(dat))
    return mtree



# def build_lsh(data, num_tables=1):
#     num_projections = 10  # Number of projections per table
#     lsh = LSHIndex(data.shape[1], num_tables, num_projections)
#     lsh.setup(data)
#     return lsh

def knn_query_kdtree(kdtree, query_point, k):
    dists, indices = kdtree.query([query_point], k)
    return dists, indices

# def knn_query_mtree(mtree, query_point, k):
#     dists, indices = mtree.query([query_point], k)
#     return dists, indices

# def knn_query_lsh(lsh, query_point, k):
#     return lsh.find_k_nearest_neighbors(query_point, k)

def build_tree(data, mode):
    if(mode == "kdtree"):
        return build_kdtree(data)
    elif(mode == "mtree"):
        return build_mtree(data)
def knn_query(tree, query_point, k, mode):
    if(mode == "kdtree"):
        _, indices = tree.query([query_point], k)
        return indices
    elif(mode == "mtree"):
        return tree.search(query_point, k=k)


def knn_query_and_time(data, query_points, k, mode):
    times = []
    results = []

    tree = build_tree(data, mode)

    for query_point in query_points:
        start_time = time.time()
        res = knn_query(tree, query_point, k, mode)
        ttaken = time.time() - start_time
        times.append(ttaken)
    with open('array.txt', 'a') as f:
        np.savetxt(f, times)
    return np.mean(times), np.std(times)

def plot_running_time(dimensions, running_times, errors, mode):
    plt.errorbar(dimensions, running_times, yerr=errors, fmt='-o')
    plt.xlabel('Dimensions')
    plt.ylabel('Average Running Time (s)')
    plt.title(f'Average Running Time of 5-NN Query vs. Dimension for {mode}')
    plt.savefig(f'q1_c_{mode}.png')  # Save the figure

def sample_query_points(data, num_query_points):
    indices = np.random.choice(data.shape[0], num_query_points, replace=False)
    return data[indices]


dimensions = [2, 4, 10, 20]
# Replace 'file_path.txt' with the path to your data file
file_path = sys.argv[1]

# Load data into a NumPy array
data = np.loadtxt(file_path)
print("data_loaded")

modes = ["kdtree","mtree"]
for mode in modes:

    # Perform k-NN query and measure time
    running_times = []
    errors = []
    accuracies = []
    for dim in dimensions:
        print("for dim = ", dim)
        data_reduced = reduce_dimensions(data, dim)

        # Sample query points
        num_query_points = 100
        query_points = sample_query_points(data_reduced, num_query_points)
        mean_time, std_time = knn_query_and_time(data_reduced, query_points, 5, mode)
        running_times.append(mean_time)
        errors.append(std_time)

    # Plotting
    plot_running_time(dimensions, running_times, errors, mode)
