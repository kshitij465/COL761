import numpy as np
import matplotlib.pyplot as plt

# Parameters
dimensions = [1, 2, 4, 8, 16, 32, 64]
n_points = 1000000
n_queries = 100
lower_bound, upper_bound = 0, 1000

# Store average ratios for each distance measure
average_ratios_L1 = []
average_ratios_L2 = []
average_ratios_Linf = []

# Helper function to compute distances
def compute_distances(query, points, distance_type='L2'):
    if distance_type == 'L1':
        distances = np.sum(np.abs(points - query), axis=1)
    elif distance_type == 'L2':
        distances = np.sqrt(np.sum(np.square(points - query), axis=1))
    elif distance_type == 'Linf':
        distances = np.max(np.abs(points - query), axis=1)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")
    return distances

for d in dimensions:
    # Generate dataset
    print(d)
    dataset = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_points, d))
    
    # Randomly select query points indices
    query_indices = np.random.choice(n_points, size=n_queries, replace=False)
    
    ratios_L1 = []
    ratios_L2 = []
    ratios_Linf = []
    
    for idx in query_indices:
        # Select the query point
        query = dataset[idx]
        
        # Exclude the query point from the dataset for distance calculations
        mask = np.arange(n_points) != idx
        modified_dataset = dataset[mask]
        
        # Compute distances
        distances_L1 = compute_distances(query, modified_dataset, 'L1')
        distances_L2 = compute_distances(query, modified_dataset, 'L2')
        distances_Linf = compute_distances(query, modified_dataset, 'Linf')
        
        # Calculate farthest and nearest distances
        farthest_L1, nearest_L1 = np.max(distances_L1), np.min(distances_L1)
        farthest_L2, nearest_L2 = np.max(distances_L2), np.min(distances_L2)
        farthest_Linf, nearest_Linf = np.max(distances_Linf), np.min(distances_Linf)
        
        # Calculate ratios and append to list
        ratios_L1.append(farthest_L1 / nearest_L1)
        ratios_L2.append(farthest_L2 / nearest_L2)
        ratios_Linf.append(farthest_Linf / nearest_Linf)
    
    # Compute average ratios for this dimension and append to results
    average_ratios_L1.append(np.mean(ratios_L1))
    average_ratios_L2.append(np.mean(ratios_L2))
    average_ratios_Linf.append(np.mean(ratios_Linf))

# Plotting the corrected results
plt.figure(figsize=(10, 6))
plt.plot(dimensions, average_ratios_L1, marker='o', label='L1 (Manhattan)')
plt.plot(dimensions, average_ratios_L2, marker='s', label='L2 (Euclidean)')
plt.plot(dimensions, average_ratios_Linf, marker='^', label='L∞ (Chebyshev)')
plt.xlabel('Dimensionality (d)')
plt.ylabel('Average Ratio of Farthest/Nearest Distance')
plt.title('Average Ratio of Farthest to Nearest Distance vs. Dimensionality')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('q1.png')  # Save the figure