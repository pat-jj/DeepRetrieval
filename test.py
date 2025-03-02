import faiss
import numpy as np
import time
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

d = 64  # Vector dimension
nb = 100000  # Size of the dataset
nq = 10000  # Size of the query set
batch_size = 10  # Define the batch size
np.random.seed(0)  # Set random seed

# Check GPU support
if faiss.get_num_gpus() > 0:
    print("Faiss supports GPU")
else:
    print("Faiss does not support GPU")

# Create the dataset and query set
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Use GPU for search
index = faiss.IndexFlatL2(d)  # Use L2 distance
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()  # Use default GPU resources
    index = faiss.index_cpu_to_gpu(res, 0, index)

# Add the dataset to the index
index.add(xb)

# Perform batch search
start_time = time.time()

# Loop through the query set in batches
all_distances = []
all_indices = []

for i in range(0, nq, batch_size):
    batch_queries = xq[i:i + batch_size]  # Slice the batch
    D, I = index.search(batch_queries, 10)  # Search the top 10 nearest neighbors for the batch
    all_distances.append(D)
    all_indices.append(I)

# Concatenate the results from all batches
all_distances = np.concatenate(all_distances, axis=0)
all_indices = np.concatenate(all_indices, axis=0)

end_time = time.time()

# Print search time
print(f"Search time: {end_time - start_time:.4f} seconds")

# You can now use all_distances and all_indices as the search results