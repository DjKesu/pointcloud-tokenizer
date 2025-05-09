# pointcloud-tokenizer

A concise tool to tokenize and reconstruct 3D point clouds using Furthest Point Sampling and K-Means clustering.

## Installation

```bash
pip install torch matplotlib open3d numpy scikit-learn
```

## Usage

**As a script**:
```bash
python tokenizer.py
```

**As a library**:
```python
from tokenizer import Tokenizer

# Initialize with desired number of tokens
tokenizer = Tokenizer(n_tokens=256)

# Tokenize point cloud (numpy array or torch tensor)
tokens, labels = tokenizer.tokenize_with_clustering(pointcloud, n_clusters=128)

# Reconstruct point cloud from tokens
reconstructed = tokenizer.reconstruct_from_tokens(tokens[:, :3], original_points, method='voronoi')

# Compute Chamfer distance for evaluation
chamfer = tokenizer.compute_chamfer_distance(original_points, reconstructed)
```
