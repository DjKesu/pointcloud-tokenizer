# tokenizer for pointclouds using Furthest Point Sampling + K-Means to cluster the points and retain the centroids and shape of the clusters
import torch
import matplotlib.pyplot as plt
import time
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans

class Tokenizer:
    def __init__(self, n_tokens=1024, device=None):
        self.n_tokens = n_tokens
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    @staticmethod
    def compute_distances(x, y=None):
        '''
        Compute euclidean distance between all points in x and y
        '''
        if y is None:
            y = x
        
        x_norm = (x**2).sum(1).view(-1,1)
        y_norm = (y**2).sum(1).view(1,-1)
        dist = x_norm + y_norm - 2*torch.mm(x, y.transpose(0,1))
        dist = torch.clamp(dist, 0, float('inf'))
        return dist
    
    def furthest_sampling(self, points, k, initial_idx=None):
        '''
        Farthest Point Sampling
        '''
        n_points = points.shape[0]

        if initial_idx is None:
            initial_idx = torch.randint(0, n_points, (1,)).item()
        
        selected_indices = torch.tensor([initial_idx], device=self.device)

        distances = self.compute_distances(points[selected_indices])

        for _ in range(k - 1):
            if len(selected_indices) == n_points:
                break
            
            mask = torch.ones(n_points, dtype=torch.bool, device=self.device)
            mask[selected_indices] = False

            distances = self.compute_distances(points[selected_indices], points)
            
            min_distances = distances.min(dim=0)[0]
            
            min_distances[selected_indices] = float('-inf')
            
            farthest_idx = torch.argmax(min_distances)
            selected_indices = torch.cat([selected_indices, farthest_idx.unsqueeze(0)])
        
        return selected_indices

    def tokenize_with_clustering(self, pointcloud, n_clusters=None):
        if n_clusters is None:
            n_clusters = int(np.sqrt(self.n_tokens))
        
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud).float()
        
        pointcloud = pointcloud.to(self.device)
        
        indices = self.furthest_sampling(pointcloud[:, :3], n_clusters)
        initial_centroids = pointcloud[indices, :3].cpu().numpy()
        
        points_np = pointcloud[:, :3].cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1)
        cluster_labels = kmeans.fit_predict(points_np)
        centroids = kmeans.cluster_centers_
        
        feature_centroids = []
        for i in range(n_clusters):
            mask = cluster_labels == i
            if np.sum(mask) > 0:
                if pointcloud.shape[1] > 3:
                    features = pointcloud[torch.from_numpy(mask), 3:].mean(dim=0)
                    feature_centroids.append(torch.cat([
                        torch.tensor(centroids[i], device=self.device),
                        features
                    ]))
                else:
                    feature_centroids.append(torch.tensor(centroids[i], device=self.device))
            
        tokens = torch.stack(feature_centroids)
        return tokens, torch.from_numpy(cluster_labels).to(self.device)
        
    @staticmethod
    def visualize_tokenization(original_pc, tokenized_pc, title):
        """
        Visualize the original and tokenized pointclouds.
        
        Args:
            original_pc (numpy.ndarray): Original pointcloud
            tokenized_pc (numpy.ndarray): Tokenized pointcloud
            title (str): Plot title
        """
        fig = plt.figure(figsize=(15, 10))
        
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(original_pc[:, 0], original_pc[:, 1], original_pc[:, 2], 
                  c='blue', s=2, alpha=0.5, label='Original Points')
        
        ax.scatter(tokenized_pc[:, 0], tokenized_pc[:, 1], tokenized_pc[:, 2], 
                  c='red', s=100, alpha=0.8, label=f'Tokens ({len(tokenized_pc)} points)')
        
        ax.set_box_aspect([1, 1, 1])
        
        ax.view_init(elev=20, azim=45)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def compute_chamfer_distance(self, points1, points2):
        '''
        Compute bidirectional Chamfer distance between two point clouds
        '''
        if isinstance(points1, np.ndarray):
            points1 = torch.from_numpy(points1).float()
        if isinstance(points2, np.ndarray):
            points2 = torch.from_numpy(points2).float()
        
        points1 = points1.to(self.device)
        points2 = points2.to(self.device)
        
        dist_matrix = self.compute_distances(points1, points2)
        
        dist1 = torch.min(dist_matrix, dim=1)[0]
        dist2 = torch.min(dist_matrix, dim=0)[0]
        
        chamfer_dist = (torch.mean(dist1) + torch.mean(dist2)) / 2
        return chamfer_dist.item()

    def compute_local_curvature(self, tokens):
        '''
        Estimate local curvature at each token based on neighboring tokens
        '''
        dist_matrix = self.compute_distances(tokens, tokens)
        
        k = min(5, len(tokens)-1)
        _, knn_idx = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)
        knn_idx = knn_idx[:, 1:]
        
        neighbors = tokens[knn_idx]
        
        centered = neighbors - tokens.unsqueeze(1)
        cov = torch.bmm(centered.transpose(1, 2), centered)
        
        eigenvals = torch.linalg.eigvalsh(cov)
        curvature = eigenvals[:, 0] / (eigenvals[:, 2] + 1e-6)
        
        return curvature

    def reconstruct_from_tokens(self, tokens, original_points, method='voronoi', k=3, density_factor=2.0):
        '''
        Reconstruct point cloud from tokens using different methods
        '''
        tokens = tokens.to(self.device)
        n_points = int(len(original_points) * density_factor)
        
        if method == 'voronoi':
            curvature = self.compute_local_curvature(tokens)
            
            curvature = 0.5 + 1.5 * (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-6)
            
            reconstructed_points = []
            total_points = 0
            
            for token_idx, token in enumerate(tokens):
                local_density = curvature[token_idx]
                points_per_token = max(10, int(n_points * local_density / len(tokens)))
                
                theta = torch.rand(points_per_token, device=self.device) * 2 * np.pi
                phi = torch.acos(2 * torch.rand(points_per_token, device=self.device) - 1)
                r = torch.pow(torch.rand(points_per_token, device=self.device), 1/3)
                
                local_points = torch.stack([
                    r * torch.sin(phi) * torch.cos(theta),
                    r * torch.sin(phi) * torch.sin(theta),
                    r * torch.cos(phi)
                ], dim=1)
                
                distances = torch.norm(tokens - token.unsqueeze(0), dim=1)
                min_dist = torch.min(distances[distances > 0])
                scale = min_dist * 0.3
                
                local_points = local_points * scale + token
                
                reconstructed_points.append(local_points)
                total_points += points_per_token
            
            reconstructed = torch.cat(reconstructed_points, dim=0)
            
            token_distances = self.compute_distances(reconstructed, tokens)
            min_token_dist = torch.min(token_distances, dim=1)[0]
            valid_points = min_token_dist < torch.mean(min_token_dist) * 2
            reconstructed = reconstructed[valid_points]
            
        elif method == 'knn':
            base_points = torch.rand(n_points, 3, device=self.device) * 2 - 1
            
            distances = self.compute_distances(base_points, tokens)
            k_nearest_dist, k_nearest_idx = torch.topk(distances, k=k, dim=1, largest=False)
            
            weights = torch.softmax(-k_nearest_dist / 0.1, dim=1).unsqueeze(-1)
            
            selected_tokens = tokens[k_nearest_idx]
            reconstructed = (selected_tokens * weights).sum(dim=1)
            
            local_density = torch.mean(k_nearest_dist, dim=1, keepdim=True)
            noise_scale = local_density * 0.1
            noise = torch.randn_like(reconstructed) * noise_scale
            reconstructed = reconstructed + noise
            
        else:
            n_samples = n_points
            rand_idx1 = torch.randint(0, len(tokens), (n_samples,), device=self.device)
            rand_idx2 = torch.randint(0, len(tokens), (n_samples,), device=self.device)
            rand_idx3 = torch.randint(0, len(tokens), (n_samples,), device=self.device)
            
            weights = torch.rand(n_samples, 3, device=self.device)
            weights = weights / weights.sum(dim=1, keepdim=True)
            
            points1 = tokens[rand_idx1]
            points2 = tokens[rand_idx2]
            points3 = tokens[rand_idx3]
            
            reconstructed = (weights[:, 0:1] * points1 + 
                           weights[:, 1:2] * points2 + 
                           weights[:, 2:3] * points3)
            
            local_density = torch.min(self.compute_distances(reconstructed, tokens), dim=1)[0]
            noise = torch.randn_like(reconstructed) * local_density.unsqueeze(1) * 0.05
            reconstructed = reconstructed + noise
        
        return reconstructed

    @staticmethod
    def visualize_reconstruction(original_pc, tokens, reconstructed_pc, title, chamfer_dist=None):
        """
        Visualize original, tokens, and reconstructed pointclouds.
        """
        fig = plt.figure(figsize=(20, 8))
        
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(original_pc[:, 0], original_pc[:, 1], original_pc[:, 2], 
                   c='blue', s=2, alpha=0.5, label='Original')
        ax1.set_title('Original')
        ax1.set_box_aspect([1,1,1])
        ax1.view_init(elev=20, azim=45)
        
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(tokens[:, 0], tokens[:, 1], tokens[:, 2], 
                   c='red', s=100, alpha=0.8, label=f'Tokens ({len(tokens)})')
        ax2.set_title('Tokens')
        ax2.set_box_aspect([1,1,1])
        ax2.view_init(elev=20, azim=45)
        
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(reconstructed_pc[:, 0], reconstructed_pc[:, 1], reconstructed_pc[:, 2], 
                   c='green', s=2, alpha=0.5, label='Reconstructed')
        if chamfer_dist is not None:
            ax3.set_title(f'Reconstructed (CD: {chamfer_dist:.4f})')
        else:
            ax3.set_title('Reconstructed')
        ax3.set_box_aspect([1,1,1])
        ax3.view_init(elev=20, azim=45)
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def main():
        try:
            shapes = []
            names = []
            
            try:
                print("Loading goat skull pointcloud...")
                mesh = o3d.io.read_triangle_mesh("Goat skull.obj")
                
                if len(np.asarray(mesh.vertices)) == 0:
                    raise Exception("Mesh has no vertices")
                
                pcd = mesh.sample_points_poisson_disk(5000)
                points = np.asarray(pcd.points)
                
                center = np.mean(points, axis=0)
                points = points - center
                scale = np.max(np.abs(points))
                points = points / scale * 5
                
                if mesh.has_vertex_colors():
                    features = np.asarray(mesh.vertex_colors)
                else:
                    features = np.random.rand(len(points), 3)
                
                goat_skull = np.hstack([points, features])
                shapes.append(goat_skull)
                names.append("Goat Skull")
                print(f"Successfully loaded goat skull with {len(points)} points")
            except Exception as e:
                print(f"Failed to load goat skull: {e}")
            
            try:
                print("Creating synthetic taurus shape...")
                n_points = 5000
                points = np.zeros((n_points, 3))
                
                theta = np.random.uniform(0, 2*np.pi, n_points)
                phi = np.random.uniform(0, np.pi, n_points)
                r = np.random.uniform(0, 1, n_points)
                
                a, b, c = 3, 2, 2.5
                points[:, 0] = a * r * np.sin(phi) * np.cos(theta)
                points[:, 1] = b * r * np.sin(phi) * np.sin(theta)
                points[:, 2] = c * r * np.cos(phi)
                
                snout_mask = (points[:, 0] > 0) & (points[:, 1]**2 + points[:, 2]**2 < 1)
                points[snout_mask, 0] += 1.5
                
                horn_mask = (points[:, 0] > 0) & (points[:, 1]**2 + points[:, 2]**2 < 0.5)
                points[horn_mask, 0] += 2
                points[horn_mask, 1] *= 1.5
                
                points += np.random.randn(*points.shape) * 0.1
                
                features = np.random.rand(len(points), 3)
                taurus = np.hstack([points, features])
                shapes.append(taurus)
                names.append("Taurus")
            except Exception as e:
                print(f"Failed to create taurus: {e}")
            
            if not shapes:
                raise Exception("No pointclouds were successfully loaded")
            
            tokenizer = Tokenizer(n_tokens=256)
            
            for i, (pointcloud, name) in enumerate(zip(shapes, names)):
                pc_tensor = torch.from_numpy(pointcloud).float().to(tokenizer.device)
                
                tokens, labels = tokenizer.tokenize_with_clustering(pc_tensor, n_clusters=128)
                
                reconstructed = tokenizer.reconstruct_from_tokens(
                    tokens[:, :3], 
                    pc_tensor[:, :3],
                    method='voronoi',
                    density_factor=4.0
                )
                
                chamfer_dist = tokenizer.compute_chamfer_distance(
                    pc_tensor[:, :3],
                    reconstructed
                )
                
                Tokenizer.visualize_reconstruction(
                    pointcloud[:, :3],
                    tokens.cpu().numpy()[:, :3],
                    reconstructed.cpu().numpy(),
                    f"Enhanced Voronoi Reconstruction of {name}\n({len(reconstructed)} points, CD: {chamfer_dist:.4f})",
                    chamfer_dist
                )
                
                print(f"{name} reconstruction:")
                print(f"- Original points: {len(pointcloud)}")
                print(f"- Reconstructed points: {len(reconstructed)}")
                print(f"- Tokens: {len(tokens)} ({len(tokens)/len(pointcloud)*100:.1f}% of original)")
                print(f"- Chamfer Distance: {chamfer_dist:.4f}")
                print()
        
        except Exception as e:
            print(f"Error processing pointclouds: {e}")
            print("Please ensure the 'Goat skull.obj' file is in the current directory")

if __name__ == "__main__":  
    Tokenizer.main()