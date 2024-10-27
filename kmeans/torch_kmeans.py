import torch
import numpy as np

class TorchKMeans:
    def __init__(self, n_clusters, max_iter=300, random_state=None, batch_size=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"TorchKMeans will use device: {self.device}")
        
    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        # Set random seed if specified
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        # Initialize centroids randomly
        n_samples = X.shape[0]
        idx = torch.randperm(n_samples)[:self.n_clusters]
        self.cluster_centers_ = X[idx].clone()
        
        for iteration in range(self.max_iter):
            if self.batch_size is None:
                # Process all data at once
                distances = torch.cdist(X, self.cluster_centers_)
                labels = torch.argmin(distances, dim=1)
            else:
                # Process data in batches
                labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)
                for i in range(0, n_samples, self.batch_size):
                    batch = X[i:i + self.batch_size]
                    distances = torch.cdist(batch, self.cluster_centers_)
                    labels[i:i + self.batch_size] = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = X[mask].mean(dim=0)
                else:
                    new_centroids[k] = self.cluster_centers_[k]
            
            # Check for convergence
            if torch.allclose(new_centroids, self.cluster_centers_):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.cluster_centers_ = new_centroids
            
        self.labels_ = labels.cpu().numpy()
        self.cluster_centers_ = self.cluster_centers_.cpu().numpy()
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        if self.batch_size is None:
            distances = torch.cdist(X, torch.from_numpy(self.cluster_centers_).to(self.device))
            labels = torch.argmin(distances, dim=1)
        else:
            n_samples = X.shape[0]
            labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)
            centroids = torch.from_numpy(self.cluster_centers_).to(self.device)
            
            for i in range(0, n_samples, self.batch_size):
                batch = X[i:i + self.batch_size]
                distances = torch.cdist(batch, centroids)
                labels[i:i + self.batch_size] = torch.argmin(distances, dim=1)
                
        return labels.cpu().numpy()

    def get_device_info(self):
        """デバイス情報とGPUメモリ使用状況を返す"""
        device_info = {
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            device_info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0) / 1024**2,  # MB
                'memory_cached': torch.cuda.memory_reserved(0) / 1024**2,  # MB
            })
            
        return device_info
