import torch
import numpy as np
from skimage import color
from .torch_kmeans import TorchKMeans

class LABTorchKMeans(TorchKMeans):
    def __init__(self, n_clusters, max_iter=300, random_state=None, batch_size=None):
        super().__init__(n_clusters, max_iter, random_state, batch_size)
        
    def rgb_to_lab_tensor(self, rgb_tensor):
        """RGB tensorをLAB tensorに変換"""
        # GPUメモリを節約するためにバッチ処理を行う
        batch_size = 10000 if self.batch_size is None else self.batch_size
        batches = []
        
        for i in range(0, rgb_tensor.shape[0], batch_size):
            batch = rgb_tensor[i:i + batch_size].cpu().numpy()
            lab_batch = color.rgb2lab(batch)
            batches.append(torch.from_numpy(lab_batch).float().to(self.device))
            
        return torch.cat(batches, dim=0)
    
    def lab_to_rgb_tensor(self, lab_tensor):
        """LAB tensorをRGB tensorに変換"""
        batch_size = 10000 if self.batch_size is None else self.batch_size
        batches = []
        
        for i in range(0, lab_tensor.shape[0], batch_size):
            batch = lab_tensor[i:i + batch_size].cpu().numpy()
            rgb_batch = color.lab2rgb(batch)
            batches.append(torch.from_numpy(rgb_batch).float().to(self.device))
            
        return torch.cat(batches, dim=0)
    
    def fit_rgb(self, rgb_array):
        """RGB配列を直接受け取り、LAB空間でクラスタリングを行う"""
        if isinstance(rgb_array, np.ndarray):
            rgb_tensor = torch.from_numpy(rgb_array).float().to(self.device)
        else:
            rgb_tensor = rgb_array.to(self.device)
        
        # RGB -> LAB変換
        lab_tensor = self.rgb_to_lab_tensor(rgb_tensor)
        
        # LAB空間でクラスタリング
        self.fit(lab_tensor)
        
        # 中心点をRGB空間に戻す
        self.cluster_centers_rgb = color.lab2rgb(self.cluster_centers_.reshape(1, -1, 3)).reshape(-1, 3)
        
        return self

    def predict_rgb(self, rgb_array):
        """RGB配列に対して予測を行う"""
        if isinstance(rgb_array, np.ndarray):
            rgb_tensor = torch.from_numpy(rgb_array).float().to(self.device)
        else:
            rgb_tensor = rgb_array.to(self.device)
        
        lab_tensor = self.rgb_to_lab_tensor(rgb_tensor)
        return self.predict(lab_tensor)
