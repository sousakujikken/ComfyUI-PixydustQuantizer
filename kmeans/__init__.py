"""
GPU-accelerated K-means clustering implementation for Pixydust Quantizer.
This module provides efficient color quantization using PyTorch for GPU acceleration.
"""

from .torch_kmeans import TorchKMeans
from .utils import estimate_optimal_batch_size
from .lab_kmeans import LABTorchKMeans

# バージョン情報
__version__ = '1.0.0'

# 公開するクラスと関数
__all__ = ['TorchKMeans', 'LABTorchKMeans', 'estimate_optimal_batch_size']

# GPUの利用可能性をチェック
import torch
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"K-means module initialized with GPU support: {device_name}")
else:
    print("K-means module initialized: GPU not available, will use CPU")

def get_module_info():
    """モジュールの情報を返す"""
    return {
        'version': __version__,
        'gpu_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'torch_version': torch.__version__,
    }
