import torch
import numpy as np

def estimate_optimal_batch_size(image_size, n_colors, device):
    """画像サイズとクラスター数に基づいて最適なバッチサイズを推定"""
    if not torch.cuda.is_available():
        return None  # CPUの場合はバッチ処理を行わない
        
    # GPUのメモリ容量を取得（単位: バイト）
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    
    # 1サンプルあたりのメモリ使用量を概算（float32を想定）
    bytes_per_sample = 4  # float32
    memory_per_sample = image_size[0] * image_size[1] * 3 * bytes_per_sample
    
    # K-means処理に必要な追加メモリを考慮
    kmeans_overhead = n_colors * 3 * bytes_per_sample
    
    # 利用可能なGPUメモリの70%を使用する想定
    available_memory = gpu_memory * 0.7
    
    # 最適なバッチサイズを計算
    batch_size = int(available_memory / (memory_per_sample + kmeans_overhead))
    
    # 最小バッチサイズは1000とする
    return max(1000, min(batch_size, image_size[0] * image_size[1]))
