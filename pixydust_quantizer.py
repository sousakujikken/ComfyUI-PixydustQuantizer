# Copyright (c) 2024 Sousakujikken HIRO
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from skimage import color
import math

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class Quantizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reduced_image": ("IMAGE",),
                "fixed_colors": ([2,4,6,8,12,16,24,32,64,96,128,192,256], {"default": 16}),
                "reduction_method": (["K-Means", "MedianCut"],),
                "dither_pattern": (["None", "2x2 Bayer", "4x4 Bayer", "8x8 Bayer"], {"default": "8x8 Bayer"}),
                "color_distance_threshold": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "batch_mode": (["All Batches", "Single Batch"], {"default": "All Batches"}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "max_batch_size": ([1, 2, 4, 8, 16, 24, 32, 48, 64], {"default": 4}),
            },
            "optional": {
                "palette_tensor": ("PALETTE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "PALETTE")
    RETURN_NAMES = ("Optimized Image", "Color Histogram", "Fixed Palette")
    FUNCTION = "optimize_palette"
    CATEGORY = "image/Pixydust Quantizer🧚✨"

    def optimize_palette(self, reduced_image, fixed_colors, reduction_method, dither_pattern, 
                        color_distance_threshold, batch_mode, batch_index, max_batch_size, 
                        palette_tensor=None):
        device = get_device()
        print(f"Using device: {device}")
        
        # Convert input image to numpy array
        image_np = self.tensor_to_numpy(reduced_image)
        batch_size = image_np.shape[0]
        
        if batch_mode == "Single Batch":
            if batch_index >= batch_size:
                raise ValueError(f"Batch index {batch_index} is out of range for batch size {batch_size}")
            image_np = image_np[batch_index:batch_index+1]
            batch_size = 1
        
        if palette_tensor is not None:
            print("Using provided palette tensor for all batches")
            palette_np = (palette_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            print("Generating palette from first batch")
            palette_np = self.generate_fixed_palette(image_np[0], fixed_colors, reduction_method, device)
            print("Palette generation complete")

        dithered_images = []
        histogram_images = []
        
        num_iterations = math.ceil(batch_size / max_batch_size)
        for iteration in range(num_iterations):
            start_idx = iteration * max_batch_size
            end_idx = min((iteration + 1) * max_batch_size, batch_size)
            current_batch = image_np[start_idx:end_idx]
            
            print(f"Processing batch {iteration + 1}/{num_iterations} (frames {start_idx}-{end_idx-1})")
            
            for img in current_batch:
                try:
                    dithered = self.apply_dithering_vectorized(
                        img,
                        palette_np,
                        dither_pattern,
                        color_distance_threshold,
                        device
                    )
                    dithered_images.append(dithered)
                    
                    histogram = self.create_color_histogram(dithered)
                    histogram_images.append(histogram)
                except Exception as e:
                    print(f"Error processing image in batch: {e}")
                    # Fallback to CPU processing if device-specific processing fails
                    dithered = self.apply_dithering_cpu(
                        img,
                        palette_np,
                        dither_pattern,
                        color_distance_threshold
                    )
                    dithered_images.append(dithered)
                    histogram = self.create_color_histogram(dithered)
                    histogram_images.append(histogram)
            
            # Clear device cache if available
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
        
        dithered_batch = np.stack(dithered_images, axis=0)
        histogram_batch = np.stack(histogram_images, axis=0)
        
        if batch_mode == "Single Batch":
            if dithered_batch.shape[0] > 1:
                dithered_batch = dithered_batch[0:1]
                histogram_batch = histogram_batch[0:1]
        
        return (
            self.numpy_to_tensor(dithered_batch),
            self.numpy_to_tensor(histogram_batch),
            torch.tensor(palette_np).float() / 255.0
        )

    def apply_dithering_cpu(self, image_np, palette_np, dither_pattern, color_distance_threshold):
        """CPU-based fallback implementation of dithering"""
        height, width = image_np.shape[:2]
        image_pixels = image_np.reshape(-1, 3).astype(np.float32) / 255.0
        palette_pixels = palette_np.astype(np.float32) / 255.0
        
        # Convert to LAB color space using numpy operations
        image_lab = color.rgb2lab(image_pixels.reshape(1, -1, 3)).reshape(-1, 3)
        palette_lab = color.rgb2lab(palette_pixels.reshape(1, -1, 3)).reshape(-1, 3)
        
        # Calculate distances
        distances = np.zeros((len(image_lab), len(palette_lab)))
        for i, pixel in enumerate(image_lab):
            distances[i] = np.sqrt(np.sum((palette_lab - pixel) ** 2, axis=1))
        
        # Get two closest colors
        indices = np.argsort(distances, axis=1)[:, :2]
        nearest_distances = np.take_along_axis(distances, indices, axis=1)
        
        # Initialize output array
        output = np.zeros_like(image_np)
        
        # Apply dithering based on pattern
        if dither_pattern != "None":
            bayer_matrix = self.create_bayer_matrix_numpy(dither_pattern)
            bayer_tiled = np.tile(bayer_matrix, 
                                (math.ceil(height/bayer_matrix.shape[0]),
                                 math.ceil(width/bayer_matrix.shape[1])))[:height, :width]
            
            for y in range(height):
                for x in range(width):
                    pixel_idx = y * width + x
                    if nearest_distances[pixel_idx, 0] <= color_distance_threshold:
                        output[y, x] = palette_np[indices[pixel_idx, 0]]
                    else:
                        threshold = bayer_tiled[y, x]
                        color_idx = indices[pixel_idx, 1] if threshold > 0.5 else indices[pixel_idx, 0]
                        output[y, x] = palette_np[color_idx]
        else:
            # No dithering, just use nearest color
            output = palette_np[indices[:, 0]].reshape(height, width, 3)
            
        return output

    def create_bayer_matrix_numpy(self, pattern):
        """Create Bayer matrix patterns using numpy"""
        if pattern == "2x2 Bayer":
            return np.array([[0, 2], [3, 1]]) / 4.0
        elif pattern == "4x4 Bayer":
            return np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ]) / 16.0
        else:  # "8x8 Bayer"
            return np.array([
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ]) / 64.0

    def generate_fixed_palette(self, image_np, fixed_colors, reduction_method, device):
        try:
            pixels = image_np.reshape(-1, 3)
            pixels_tensor = torch.from_numpy(pixels).float().to(device) / 255.0
            
            if device.type != 'cpu':
                try:
                    lab_pixels = self.rgb_to_lab_batch(pixels_tensor, device=device)
                except Exception as e:
                    print(f"Error in device-specific color conversion: {e}")
                    # Fallback to CPU
                    lab_pixels = color.rgb2lab(pixels.reshape(-1, 3) / 255.0)
            else:
                lab_pixels = color.rgb2lab(pixels.reshape(-1, 3) / 255.0)

            if reduction_method == "K-Means":
                kmeans = KMeans(n_clusters=fixed_colors, random_state=42, n_init=1)
                kmeans.fit(lab_pixels if isinstance(lab_pixels, np.ndarray) else lab_pixels.cpu().numpy())
                centers = kmeans.cluster_centers_
            else:
                centers = self._median_cut_lab(
                    lab_pixels if isinstance(lab_pixels, np.ndarray) else lab_pixels.cpu().numpy(), 
                    fixed_colors
                )

            centers_tensor = torch.from_numpy(centers).float().to(device)
            if device.type != 'cpu':
                try:
                    rgb_centers = self.lab_to_rgb_batch(centers_tensor, device=device)
                except Exception as e:
                    print(f"Error in device-specific color conversion: {e}")
                    # Fallback to CPU
                    rgb_centers = torch.from_numpy(
                        color.lab2rgb(centers.reshape(-1, 3)).reshape(-1, 3)
                    ).float()
            else:
                rgb_centers = torch.from_numpy(
                    color.lab2rgb(centers.reshape(-1, 3)).reshape(-1, 3)
                ).float()

            return (rgb_centers.cpu().numpy() * 255).astype(np.uint8)
        
        except Exception as e:
            print(f"Error in palette generation: {e}")
            # Fallback to simple color reduction
            return self._simple_color_reduction(image_np, fixed_colors)

    @staticmethod
    @torch.no_grad()
    def rgb_to_lab_batch(rgb_tensor, batch_size=10000):
        batches = []
        for i in range(0, rgb_tensor.shape[0], batch_size):
            batch = rgb_tensor[i:i+batch_size]
            lab_batch = torch.tensor(color.rgb2lab(batch.cpu().numpy()), device=rgb_tensor.device)
            batches.append(lab_batch)
        return torch.cat(batches)

    @staticmethod
    @torch.no_grad()
    def lab_to_rgb_batch(lab_tensor, batch_size=10000):
        batches = []
        for i in range(0, lab_tensor.shape[0], batch_size):
            batch = lab_tensor[i:i+batch_size]
            rgb_batch = torch.tensor(color.lab2rgb(batch.cpu().numpy()), device=lab_tensor.device)
            batches.append(rgb_batch)
        return torch.cat(batches)

    def create_bayer_matrix_tensor(self, pattern, device):
        if pattern == "2x2 Bayer":
            base = torch.tensor([
                [0, 2],
                [3, 1]
            ], device=device, dtype=torch.float32) / 4.0
        elif pattern == "4x4 Bayer":
            base = torch.tensor([
                [0,  8,  2,  10],
                [12, 4,  14, 6],
                [3,  11, 1,  9],
                [15, 7,  13, 5]
            ], device=device, dtype=torch.float32) / 16.0
        else:  # "8x8 Bayer"
            base = torch.tensor([
                [ 0, 32,  8, 40,  2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44,  4, 36, 14, 46,  6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [ 3, 35, 11, 43,  1, 33,  9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47,  7, 39, 13, 45,  5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ], device=device, dtype=torch.float32) / 64.0
        
        return base

    def tile_bayer_matrix(self, matrix, height, width):
        """
        ベイヤー行列を画像サイズにタイリング
        """
        matrix_h, matrix_w = matrix.shape
        rows = math.ceil(height / matrix_h)
        cols = math.ceil(width / matrix_w)
        tiled = matrix.repeat(rows, cols)
        return tiled[:height, :width]

    def apply_dithering_vectorized(self, image_np, palette_np, dither_pattern, color_distance_threshold, device):
        height, width = image_np.shape[:2]
        
        # Convert image and palette to GPU tensors
        image_tensor = torch.from_numpy(image_np).float().to(device) / 255.0  # [H,W,3]
        palette_tensor = torch.from_numpy(palette_np).float().to(device) / 255.0  # [P,3]
        
        # Convert to LAB color space
        image_pixels = image_tensor.reshape(-1, 3)  # [H*W,3]
        image_lab = self.rgb_to_lab_batch(image_pixels)  # [H*W,3]
        palette_lab = self.rgb_to_lab_batch(palette_tensor)  # [P,3]
        
        # Calculate distances between each pixel and all palette colors
        distances = torch.cdist(image_lab, palette_lab)  # [H*W,P]
        
        # Get two closest colors and their distances
        distances_topk, indices_topk = torch.topk(distances, k=2, largest=False, dim=1)  # [H*W,2]
        
        # Get actual nearest color index and its distance
        nearest_idx = indices_topk[:, 0]  # [H*W]
        nearest_distance = distances_topk[:, 0]  # [H*W]
        
        # Get c1 and c2 colors in LAB space
        c1 = palette_lab[indices_topk[:, 0]]  # [H*W,3]
        c2 = palette_lab[indices_topk[:, 1]]  # [H*W,3]
        
        # Get luminance values
        pixel_L = image_lab[:, 0]  # [H*W]
        c1_L = c1[:, 0]  # [H*W]
        c2_L = c2[:, 0]  # [H*W]
        
        # Sort c1 and c2 by luminance for dithering
        swap_mask = c1_L > c2_L  # [H*W]
        c1_L_sorted = torch.where(swap_mask, c2_L, c1_L)
        c2_L_sorted = torch.where(swap_mask, c1_L, c2_L)
        indices_sorted = indices_topk.clone()
        indices_sorted[swap_mask, 0] = indices_topk[swap_mask, 1]
        indices_sorted[swap_mask, 1] = indices_topk[swap_mask, 0]
        
        # Create masks for different processing paths
        no_dither_mask = nearest_distance <= color_distance_threshold  # [H*W]
        apply_dither_mask = ~no_dither_mask  # [H*W]
        
        # Generate Bayer matrix threshold for dithering
        if dither_pattern != "None":
            bayer_matrix = self.create_bayer_matrix_tensor(dither_pattern, device)  # [n,n]
            bayer_tiled = self.tile_bayer_matrix(bayer_matrix, height, width)  # [H,W]
            bayer_flat = bayer_tiled.reshape(-1)  # [H*W]
        else:
            bayer_flat = torch.zeros(height * width, device=device)
        
        # Calculate dithering threshold
        threshold = c1_L_sorted + (c2_L_sorted - c1_L_sorted) * bayer_flat  # [H*W]
        dither_mask = (pixel_L > threshold) & apply_dither_mask  # [H*W]
        
        # Initialize final color indices with nearest colors
        final_color_indices = nearest_idx.clone()  # [H*W]
        
        # For pixels needing dithering, choose between sorted c1 and c2
        dither_pixels_mask = apply_dither_mask
        final_color_indices[dither_pixels_mask] = torch.where(
            dither_mask[dither_pixels_mask],
            indices_sorted[dither_pixels_mask, 1],  # Use c2 for pixels above threshold
            indices_sorted[dither_pixels_mask, 0]   # Use c1 for pixels below threshold
        )
        
        # Get final RGB colors
        final_colors = palette_tensor[final_color_indices]  # [H*W,3]
        
        # Reshape to image
        dithered_image = final_colors.reshape(height, width, 3)
        
        return (dithered_image.cpu().numpy() * 255).astype(np.uint8)

    def tensor_to_numpy(self, tensor):
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def numpy_to_tensor(self, array):
        return torch.from_numpy(array.astype(np.float32) / 255.0)

    def create_color_histogram(self, image):
        # NumPy配列をPIL Imageに変換
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        colors = pil_image.getcolors(pil_image.width * pil_image.height)
        colors.sort(reverse=True, key=lambda x: x[0])
        
        num_colors = len(colors)
        bar_width = max(1, 256 // num_colors)
        histogram_width = bar_width * num_colors
        
        histogram = Image.new('RGB', (histogram_width, 100), color='white')
        draw = ImageDraw.Draw(histogram)
        
        max_count = colors[0][0]
        for i, (count, color) in enumerate(colors):
            height = int((count / max_count) * 100)
            x_start = i * bar_width
            x_end = x_start + bar_width
            draw.rectangle([x_start, 100-height, x_end, 100], fill=color)
        
        # If the histogram is wider than 256 pixels, resize it to 256 pixels width
        if histogram_width > 256:
            histogram = histogram.resize((256, 100), Image.LANCZOS)
        
        # Convert to numpy array for tensor conversion
        return np.array(histogram)

    def _median_cut_lab(self, lab_pixels, depth):
        def cut(pixels, depth):
            if depth == 0 or len(pixels) == 0:
                return [np.mean(pixels, axis=0)]
            
            max_dim = np.argmax(np.max(pixels, axis=0) - np.min(pixels, axis=0))
            pixels = pixels[pixels[:, max_dim].argsort()]
            median = len(pixels) // 2
            
            return cut(pixels[:median], depth - 1) + cut(pixels[median:], depth - 1)
        
        return np.array(cut(lab_pixels, int(np.log2(depth))))

    def _simple_color_reduction(self, image_np, fixed_colors):
        """Simple color reduction fallback method"""
        pixels = image_np.reshape(-1, 3)
        kmeans = KMeans(n_clusters=fixed_colors, random_state=42, n_init=1)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_.astype(np.uint8)
        
class MPSCompatibleQuantizer:
    @staticmethod
    def rgb_to_perceptual_weights(rgb_tensor):
        """
        RGB値から知覚的な重みを計算
        YUV変換の輝度成分の係数を利用
        """
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_tensor.device)
        return (rgb_tensor * weights.view(1, 3)).sum(dim=1, keepdim=True)

    @staticmethod
    def calculate_color_distance(colors1, colors2, device):
        """
        知覚的な重みを考慮した色距離の計算
        """
        # RGB各成分の差分を計算
        diff = colors1.unsqueeze(1) - colors2.unsqueeze(0)
        
        # 知覚的な重みを適用
        weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 1, 3)
        weighted_diff = diff * weights
        
        # ユークリッド距離を計算
        return torch.sqrt((weighted_diff ** 2).sum(dim=2))

    def cluster_colors(self, pixels, n_colors, device):
        """
        k-means++に基づく独自のカラークラスタリング実装
        """
        # 初期クラスタ中心をk-means++で選択
        centers = self.kmeans_pp_initialization(pixels, n_colors, device)
        
        max_iterations = 100
        prev_centers = None
        
        for _ in range(max_iterations):
            # 各ピクセルと全クラスタ中心との距離を計算
            distances = self.calculate_color_distance(pixels, centers, device)
            
            # 最も近いクラスタを割り当て
            labels = torch.argmin(distances, dim=1)
            
            # クラスタ中心を更新
            new_centers = torch.zeros_like(centers)
            for i in range(n_colors):
                mask = (labels == i)
                if mask.any():
                    new_centers[i] = pixels[mask].mean(dim=0)
                else:
                    new_centers[i] = centers[i]
            
            # 収束判定
            if prev_centers is not None and torch.allclose(centers, new_centers, rtol=1e-4):
                break
                
            centers = new_centers
            prev_centers = centers
        
        return centers, labels

    def kmeans_pp_initialization(self, pixels, n_colors, device):
        """
        k-means++によるクラスタ中心の初期化
        """
        n_pixels = pixels.shape[0]
        centers = torch.zeros((n_colors, 3), device=device)
        
        # 最初のクラスタ中心をランダムに選択
        first_center_idx = torch.randint(0, n_pixels, (1,))
        centers[0] = pixels[first_center_idx]
        
        # 残りのクラスタ中心を選択
        for i in range(1, n_colors):
            # 各ピクセルと既存クラスタ中心との最小距離を計算
            distances = self.calculate_color_distance(pixels, centers[:i], device)
            min_distances = torch.min(distances, dim=1)[0]
            
            # 距離の二乗を確率として使用
            probabilities = min_distances ** 2
            probabilities /= probabilities.sum()
            
            # 新しいクラスタ中心を選択
            next_center_idx = torch.multinomial(probabilities, 1)
            centers[i] = pixels[next_center_idx]
        
        return centers

    def create_bayer_pattern(self, pattern_type, device):
        """
        ベイヤーパターンの生成（変更なし）
        """
        if pattern_type == "2x2 Bayer":
            matrix = torch.tensor([
                [0, 2],
                [3, 1]
            ], device=device, dtype=torch.float32) / 4.0
        elif pattern_type == "4x4 Bayer":
            matrix = torch.tensor([
                [0,  8,  2,  10],
                [12, 4,  14, 6],
                [3,  11, 1,  9],
                [15, 7,  13, 5]
            ], device=device, dtype=torch.float32) / 16.0
        else:  # 8x8 Bayer
            matrix = torch.tensor([
                [ 0, 32,  8, 40,  2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44,  4, 36, 14, 46,  6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [ 3, 35, 11, 43,  1, 33,  9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47,  7, 39, 13, 45,  5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ], device=device, dtype=torch.float32) / 64.0
            
        return matrix

    def apply_dithering(self, image, palette, dither_pattern, threshold, device):
        """
        ディザリングの適用
        """
        height, width = image.shape[:2]
        image_tensor = torch.from_numpy(image).float().to(device) / 255.0
        palette_tensor = torch.from_numpy(palette).float().to(device) / 255.0
        
        # 画像をピクセル配列に変換
        pixels = image_tensor.reshape(-1, 3)
        
        # 各ピクセルと全パレット色との距離を計算
        distances = self.calculate_color_distance(pixels, palette_tensor, device)
        
        # 2つの最も近い色を取得
        distances_values, indices = torch.topk(distances, k=2, largest=False, dim=1)
        
        # ディザリングパターンの生成
        if dither_pattern != "None":
            bayer_matrix = self.create_bayer_pattern(dither_pattern, device)
            pattern = self.tile_pattern(bayer_matrix, height, width).reshape(-1)
            
            # 知覚的な輝度に基づいて色の順序を決定
            color1 = palette_tensor[indices[:, 0]]
            color2 = palette_tensor[indices[:, 1]]
            lum1 = self.rgb_to_perceptual_weights(color1)
            lum2 = self.rgb_to_perceptual_weights(color2)
            
            # 輝度が低い色を color1 に設定
            swap_mask = lum1 > lum2
            indices[swap_mask] = indices[swap_mask].flip(1)
            
            # ディザリングの適用
            use_second_color = pattern > threshold
            final_indices = torch.where(use_second_color, indices[:, 1], indices[:, 0])
        else:
            final_indices = indices[:, 0]
        
        # 最終的な色の割り当て
        result = palette_tensor[final_indices].reshape(height, width, 3)
        
        return (result.cpu().numpy() * 255).astype(np.uint8)

    def tile_pattern(self, pattern, height, width):
        """
        パターンを画像サイズにタイリング
        """
        ph, pw = pattern.shape
        rows = math.ceil(height / ph)
        cols = math.ceil(width / pw)
        return pattern.repeat(rows, cols)[:height, :width]

# Add these to your NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Quantizer": Quantizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Quantizer": "Pixydust Quantizer",
}