# Copyright (c) 2024 Sousakujikken HIRO
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from skimage import color
from .kmeans.torch_kmeans import TorchKMeans
from .kmeans.utils import estimate_optimal_batch_size
import math

COLOR_REDUCTION_METHODS = ["K-Means", "MedianCut", "Pillow Quantize"]

class PixydustQuantize1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_reduction_method": (COLOR_REDUCTION_METHODS,),
                "max_colors": ([2,4,8,16,32,64,96,128,192,256], {"default": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "PALETTE")
    RETURN_NAMES = ("Reduced Color Image", "Palette Preview", "Palette Tensor")
    FUNCTION = "process_image"
    CATEGORY = "image/Pixydust Quantizer🧚✨"
    OUTPUT_NODE = True


    def process_image(self, image, color_reduction_method, max_colors):
        pil_image = self.tensor_to_pil(image)
        
        reduced_image, final_palette_colors = self.reduce_colors_fixed_palette(
            pil_image, color_reduction_method, max_colors
        )
        
        palette_preview = self.create_palette_image(final_palette_colors)
        
        palette_tensor = torch.tensor(final_palette_colors).float() / 255.0
        
        return (
            self.pil_to_tensor(reduced_image),
            self.pil_to_tensor(palette_preview),
            palette_tensor
        )

    def reduce_colors_fixed_palette(self, image, method, max_colors):
        if method == "Pillow Quantize":
            return self.pillow_quantize(image, max_colors)
        elif method == "K-Means":
            return self.kmeans_quantize(image, max_colors)
        elif method == "MedianCut":
            return self.median_cut_quantize(image, max_colors)
        else:
            raise ValueError(f"Unknown color reduction method: {method}")


    def tensor_to_pil(self, tensor):
        return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def pil_to_tensor(self, pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)


    def create_palette_image(self, colors, tile_size=(16, 16)):
        rows = int(np.ceil(len(colors) / 16))
        img = Image.new('RGB', (16 * tile_size[0], rows * tile_size[1]), color='white')
        draw = ImageDraw.Draw(img)
        for i, color in enumerate(colors):
            x = (i % 16) * tile_size[0]
            y = (i // 16) * tile_size[1]
            draw.rectangle([x, y, x + tile_size[0], y + tile_size[1]], fill=color)
        return img

    def pillow_quantize(self, image, max_colors):
        reduced_image = image.quantize(colors=max_colors, method=Image.MEDIANCUT)
        palette = reduced_image.getpalette()[:max_colors*3]
        final_palette_colors = [tuple(palette[i:i+3]) for i in range(0, len(palette), 3)]
        return reduced_image.convert('RGB'), final_palette_colors

    # def kmeans_quantize(self, image, max_colors):
    #     pixels = np.array(image).reshape(-1, 3)
    #     kmeans = KMeans(n_clusters=max_colors, random_state=42)
    #     labels = kmeans.fit_predict(pixels)
    #     centers = kmeans.cluster_centers_
    #     reduced_image = centers[labels].reshape(image.size[1], image.size[0], 3).astype(np.uint8)
    #     final_palette_colors = [tuple(map(int, color)) for color in centers]
    #     return Image.fromarray(reduced_image), final_palette_colors
    
    def kmeans_quantize(self, image, max_colors):
        pixels = np.array(image).reshape(-1, 3)
        
        # 最適なバッチサイズを推定
        batch_size = estimate_optimal_batch_size(image.size, max_colors, torch.device('cuda'))
        
        # GPUベースのK-means使用
        kmeans = TorchKMeans(n_clusters=max_colors, random_state=42, batch_size=batch_size)
        
        # デバイス情報をログ出力
        device_info = kmeans.get_device_info()
        print("K-means clustering device info:", device_info)
        
        # クラスタリング実行
        labels = kmeans.fit(pixels).labels_
        centers = kmeans.cluster_centers_
        
        # 結果を画像形式に再構成
        reduced_image = centers[labels].reshape(image.size[1], image.size[0], 3).astype(np.uint8)
        final_palette_colors = [tuple(map(int, color)) for color in centers]
        
        return Image.fromarray(reduced_image), final_palette_colors
    
    def median_cut_quantize(self, image, max_colors):
        def median_cut(colors, depth):
            if len(colors) == 0:
                return []
            if depth == 0 or len(colors) == 1:
                return [np.mean(colors, axis=0).astype(int)]
            
            channel = np.argmax(np.max(colors, axis=0) - np.min(colors, axis=0))
            colors = sorted(colors, key=lambda x: x[channel])
            median = len(colors) // 2
            return (median_cut(colors[:median], depth - 1) +
                    median_cut(colors[median:], depth - 1))

        pixels = np.array(image)
        unique_colors = np.unique(pixels.reshape(-1, 3), axis=0)
        depth = int(np.log2(max_colors))
        palette = median_cut(unique_colors, depth)
        
        # Map each pixel to the closest palette color
        palette_array = np.array(palette)
        pixels_flat = pixels.reshape(-1, 3)
        distances = np.linalg.norm(pixels_flat[:, np.newaxis] - palette_array, axis=2)
        closest_palette_indices = np.argmin(distances, axis=1)
        reduced_pixels = palette_array[closest_palette_indices]
        
        reduced_image = reduced_pixels.reshape(pixels.shape).astype(np.uint8)
        final_palette_colors = [tuple(color) for color in palette]
        return Image.fromarray(reduced_image), final_palette_colors


class PixydustQuantize2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reduced_image": ("IMAGE",),
                "fixed_colors": ([2,4,6,8,12,16,24,32], {"default": 16}),
                "reduction_method": (["K-Means", "MedianCut"],),
                "dither_pattern": (["None", "2x2 Bayer", "4x4 Bayer", "8x8 Bayer"], {"default": "8x8 Bayer"}),
                "color_distance_threshold": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "palette_tensor": ("PALETTE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "PALETTE")
    RETURN_NAMES = ("Optimized Image", "Color Histogram", "Fixed Palette")
    FUNCTION = "optimize_palette"
    CATEGORY = "image/Pixydust Quantizer🧚✨"

    def optimize_palette(self, reduced_image, fixed_colors, reduction_method, dither_pattern, color_distance_threshold, palette_tensor=None):
        # Convert input image to numpy array
        image_np = self.tensor_to_numpy(reduced_image)
        
        # Generate or use provided palette
        if palette_tensor is not None:
            palette_np = (palette_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            palette_np = self.generate_fixed_palette(image_np, fixed_colors, reduction_method)
        
        # Apply optimized dithering
        dithered_image = self.apply_dithering_vectorized(image_np, palette_np, dither_pattern, color_distance_threshold)
        
        # Create color histogram
        histogram_image = self.create_color_histogram(dithered_image)
        
        return (
            self.numpy_to_tensor(dithered_image),
            self.numpy_to_tensor(histogram_image),
            torch.tensor(palette_np).float() / 255.0
        )

    def generate_fixed_palette(self, image_np, fixed_colors, reduction_method):
        # Reshape image to 2D array of pixels
        pixels = image_np.reshape(-1, 3)
        
        # Convert to LAB color space
        with torch.no_grad():
            pixels_tensor = torch.from_numpy(pixels).float().cuda() / 255.0
            lab_pixels = self.rgb_to_lab_batch(pixels_tensor)
        
        if reduction_method == "K-Means":
            kmeans = KMeans(n_clusters=fixed_colors, random_state=42, n_init=1)
            kmeans.fit(lab_pixels.cpu().numpy())
            centers = kmeans.cluster_centers_
        else:  # MedianCut
            centers = self._median_cut_lab(lab_pixels.cpu().numpy(), fixed_colors)
        
        # Convert centers back to RGB
        with torch.no_grad():
            centers_tensor = torch.from_numpy(centers).float().cuda()
            rgb_centers = self.lab_to_rgb_batch(centers_tensor)
        
        return (rgb_centers.cpu().numpy() * 255).astype(np.uint8)

    @torch.no_grad()
    def apply_dithering_vectorized(self, image_np, palette_np, dither_pattern, color_distance_threshold):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        height, width = image_np.shape[:2]
        
        # テンソルに変換し、正規化
        image_tensor = torch.from_numpy(image_np).float().to(device) / 255.0
        palette_tensor = torch.from_numpy(palette_np).float().to(device) / 255.0
        
        # LAB色空間に変換
        image_pixels = image_tensor.reshape(-1, 3)
        image_lab = self.rgb_to_lab_batch(image_pixels)
        palette_lab = self.rgb_to_lab_batch(palette_tensor)
        
        # 各ピクセルに対する全パレット色との距離を計算
        distances = torch.cdist(image_lab, palette_lab)  # shape: [num_pixels, num_palette_colors]
        
        # 最も近い2つのパレット色のインデックスを取得
        values, indices = torch.topk(distances, k=2, largest=False, dim=1)  # [num_pixels, 2]
        
        # c1: 最も近い色, c2: 2番目に近い色
        c1 = palette_lab[indices[:, 0]]  # [num_pixels, 3]
        c2 = palette_lab[indices[:, 1]]  # [num_pixels, 3]
        
        # ピクセルの輝度を取得
        pixel_L = image_lab[:, 0]  # [num_pixels]
        
        # c1とc2の輝度を取得
        c1_L = c1[:, 0]  # [num_pixels]
        c2_L = c2[:, 0]  # [num_pixels]
        
        # c1_Lがc2_L以下となるようにソート
        swap_mask = c1_L > c2_L  # [num_pixels]
        c1_L_sorted = c1_L.clone()
        c2_L_sorted = c2_L.clone()
        c1_L_sorted[swap_mask] = c2_L[swap_mask]
        c2_L_sorted[swap_mask] = c1_L[swap_mask]
        indices_sorted = indices.clone()
        indices_sorted[swap_mask, 0] = indices[swap_mask, 1]
        indices_sorted[swap_mask, 1] = indices[swap_mask, 0]
        
        # ソート後の値を更新
        c1_L = c1_L_sorted
        c2_L = c2_L_sorted
        indices = indices_sorted
        
        # Compute distance between image pixels and c1
        pixel_to_c1_distance = torch.norm(image_lab - c1, dim=1)
        
        # Determine pixels that are close enough (no dithering needed)
        no_dither_mask = pixel_to_c1_distance <= color_distance_threshold  # [num_pixels] bool
        
        # Determine pixels that need dithering
        apply_dither_mask = ~no_dither_mask  # [num_pixels] bool


        # ベイヤー行列の作成とタイリング
        if dither_pattern != "None":
            bayer_matrix = self.create_bayer_matrix_tensor(dither_pattern, device)  # [n, n] in [0,1)
            bayer_tiled = self.tile_bayer_matrix(bayer_matrix, height, width)  # [height, width]
            bayer_flat = bayer_tiled.reshape(-1)  # [num_pixels]
        else:
            bayer_flat = torch.zeros(height * width, device=device)
        
        # 閾値の計算
        # threshold = c1_L + (c2_L - c1_L) * bayer_value
        threshold = c1_L + (c2_L - c1_L) * bayer_flat  # [num_pixels]
        
        # ディザリングマスクの生成
        # pixel_L > threshold なら c2 を選択、それ以外は c1 を選択
        # さらに、ディザリングを適用するピクセルに限定
        dither_mask = (pixel_L > threshold) & apply_dither_mask  # [num_pixels] bool
        
        # Initialize final color indices to c1
        final_color_indices = indices[:, 0].clone()  # [num_pixels]
        
        # For pixels needing dithering and where dither_mask is True, set to c2
        final_color_indices = torch.where(dither_mask, indices[:, 1], final_color_indices)
        
        # For pixels not needing dithering, keep as c1
        # Already set to c1 in final_color_indices
        
        # Get final colors
        final_colors = palette_tensor[final_color_indices]  # [num_pixels, 3]
        
        # Reshape to image
        dithered_image = final_colors.reshape(height, width, 3)
        
        return (dithered_image.cpu().numpy() * 255).astype(np.uint8)


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


    def tensor_to_numpy(self, tensor):
        return (tensor.cpu().numpy()[0] * 255).astype(np.uint8)

    def numpy_to_tensor(self, array):
        return torch.from_numpy(array.astype(np.float32) / 255.0).unsqueeze(0)

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

    # # LAB Projection Dithering methods
    # def _apply_dithering(self, image, palette, dither_pattern, color_distance_threshold):
    #     width, height = image.size
    #     pixels = np.array(image)
    #     lab_pixels = color.rgb2lab(pixels / 255.0)
    #     lab_palette = color.rgb2lab(np.array(palette) / 255.0)
    #     tree = cKDTree(lab_palette)
        
    #     bayer_matrix = self._create_bayer_matrix(dither_pattern)
        
    #     dithered_pixels = np.zeros_like(pixels)
        
    #     for y in range(height):
    #         for x in range(width):
    #             old_color_lab = lab_pixels[y, x]
    #             distances, indices = tree.query(old_color_lab, k=2)
                
    #             if distances[0] <= color_distance_threshold:
    #                 new_color_lab = lab_palette[indices[0]]
    #             elif bayer_matrix is not None:
    #                 color1_lab = lab_palette[indices[0]]
    #                 color2_lab = lab_palette[indices[1]]
                    
    #                 t = self._project_point_to_line_segment(old_color_lab, color1_lab, color2_lab)
                    
    #                 threshold = bayer_matrix[y % bayer_matrix.shape[0], x % bayer_matrix.shape[1]]
                    
    #                 if t < threshold:
    #                     new_color_lab = color1_lab
    #                 else:
    #                     new_color_lab = color2_lab
    #             else:
    #                 new_color_lab = lab_palette[indices[0]]
                
    #             new_color_rgb = color.lab2rgb(new_color_lab.reshape(1, 1, 3)).reshape(3)
    #             dithered_pixels[y, x] = (new_color_rgb * 255).astype(np.uint8)
        
    #     return Image.fromarray(dithered_pixels)

    # def _create_bayer_matrix(self, pattern):
    #     if pattern == "2x2 Bayer":
    #         n = 2
    #     elif pattern == "4x4 Bayer":
    #         n = 4
    #     elif pattern == "8x8 Bayer":
    #         n = 8
    #     else:
    #         return None

    #     base = np.array([[0, 2], [3, 1]])
    #     for _ in range(int(np.log2(n)) - 1):
    #         base = np.block([[4*base, 4*base+2], [4*base+3, 4*base+1]])
    #     return base / (n * n)

    # def _project_point_to_line_segment(self, point, line_start, line_end):
    #     line_vec = line_end - line_start
    #     point_vec = point - line_start
    #     t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
    #     t = np.clip(t, 0, 1)
    #     return t


# Add these to your NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "PixydustQuantize1": PixydustQuantize1,
    "PixydustQuantize2": PixydustQuantize2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixydustQuantize1": "Pixydust Quantize-1",
    "PixydustQuantize2": "Pixydust Quantize-2",
}