# Copyright (c) 2024 Sousakujikken HIRO
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from skimage import color


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

    def kmeans_quantize(self, image, max_colors):
        pixels = np.array(image).reshape(-1, 3)
        kmeans = KMeans(n_clusters=max_colors, random_state=42)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
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
        pil_reduced = self.tensor_to_pil(reduced_image)
        
        if palette_tensor is not None:
            fixed_palette = (palette_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            fixed_palette = self.generate_fixed_palette(pil_reduced, fixed_colors, reduction_method)
        
        optimized_image = self._apply_dithering(pil_reduced, fixed_palette, dither_pattern, color_distance_threshold)
        color_histogram = self.create_color_histogram(optimized_image)
        
        return (
            self.pil_to_tensor(optimized_image),
            self.pil_to_tensor(color_histogram),
            torch.tensor(fixed_palette).float() / 255.0
        )

    def tensor_to_pil(self, tensor):
        return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def pil_to_tensor(self, pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def generate_fixed_palette(self, image, fixed_colors, reduction_method):
        pixels = np.array(image)
        lab_pixels = color.rgb2lab(pixels.reshape(-1, 3) / 255.0)
        
        if reduction_method == "K-Means":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=fixed_colors, random_state=42)
            kmeans.fit(lab_pixels)
            center_colors = kmeans.cluster_centers_
        else:  # MedianCut
            center_colors = self._median_cut_lab(lab_pixels, fixed_colors)
        
        rgb_colors = color.lab2rgb(center_colors.reshape(1, -1, 3)).reshape(-1, 3)
        return (rgb_colors * 255).astype(np.uint8)

    def create_color_histogram(self, image):
        colors = image.getcolors(image.width * image.height)
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
        
        return histogram

    def _median_cut_lab(self, lab_pixels, depth):
        def cut(pixels, depth):
            if depth == 0 or len(pixels) == 0:
                return [np.mean(pixels, axis=0)]
            
            max_dim = np.argmax(np.max(pixels, axis=0) - np.min(pixels, axis=0))
            pixels = pixels[pixels[:, max_dim].argsort()]
            median = len(pixels) // 2
            
            return cut(pixels[:median], depth - 1) + cut(pixels[median:], depth - 1)
        
        return np.array(cut(lab_pixels, int(np.log2(depth))))

    # LAB Projection Dithering methods
    def _apply_dithering(self, image, palette, dither_pattern, color_distance_threshold):
        width, height = image.size
        pixels = np.array(image)
        lab_pixels = color.rgb2lab(pixels / 255.0)
        lab_palette = color.rgb2lab(np.array(palette) / 255.0)
        tree = cKDTree(lab_palette)
        
        bayer_matrix = self._create_bayer_matrix(dither_pattern)
        
        dithered_pixels = np.zeros_like(pixels)
        
        for y in range(height):
            for x in range(width):
                old_color_lab = lab_pixels[y, x]
                distances, indices = tree.query(old_color_lab, k=2)
                
                if distances[0] <= color_distance_threshold:
                    new_color_lab = lab_palette[indices[0]]
                elif bayer_matrix is not None:
                    color1_lab = lab_palette[indices[0]]
                    color2_lab = lab_palette[indices[1]]
                    
                    t = self._project_point_to_line_segment(old_color_lab, color1_lab, color2_lab)
                    
                    threshold = bayer_matrix[y % bayer_matrix.shape[0], x % bayer_matrix.shape[1]]
                    
                    if t < threshold:
                        new_color_lab = color1_lab
                    else:
                        new_color_lab = color2_lab
                else:
                    new_color_lab = lab_palette[indices[0]]
                
                new_color_rgb = color.lab2rgb(new_color_lab.reshape(1, 1, 3)).reshape(3)
                dithered_pixels[y, x] = (new_color_rgb * 255).astype(np.uint8)
        
        return Image.fromarray(dithered_pixels)

    def _create_bayer_matrix(self, pattern):
        if pattern == "2x2 Bayer":
            n = 2
        elif pattern == "4x4 Bayer":
            n = 4
        elif pattern == "8x8 Bayer":
            n = 8
        else:
            return None

        base = np.array([[0, 2], [3, 1]])
        for _ in range(int(np.log2(n)) - 1):
            base = np.block([[4*base, 4*base+2], [4*base+3, 4*base+1]])
        return base / (n * n)

    def _project_point_to_line_segment(self, point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        t = np.clip(t, 0, 1)
        return t


# Add these to your NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "PixydustQuantize1": PixydustQuantize1,
    "PixydustQuantize2": PixydustQuantize2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixydustQuantize1": "Pixydust Quantize-1",
    "PixydustQuantize2": "Pixydust Quantize-2",
}