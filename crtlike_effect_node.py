# Copyright (c) 2024 Sousakujikken HIRO
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn.functional as F

class CRTLikeEffectNode:
    kernel_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gaussian_width_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "gaussian_width_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "intensity_scale": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "gaussian_kernel_size": ([5, 7, 9, 11, 13, 15], {"default": 11}),
                "enable_resize": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "resize_pixels": ([128, 192, 256, 320, 384, 448, 512], {"default": 256}),
                "max_batch_size": ([1, 2, 4, 8, 16, 24, 32, 48, 64], {"default": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_crt_effect"
    CATEGORY = "image/Pixydust Quantizer🧚✨"

    def get_gaussian_kernel(self, kernel_size, width_x, width_y, gamma, device, dtype):
        key = (kernel_size, width_x, width_y, gamma)
        if key not in self.kernel_cache:
            x = torch.arange(kernel_size, device=device, dtype=dtype).float()
            y = torch.arange(kernel_size, device=device, dtype=dtype).float()
            Y, X = torch.meshgrid(y, x, indexing='ij')

            distance_squared = (((X - (kernel_size / 2 - 0.5)) / width_x) ** 2 +
                              ((Y - (kernel_size / 2 - 0.5)) / width_y) ** 2)
            gaussian = torch.exp(-distance_squared / 2)
            gaussian = torch.pow(gaussian, 1 / gamma)
            gaussian = gaussian / gaussian.sum()
            gaussian = gaussian.unsqueeze(0).unsqueeze(0)

            self.kernel_cache[key] = gaussian.to(device=device, dtype=dtype)
        return self.kernel_cache[key]

    def process_batch(self, batch, gaussian_kernel, device, intensity_scale, gamma):
        """
        単一のミニバッチを処理。データ型の一貫性を保証。
        """
        # バッチとカーネルを同じデータ型(half)に変換
        batch = batch.half()
        gaussian_kernel = gaussian_kernel.half()  # カーネルもhalf精度に変換

        B, C, H, W = batch.shape

        # 4倍にアップスケール
        upscaled = F.interpolate(batch, scale_factor=4, mode='nearest')
        _, _, H4, W4 = upscaled.shape

        # インデックス生成（bool型のままで問題ない）
        rows = torch.arange(H4, device=device).view(1, 1, H4, 1) % 4
        cols = torch.arange(W4, device=device).view(1, 1, 1, W4) % 4
        rows = rows.expand(B, C, H4, W4)
        cols = cols.expand(B, C, H4, W4)

        # マスク生成（half精度に変換）
        R_mask = ((cols == 0) & (rows < 3)).half()
        G_mask = ((cols == 1) & (rows < 3)).half()
        B_mask = ((cols == 2) & (rows < 3)).half()

        # シフト処理
        shifted = torch.zeros_like(upscaled)
        shifted[:, 0, :, :] = upscaled[:, 0, :, :] * R_mask[:, 0, :, :]
        shifted[:, 1, :, :] = upscaled[:, 1, :, :] * G_mask[:, 1, :, :]
        shifted[:, 2, :, :] = upscaled[:, 2, :, :] * B_mask[:, 2, :, :]

        # 畳み込み適用
        convolved = F.conv2d(shifted, gaussian_kernel, padding=0, groups=C)

        # 後処理
        convolved = convolved * intensity_scale
        convolved = torch.pow(convolved, 1 / gamma)
        convolved = torch.clamp(convolved, 0.0, 1.0)

        # メモリ解放
        del shifted, upscaled, R_mask, G_mask, B_mask
        torch.cuda.empty_cache()

        return convolved.permute(0, 2, 3, 1).float()

    def apply_crt_effect_gpu_optimized(self, image, gaussian_width_x, gaussian_width_y, intensity_scale, gaussian_kernel_size, gamma, max_batch_size):
        """
        バッチサイズを制限してGPUメモリを効率的に使用
        データ型の一貫性を保証
        """
        device = image.device
        B, C, H, W = image.shape
        print(f"\nStarting batch processing: total batch size = {B}")

        # ガウシアンカーネルを準備（まだfloat32のまま）
        gaussian = self.get_gaussian_kernel(gaussian_kernel_size, gaussian_width_x, gaussian_width_y, gamma, device, torch.float32)
        gaussian_kernel = gaussian.repeat(C, 1, 1, 1)

        # バッチを分割して処理
        outputs = []
        for i in range(0, B, max_batch_size):
            batch = image[i:i + max_batch_size]
            current_batch_size = batch.shape[0]
            print(f"\nProcessing mini-batch {i//max_batch_size + 1}: size = {current_batch_size}")
            
            try:
                processed_batch = self.process_batch(
                    batch, gaussian_kernel, device, 
                    intensity_scale, gamma
                )
                print(f"Mini-batch {i//max_batch_size + 1} processed: shape = {processed_batch.shape}")
                
                outputs.append(processed_batch.cpu())
                print(f"Mini-batch {i//max_batch_size + 1} transferred to CPU")
                
                del processed_batch
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM detected, processing images individually in mini-batch {i//max_batch_size + 1}")
                    for j, single_image in enumerate(batch):
                        processed_single = self.process_batch(
                            single_image.unsqueeze(0), 
                            gaussian_kernel, device,
                            intensity_scale, gamma
                        )
                        print(f"Processed individual image {j+1}/{current_batch_size} in mini-batch {i//max_batch_size + 1}")
                        outputs.append(processed_single.cpu())
                        del processed_single
                        torch.cuda.empty_cache()
                else:
                    raise e

        # 全バッチの結果を結合
        final_output = torch.cat(outputs, dim=0)
        print(f"\nAll mini-batches combined: final shape = {final_output.shape}")
        return (final_output,)

    def apply_crt_effect(self, image, gaussian_width_x, gaussian_width_y, intensity_scale, gamma, 
                        gaussian_kernel_size, enable_resize, resize_pixels, max_batch_size):
        """
        メインの処理関数
        """
        print(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        if image.max() > 1.0:
            image = image / 255.0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = image.to(device, dtype=torch.float32)

        B, H, W, C = image.shape
        if C != 3:
            raise ValueError(f"Expected image with 3 channels (RGB), but got {C} channels.")

        # [B, C, H, W]形式に変換
        image = image.permute(0, 3, 1, 2)

        # リサイズ処理
        if enable_resize:
            if H > W:
                new_h, new_w = resize_pixels, int(resize_pixels * W / H)
            else:
                new_h, new_w = int(resize_pixels * H / W), resize_pixels

            try:
                image = F.interpolate(image, size=(new_h, new_w), mode='nearest')
                print(f"After resize: {image.shape}")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # リサイズでメモリ不足の場合、バッチごとに処理
                    resized_batches = []
                    for i in range(0, B, max_batch_size):
                        batch = image[i:i + max_batch_size]
                        resized_batch = F.interpolate(batch, size=(new_h, new_w), mode='nearest')
                        resized_batches.append(resized_batch)
                        del batch
                        torch.cuda.empty_cache()
                    image = torch.cat(resized_batches, dim=0)
                else:
                    raise e
        else:
            print(f"No resizing applied. Shape: {image.shape}")

        # CRT効果の適用
        return self.apply_crt_effect_gpu_optimized(
            image, gaussian_width_x, gaussian_width_y,
            intensity_scale, gaussian_kernel_size, gamma,
            max_batch_size
        )

class XYBlurNode:
    kernel_cache = {}  # キャッシュを追加してカーネル再計算を防ぐ

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigma_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sigma_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_batch_size": ([1, 2, 4, 8, 16, 24, 32], {"default": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_blur"
    CATEGORY = "image/Pixydust Quantizer🧚✨"

    def create_gaussian_kernel(self, kernel_size, sigma_x, sigma_y, device, dtype):
        """
        Generates an anisotropic Gaussian kernel with independent vertical and horizontal components.
        Implements caching for performance optimization.
        """
        cache_key = (kernel_size, sigma_x, sigma_y, dtype)
        if cache_key not in self.kernel_cache:
            # Generate grid coordinates
            x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
            y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
            y, x = torch.meshgrid(y, x, indexing='ij')
            
            # Calculate Gaussian function
            gaussian = torch.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
            
            # Normalize
            gaussian = gaussian / gaussian.sum()
            
            # Cache the kernel
            self.kernel_cache[cache_key] = gaussian.unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        
        return self.kernel_cache[cache_key]

    def calculate_kernel_size(self, sigma_x, sigma_y, max_kernel_size=31):
        """
        Calculates appropriate kernel size based on sigma values.
        """
        max_sigma = max(sigma_x, sigma_y)
        kernel_size = int(6 * max_sigma + 1)
        kernel_size = max(3, min(kernel_size + (kernel_size % 2 == 0), max_kernel_size))
        return kernel_size

    def process_batch(self, batch, kernel, pad_size, intensity, device):
        """
        Process a single batch of images with the blur effect.
        """
        # Ensure correct data type (half precision for GPU efficiency)
        batch = batch.half()
        kernel = kernel.half()

        # Apply convolution
        blurred = F.conv2d(
            batch,
            kernel,
            padding=pad_size,
            groups=batch.shape[1]  # Process each channel independently
        )

        # Apply intensity blending
        blurred = batch * (1 - intensity) + blurred * intensity

        # Clamp values
        blurred = torch.clamp(blurred, 0, 1)

        # Convert back to float32 for output
        return blurred.float()

    def apply_blur_optimized(self, image, sigma_x, sigma_y, intensity, max_batch_size):
        """
        Applies blur effect with optimized batch processing and GPU memory management.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, C, H, W = image.shape
        print(f"\nProcessing total batch size: {B}")

        # Calculate kernel size and create kernel
        kernel_size = self.calculate_kernel_size(sigma_x, sigma_y)
        kernel = self.create_gaussian_kernel(kernel_size, sigma_x, sigma_y, device, torch.float32)
        kernel = kernel.repeat(C, 1, 1, 1)
        pad_size = kernel_size // 2

        # Process batches
        outputs = []
        for i in range(0, B, max_batch_size):
            batch = image[i:i + max_batch_size]
            current_batch_size = batch.shape[0]
            print(f"\nProcessing mini-batch {i//max_batch_size + 1}: size = {current_batch_size}")

            try:
                processed_batch = self.process_batch(
                    batch, kernel, pad_size, intensity, device
                )
                print(f"Mini-batch {i//max_batch_size + 1} processed: shape = {processed_batch.shape}")
                
                # Move processed batch to CPU to free GPU memory
                outputs.append(processed_batch.cpu())
                print(f"Mini-batch {i//max_batch_size + 1} transferred to CPU")
                
                del processed_batch
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM detected, processing images individually in mini-batch {i//max_batch_size + 1}")
                    # Process images one by one if batch processing fails
                    for j, single_image in enumerate(batch):
                        processed_single = self.process_batch(
                            single_image.unsqueeze(0),
                            kernel, pad_size, intensity, device
                        )
                        print(f"Processed individual image {j+1}/{current_batch_size} in mini-batch {i//max_batch_size + 1}")
                        outputs.append(processed_single.cpu())
                        del processed_single
                        torch.cuda.empty_cache()
                else:
                    raise e

        # Combine all processed batches
        final_output = torch.cat(outputs, dim=0)
        print(f"\nAll mini-batches combined: final shape = {final_output.shape}")
        return final_output

    def apply_blur(self, image: torch.Tensor, sigma_x: float, sigma_y: float, 
                  intensity: float, max_batch_size: int):
        """
        Main entry point for the blur effect.
        """
        print(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Normalize input if necessary
        if image.max() > 1.0:
            image = image / 255.0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = image.to(device)

        # Convert from [B, H, W, C] to [B, C, H, W]
        image = image.permute(0, 3, 1, 2)

        # Apply optimized blur processing
        blurred = self.apply_blur_optimized(
            image, sigma_x, sigma_y, intensity, max_batch_size
        )

        # Convert back to [B, H, W, C] format
        blurred = blurred.permute(0, 2, 3, 1)

        # Print debug information
        print(f"After blur: mean={blurred.mean().item():.4f}, "
              f"std={blurred.std().item():.4f}, "
              f"min={blurred.min().item():.4f}, "
              f"max={blurred.max().item():.4f}")

        return (blurred,)
    
# class XYBlurNode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "sigma_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
#                 "sigma_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
#                 "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
#             }
#         }

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "apply_blur"
#     CATEGORY = "image/Pixydust Quantizer🧚✨"

#     def create_gaussian_kernel(self, kernel_size, sigma_x, sigma_y, device, dtype):
#         """
#         Generates an anisotropic Gaussian kernel with independent vertical and horizontal components.
#         """
#         # Generate grid coordinates
#         x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
#         y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
#         y, x = torch.meshgrid(y, x, indexing='ij')
        
#         # Calculate Gaussian function (independent for vertical and horizontal)
#         gaussian = torch.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
        
#         # Normalize so that the sum is 1
#         gaussian = gaussian / gaussian.sum()
        
#         # Adjust the shape of the kernel to [1, 1, kernel_size, kernel_size]
#         return gaussian.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

#     def calculate_kernel_size(self, sigma_x, sigma_y, max_kernel_size=31):
#         """
#         Calculates an appropriate kernel size based on sigma_x and sigma_y.
#         Typically, the kernel size is set to 6 * sigma and rounded to an odd number.
#         """
#         # Calculate kernel size based on the largest sigma
#         max_sigma = max(sigma_x, sigma_y)
#         kernel_size = int(6 * max_sigma + 1)
        
#         # Adjust kernel size to be odd
#         if kernel_size % 2 == 0:
#             kernel_size += 1
        
#         # Set minimum kernel size to 3
#         kernel_size = max(kernel_size, 3)
        
#         # Set maximum kernel size
#         kernel_size = min(kernel_size, max_kernel_size)
        
#         return kernel_size

#     def apply_blur(self, image: torch.Tensor, sigma_x: float, sigma_y: float, intensity: float):
#         """
#         Applies an anisotropic Gaussian blur to the image.
#         The kernel size is automatically calculated based on sigma_x and sigma_y.
#         """
#         device = image.device
#         dtype = image.dtype
#         B, H, W, C = image.shape

#         # Calculate kernel size
#         kernel_size = self.calculate_kernel_size(sigma_x, sigma_y)
#         print(f"Calculated kernel_size: {kernel_size} based on sigma_x: {sigma_x}, sigma_y: {sigma_y}")

#         # Convert to [B, C, H, W] format
#         x = image.permute(0, 3, 1, 2)

#         # Generate Gaussian kernel
#         kernel = self.create_gaussian_kernel(kernel_size, sigma_x, sigma_y, device, dtype)

#         # Repeat the same kernel for each channel
#         kernel = kernel.repeat(C, 1, 1, 1)

#         # Calculate padding size
#         pad_size = kernel_size // 2

#         # Apply convolution operation (independently for each channel)
#         blurred = F.conv2d(
#             x,
#             kernel,
#             padding=pad_size,
#             groups=C  # Process each channel independently
#         )

#         # Apply the intensity parameter
#         # Allows intensity to be in the range [0, 2]
#         # intensity = 1.0 for normal blur
#         # intensity < 1.0 to decrease the blur intensity
#         # intensity > 1.0 to increase the blur intensity
#         blurred = x * (1 - intensity) + blurred * intensity

#         # Convert back to [B, H, W, C] format
#         blurred = blurred.permute(0, 2, 3, 1)

#         # Clamp the values to [0, 1]
#         blurred = torch.clamp(blurred, 0, 1)

#         # Output debug information
#         print(f"After blur: mean={blurred.mean().item():.4f}, std={blurred.std().item():.4f}, min={blurred.min().item():.4f}, max={blurred.max().item():.4f}")

#         # Move the tensor to CPU before returning
#         blurred = blurred.cpu()

#         return (blurred,)

NODE_CLASS_MAPPINGS = {
    "CRTLikeEffectNode": CRTLikeEffectNode,
    "XYBlurNode": XYBlurNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTLikeEffectNode": "CRTLike Effect",
    "XYBlurNode": "XY Blur"
}
