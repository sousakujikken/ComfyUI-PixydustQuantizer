# Copyright (c) 2024 Sousakujikken HIRO
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn.functional as F
# from PIL import Image
# import numpy as np

class CRTLikeEffectNode:
    kernel_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gaussian_width_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "gaussian_width_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "intensity_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 5.0, "step": 0.1}),
                "gaussian_kernel_size": ([5, 7, 9, 11, 13, 15], {"default": 11}),
                "enable_resize": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "resize_pixels": ([128, 192, 256, 320, 384, 448, 512], {"default": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_crt_effect"
    CATEGORY = "image/Pixydust Quantizer🧚✨"

    def get_gaussian_kernel(self, kernel_size, width_x, width_y, gamma, device, dtype):
        """
        Retrieves or computes the Gaussian kernel based on the provided parameters.
        Caches the kernel for future use to avoid redundant computations.
        """
        key = (kernel_size, width_x, width_y, gamma)
        if key not in self.kernel_cache:
            # Generate grid coordinates
            x = torch.arange(kernel_size, device=device, dtype=dtype).float()
            y = torch.arange(kernel_size, device=device, dtype=dtype).float()
            Y, X = torch.meshgrid(y, x, indexing='ij')

            # Calculate Gaussian function with separate widths
            distance_squared = (((X - (kernel_size / 2 - 0.5)) / width_x) ** 2 +
                                ((Y - (kernel_size / 2 - 0.5)) / width_y) ** 2)
            gaussian = torch.exp(-distance_squared / 2)

            # Apply gamma correction
            gaussian = torch.pow(gaussian, 1 / gamma)

            # Normalize the kernel
            gaussian = gaussian / gaussian.sum()

            # Reshape to [1, 1, K, K] for convolution
            gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

            self.kernel_cache[key] = gaussian.to(device=device, dtype=dtype)
        return self.kernel_cache[key]

    def apply_crt_effect_gpu_optimized(self, image, gaussian_width_x, gaussian_width_y, intensity_scale, gaussian_kernel_size, gamma):
        """
        Optimized GPU implementation of the CRT-like effect using vectorized operations,
        precomputed Gaussian kernels, and mixed precision.
        """
        # Convert image to float16 for mixed precision
        image = image.half()

        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype  # Should be torch.float16

        # Validate the number of channels
        if C != 3:
            raise ValueError(f"Expected image with 3 channels (RGB), but got {C} channels.")

        # Upscale image by 4x using nearest
        upscaled_image = F.interpolate(image, scale_factor=4, mode='nearest')  # [B, 3, H*4, W*4]
        B, C, H4, W4 = upscaled_image.shape

        # Create row and column indices for 4x4 blocks
        rows = torch.arange(H4, device=device).view(1, 1, H4, 1) % 4  # [1, 1, H4, 1]
        cols = torch.arange(W4, device=device).view(1, 1, 1, W4) % 4  # [1, 1, 1, W4]
        rows = rows.expand(B, C, H4, W4)
        cols = cols.expand(B, C, H4, W4)

        # Create masks for R, G, B channels
        R_mask = ((cols == 0) & (rows < 3)).float()  # [B, C, H4, W4]
        G_mask = ((cols == 1) & (rows < 3)).float()
        B_mask = ((cols == 2) & (rows < 3)).float()

        # Initialize shifted_images with zeros
        shifted_images = torch.zeros_like(upscaled_image, device=device, dtype=dtype)

        # Assign shifted values for each channel
        shifted_images[:, 0, :, :] = upscaled_image[:, 0, :, :] * R_mask[:, 0, :, :]
        shifted_images[:, 1, :, :] = upscaled_image[:, 1, :, :] * G_mask[:, 1, :, :]
        shifted_images[:, 2, :, :] = upscaled_image[:, 2, :, :] * B_mask[:, 2, :, :]

        # Retrieve Gaussian kernel
        gaussian = self.get_gaussian_kernel(gaussian_kernel_size, gaussian_width_x, gaussian_width_y, gamma, device, dtype)  # [1,1,K,K]

        # Repeat the kernel for each channel
        gaussian = gaussian.repeat(C, 1, 1, 1)  # [3,1,K,K]

        # Apply convolution with the Gaussian kernel, per channel
        # Here, groups=3 to apply each kernel to its corresponding channel
        convolved = F.conv2d(shifted_images, gaussian, padding=0, groups=C)  # [B, 3, H*4, W*4]

        # Apply intensity scaling
        convolved = convolved * intensity_scale  # [B, 3, H*4, W*4]

        # Apply gamma correction
        convolved = torch.pow(convolved, 1 / gamma)  # [B, 3, H*4, W*4]

        # Clamp the values to [0, 1]
        convolved = torch.clamp(convolved, 0.0, 1.0)

        # Restore channel order and convert back to float32
        convolved = convolved.permute(0, 2, 3, 1).float().cpu()  # [B, H*4, W*4, C]

        return (convolved,)

    def apply_crt_effect(self, image, gaussian_width_x, gaussian_width_y, intensity_scale, gamma, gaussian_kernel_size, enable_resize, resize_pixels):
        """
        Applies the CRT-like effect to the input image with optional resizing.
        Utilizes optimized GPU processing with vectorization, precomputed kernels, and mixed precision.
        """
        print(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Ensure image is in float format
        if image.max() > 1.0:
            image = image / 255.0

        # Move image to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = image.to(device, dtype=torch.float32)

        B, H, W, C = image.shape

        # Validate the number of channels
        if C != 3:
            raise ValueError(f"Expected image with 3 channels (RGB), but got {C} channels.")

        # Permute to [B, C, H, W] regardless of enable_resize
        image = image.permute(0, 3, 1, 2)  # [B, C, H, W]

        if enable_resize:
            if H > W:
                new_h, new_w = resize_pixels, int(resize_pixels * W / H)
            else:
                new_h, new_w = int(resize_pixels * H / W), resize_pixels

            image = F.interpolate(image, size=(new_h, new_w), mode='nearest')  # [B, C, new_h, new_w]
            print(f"After initial resize: {image.shape}")
        else:
            print(f"No resizing applied. Image shape remains: {image.shape}")

        # Apply CRT effect with optimized GPU processing
        crt_image = self.apply_crt_effect_gpu_optimized(
            image,
            gaussian_width_x,
            gaussian_width_y,
            intensity_scale,
            gaussian_kernel_size,
            gamma
        )

        print(f"Final output shape: {crt_image[0].shape}")
        return crt_image

class XYBlurNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigma_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sigma_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_blur"
    CATEGORY = "image/Pixydust Quantizer🧚✨"

    def create_gaussian_kernel(self, kernel_size, sigma_x, sigma_y, device, dtype):
        """
        Generates an anisotropic Gaussian kernel with independent vertical and horizontal components.
        """
        # Generate grid coordinates
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
        y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        # Calculate Gaussian function (independent for vertical and horizontal)
        gaussian = torch.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
        
        # Normalize so that the sum is 1
        gaussian = gaussian / gaussian.sum()
        
        # Adjust the shape of the kernel to [1, 1, kernel_size, kernel_size]
        return gaussian.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

    def calculate_kernel_size(self, sigma_x, sigma_y, max_kernel_size=31):
        """
        Calculates an appropriate kernel size based on sigma_x and sigma_y.
        Typically, the kernel size is set to 6 * sigma and rounded to an odd number.
        """
        # Calculate kernel size based on the largest sigma
        max_sigma = max(sigma_x, sigma_y)
        kernel_size = int(6 * max_sigma + 1)
        
        # Adjust kernel size to be odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Set minimum kernel size to 3
        kernel_size = max(kernel_size, 3)
        
        # Set maximum kernel size
        kernel_size = min(kernel_size, max_kernel_size)
        
        return kernel_size

    def apply_blur(self, image: torch.Tensor, sigma_x: float, sigma_y: float, intensity: float):
        """
        Applies an anisotropic Gaussian blur to the image.
        The kernel size is automatically calculated based on sigma_x and sigma_y.
        """
        device = image.device
        dtype = image.dtype
        B, H, W, C = image.shape

        # Calculate kernel size
        kernel_size = self.calculate_kernel_size(sigma_x, sigma_y)
        print(f"Calculated kernel_size: {kernel_size} based on sigma_x: {sigma_x}, sigma_y: {sigma_y}")

        # Convert to [B, C, H, W] format
        x = image.permute(0, 3, 1, 2)

        # Generate Gaussian kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma_x, sigma_y, device, dtype)

        # Repeat the same kernel for each channel
        kernel = kernel.repeat(C, 1, 1, 1)

        # Calculate padding size
        pad_size = kernel_size // 2

        # Apply convolution operation (independently for each channel)
        blurred = F.conv2d(
            x,
            kernel,
            padding=pad_size,
            groups=C  # Process each channel independently
        )

        # Apply the intensity parameter
        # Allows intensity to be in the range [0, 2]
        # intensity = 1.0 for normal blur
        # intensity < 1.0 to decrease the blur intensity
        # intensity > 1.0 to increase the blur intensity
        blurred = x * (1 - intensity) + blurred * intensity

        # Convert back to [B, H, W, C] format
        blurred = blurred.permute(0, 2, 3, 1)

        # Clamp the values to [0, 1]
        blurred = torch.clamp(blurred, 0, 1)

        # Output debug information
        print(f"After blur: mean={blurred.mean().item():.4f}, std={blurred.std().item():.4f}, min={blurred.min().item():.4f}, max={blurred.max().item():.4f}")

        # Move the tensor to CPU before returning
        blurred = blurred.cpu()

        return (blurred,)

NODE_CLASS_MAPPINGS = {
    "CRTLikeEffectNode": CRTLikeEffectNode,
    "XYBlurNode": XYBlurNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTLikeEffectNode": "CRTLike Effect",
    "XYBlurNode": "XY Blur"
}
