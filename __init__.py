# import sys
# import subprocess

# def install_package(package_name):
#     print(f"## Pixydust Quantizer: installing {package_name}")
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
#     print(f"## Pixydust Quantizer: {package_name} installation completed")

# # Required packages
# required_packages = ['opencv-python', 'numpy', 'torch', 'Pillow', 'scikit-learn', 'scipy', 'scikit-image']

# # Check and install required packages
# for package in required_packages:
#     try:
#         __import__(package.replace('-', '_'))
#     except ImportError:
#         install_package(package)

print("### Loading: Pixydust Quantizer")

# from .pixydust_quantizer import ColorReducerNode, PaletteOptimizationNode

# NODE_CLASS_MAPPINGS = {
#     "ColorReducerNode": ColorReducerNode,
#     "PaletteOptimizationNode": PaletteOptimizationNode,
# }

# # Add these to your NODE_DISPLAY_NAME_MAPPINGS
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "ColorReducerNode": "Pixydust Quantize-1",
#     "PaletteOptimizationNode": "Pixydust Quantize-2",
# }
from .pixydust_quantizer import NODE_CLASS_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
