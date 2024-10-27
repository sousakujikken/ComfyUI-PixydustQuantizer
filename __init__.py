VERSION = "1.1.0"
print(f"### Loading: Pixydust Quantizer v{VERSION}")

# Import node mappings from each module
from .pixydust_quantizer import NODE_CLASS_MAPPINGS as PIXYDUST_NODES
from .pixydust_quantizer import NODE_DISPLAY_NAME_MAPPINGS as PIXYDUST_DISPLAY_NAMES
from .crtlike_effect_node import NODE_CLASS_MAPPINGS as CRT_NODES
from .crtlike_effect_node import NODE_DISPLAY_NAME_MAPPINGS as CRT_DISPLAY_NAMES

# Merge all node mappings
NODE_CLASS_MAPPINGS = {
    **PIXYDUST_NODES,
    **CRT_NODES
}

# Merge all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    **PIXYDUST_DISPLAY_NAMES,
    **CRT_DISPLAY_NAMES
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']