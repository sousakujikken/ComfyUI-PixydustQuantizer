VERSION = "2.0.0"
print(f"### Loading: Pixydust Quantizer v{VERSION}")

# Import node mappings from each module
from .crtlike_effect_node import NODE_CLASS_MAPPINGS as CRT_NODES
from .crtlike_effect_node import NODE_DISPLAY_NAME_MAPPINGS as CRT_DISPLAY_NAMES
from .pixydust_quantizer import NODE_CLASS_MAPPINGS as QUANTIZER_NODE
from .pixydust_quantizer import NODE_DISPLAY_NAME_MAPPINGS as QUANTIZER_DISPLAY_NAMES

# Merge all node mappings
NODE_CLASS_MAPPINGS = {
    **CRT_NODES,
    **QUANTIZER_NODE,
}

# Merge all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    **CRT_DISPLAY_NAMES,
    **QUANTIZER_DISPLAY_NAMES,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']