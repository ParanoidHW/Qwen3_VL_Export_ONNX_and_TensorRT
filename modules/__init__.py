_EXPORTS = {
    "Qwen3VLTextModelOpt": ("modules.qwen3_vl._text", "Qwen3VLTextModelOpt"),
    "Qwen3VLVisualModelOpt": ("modules.qwen3_vl._visual", "Qwen3VLVisualModelOpt"),
    "Qwen3VLModelOpt": ("modules.qwen3_vl._visual_language", "Qwen3VLModelOpt"),
    "Qwen3VLForConditionalGenerationOpt": (
        "modules.qwen3_vl._generation",
        "Qwen3VLForConditionalGenerationOpt",
    ),
    "Qwen35VLTextModelOpt": ("modules.qwen35_vl._text", "Qwen35VLTextModelOpt"),
    "Qwen35VLTextModelWithCacheOpt": ("modules.qwen35_vl._text", "Qwen35VLTextModelWithCacheOpt"),
    "Qwen35VLVisualModelOpt": ("modules.qwen35_vl._visual", "Qwen35VLVisualModelOpt"),
    "Qwen35VLModelOpt": ("modules.qwen35_vl._visual_language", "Qwen35VLModelOpt"),
    "Qwen35VLForConditionalGenerationOpt": (
        "modules.qwen35_vl._generation",
        "Qwen35VLForConditionalGenerationOpt",
    ),
    "Qwen35VLTokenEmbeddingOpt": ("modules.qwen35_vl._generation", "Qwen35VLTokenEmbeddingOpt"),
    "Qwen35VLEmbedSelectOpt": ("modules.qwen35_vl._generation", "Qwen35VLEmbedSelectOpt"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
