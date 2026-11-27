"""ComfyUI extension entry point."""


async def comfy_entrypoint():
    from .nodes import MossGuidanceExtension

    return MossGuidanceExtension()
