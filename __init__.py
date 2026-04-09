"""ComfyUI-BatchSplit — package entry point.

Registers six nodes via the V3 ComfyExtension API:
  BatchSplitByStride, MaskSplitByStride  — split by explicit stride
  BatchSplitAuto, MaskSplitAuto          — split by inferred stride (num_batches)
  SaveSplitBatches, SaveSplitMasks       — write frames to disk
"""
from __future__ import annotations

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes import (
    BatchSplitByStride,
    BatchSplitAuto,
    MaskSplitByStride,
    MaskSplitAuto,
    SaveSplitBatches,
    SaveSplitMasks,
)


class BatchSplitExtension(ComfyExtension):
    """Extension that registers all six BatchSplit nodes with ComfyUI."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BatchSplitByStride,
            MaskSplitByStride,
            BatchSplitAuto,
            MaskSplitAuto,
            SaveSplitBatches,
            SaveSplitMasks,
        ]


async def comfy_entrypoint() -> BatchSplitExtension:
    """ComfyUI V3 entry point — returns the extension instance."""
    return BatchSplitExtension()
