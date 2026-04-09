"""ComfyUI-BatchSplit — split interleaved batches into per-track sub-batches."""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image as PILImage

from comfy_api.latest import io


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _split_tensor(tensor: torch.Tensor, stride: int, tag: str) -> list[torch.Tensor]:
    """Split *tensor* along dim 0 into equal-sized chunks of *stride* frames.

    Args:
        tensor: Input tensor whose first dimension is the frame count.
        stride: Number of frames per output sub-batch.
        tag:    Node name used in warning messages.

    Returns:
        List of cloned sub-tensors, each of shape ``(stride, ...)``.
    """
    total = tensor.shape[0]
    n = total // stride
    if total % stride != 0:
        print(
            f"[{tag}] Warning: {total} frames is not divisible by stride {stride}. "
            f"Truncating {total % stride} trailing frame(s)."
        )
    return [tensor[i * stride:(i + 1) * stride].clone() for i in range(n)]


# ---------------------------------------------------------------------------
# Image split nodes
# ---------------------------------------------------------------------------

class BatchSplitByStride(io.ComfyNode):
    """Split an interleaved IMAGE batch into sub-batches using an explicit stride."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BatchSplitByStride",
            display_name="Batch Split by Stride (Image)",
            category="custom/batch",
            description=(
                "Split a flat interleaved IMAGE batch (N*stride × H × W × C) "
                "into a list of sub-tensors, each of shape (stride, H, W, C)."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Int.Input(
                    "stride",
                    default=10,
                    min=1,
                    max=9999,
                    tooltip="Number of frames per sub-batch (= source video length in frames).",
                ),
            ],
            outputs=[
                io.Image.Output("sub_batches", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, stride: int) -> io.NodeOutput:
        return io.NodeOutput(_split_tensor(images, stride, "BatchSplitByStride"))


class BatchSplitAuto(io.ComfyNode):
    """Split an interleaved IMAGE batch into N equal sub-batches (stride inferred)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BatchSplitAuto",
            display_name="Batch Split Auto (Image)",
            category="custom/batch",
            description=(
                "Infer stride as total_frames // num_batches, then split the IMAGE tensor "
                "into that many equal sub-batches."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Int.Input(
                    "num_batches",
                    default=1,
                    min=1,
                    max=999,
                    tooltip="Number of sub-batches to produce (total_frames // video_length).",
                ),
            ],
            outputs=[
                io.Image.Output("sub_batches", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, num_batches: int) -> io.NodeOutput:
        stride = images.shape[0] // num_batches
        return io.NodeOutput(_split_tensor(images, stride, "BatchSplitAuto"))


# ---------------------------------------------------------------------------
# Mask split nodes
# ---------------------------------------------------------------------------

class MaskSplitByStride(io.ComfyNode):
    """Split an interleaved MASK batch into sub-batches using an explicit stride."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MaskSplitByStride",
            display_name="Batch Split by Stride (Mask)",
            category="custom/batch",
            description=(
                "Split a flat interleaved MASK batch (N*stride × H × W) "
                "into a list of sub-tensors, each of shape (stride, H, W)."
            ),
            inputs=[
                io.Mask.Input("images"),
                io.Int.Input(
                    "stride",
                    default=10,
                    min=1,
                    max=9999,
                    tooltip="Number of frames per sub-batch (= source video length in frames).",
                ),
            ],
            outputs=[
                io.Mask.Output("sub_batches", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, stride: int) -> io.NodeOutput:
        return io.NodeOutput(_split_tensor(images, stride, "MaskSplitByStride"))


class MaskSplitAuto(io.ComfyNode):
    """Split an interleaved MASK batch into N equal sub-batches (stride inferred)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MaskSplitAuto",
            display_name="Batch Split Auto (Mask)",
            category="custom/batch",
            description=(
                "Infer stride as total_frames // num_batches, then split the MASK tensor "
                "into that many equal sub-batches."
            ),
            inputs=[
                io.Mask.Input("images"),
                io.Int.Input(
                    "num_batches",
                    default=1,
                    min=1,
                    max=999,
                    tooltip="Number of sub-batches to produce (total_frames // video_length).",
                ),
            ],
            outputs=[
                io.Mask.Output("sub_batches", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, num_batches: int) -> io.NodeOutput:
        stride = images.shape[0] // num_batches
        return io.NodeOutput(_split_tensor(images, stride, "MaskSplitAuto"))


# ---------------------------------------------------------------------------
# Save nodes
# ---------------------------------------------------------------------------

class SaveSplitBatches(io.ComfyNode):
    """Save each IMAGE sub-batch to disk as individually indexed frame files."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveSplitBatches",
            display_name="Save Split Batches (Image)",
            category="custom/batch",
            description=(
                "Receive a list of IMAGE sub-batch tensors and write every frame to disk "
                "as {prefix}_{batch_idx:02d}_frame_{frame_idx:04d}.{ext}."
            ),
            is_output_node=True,
            is_input_list=True,
            inputs=[
                io.Image.Input("batches"),
                io.String.Input("output_dir", default="output/batches"),
                io.String.Input("prefix", default="batch"),
                io.Combo.Input("image_format", options=["PNG", "JPEG", "WEBP"], default="PNG"),
                io.Int.Input("jpeg_quality", default=95, min=1, max=100),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        batches: list[torch.Tensor],
        output_dir: list[str],
        prefix: list[str],
        image_format: list[str],
        jpeg_quality: list[int],
    ) -> io.NodeOutput:
        # With is_input_list=True all inputs arrive as lists; unpack scalar params.
        out_dir = output_dir[0]
        pfx = prefix[0]
        fmt = image_format[0].upper()
        quality = jpeg_quality[0]
        ext = fmt.lower()

        os.makedirs(out_dir, exist_ok=True)

        for i, batch in enumerate(batches):
            for f in range(batch.shape[0]):
                frame = batch[f].cpu().detach()
                arr = (frame.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                img = PILImage.fromarray(arr)
                filename = f"{pfx}_{i:02d}_frame_{f:04d}.{ext}"
                filepath = os.path.join(out_dir, filename)
                if fmt == "JPEG":
                    img.save(filepath, quality=quality)
                else:
                    img.save(filepath)

        return io.NodeOutput()


class SaveSplitMasks(io.ComfyNode):
    """Save each MASK sub-batch to disk as grayscale PNG frame files."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveSplitMasks",
            display_name="Save Split Batches (Mask)",
            category="custom/batch",
            description=(
                "Receive a list of MASK sub-batch tensors and write every frame to disk "
                "as grayscale PNG: {prefix}_{batch_idx:02d}_frame_{frame_idx:04d}.png. "
                "Masks are always saved as PNG regardless of the image_format setting."
            ),
            is_output_node=True,
            is_input_list=True,
            inputs=[
                io.Mask.Input("batches"),
                io.String.Input("output_dir", default="output/batches"),
                io.String.Input("prefix", default="batch"),
                io.Combo.Input(
                    "image_format",
                    options=["PNG", "JPEG", "WEBP"],
                    default="PNG",
                    tooltip="Masks are always saved as PNG regardless of this setting.",
                ),
                io.Int.Input("jpeg_quality", default=95, min=1, max=100),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        batches: list[torch.Tensor],
        output_dir: list[str],
        prefix: list[str],
        image_format: list[str],
        jpeg_quality: list[int],
    ) -> io.NodeOutput:
        # With is_input_list=True all inputs arrive as lists; unpack scalar params.
        out_dir = output_dir[0]
        pfx = prefix[0]

        os.makedirs(out_dir, exist_ok=True)

        for i, batch in enumerate(batches):
            for f in range(batch.shape[0]):
                frame = batch[f].cpu().detach()
                arr = (frame.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                img = PILImage.fromarray(arr, mode="L")
                filename = f"{pfx}_{i:02d}_frame_{f:04d}.png"
                filepath = os.path.join(out_dir, filename)
                img.save(filepath)

        return io.NodeOutput()


print("[comfyui_batch_split] Loaded — 6 nodes registered.")
