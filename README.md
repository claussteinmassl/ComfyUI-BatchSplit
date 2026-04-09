# ComfyUI-BatchSplit

Split a flat, interleaved batch tensor back into separate per-track sub-batches, then optionally save each track's frames to disk. A typical use case is post-processing the output of a multi-object tracker (e.g. YOLO) that stacks all tracks into a single batch before passing it through ComfyUI: this package undoes that stacking so downstream nodes can work on one track at a time.

## Installation

1. Copy (or symlink) this folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.

No extra dependencies are required — only `torch` and `Pillow`, both of which ship with ComfyUI.

## Node reference

| Node | Inputs | Output | Use case |
|---|---|---|---|
| **Batch Split by Stride (Image)** | `images` IMAGE, `stride` INT | IMAGE list | Split when you know the exact frame count per track |
| **Batch Split by Stride (Mask)** | `images` MASK, `stride` INT | MASK list | Same, for mask tensors |
| **Batch Split Auto (Image)** | `images` IMAGE, `num_batches` INT | IMAGE list | Split when you know the number of tracks |
| **Batch Split Auto (Mask)** | `images` MASK, `num_batches` INT | MASK list | Same, for mask tensors |
| **Save Split Batches (Image)** | `batches` IMAGE list, `output_dir`, `prefix`, `image_format`, `jpeg_quality` | — | Save every frame of every sub-batch to disk |
| **Save Split Batches (Mask)** | `batches` MASK list, `output_dir`, `prefix`, `image_format`, `jpeg_quality` | — | Same, always saved as grayscale PNG |

### Input details

- **stride** — number of frames per track (equals the original video length).
- **num_batches** — number of tracks; stride is inferred as `total_frames // num_batches`.
- **output_dir** — directory to write files into (created automatically).
- **prefix** — filename prefix; files are named `{prefix}_{i:02d}_frame_{f:04d}.{ext}`.
- **image_format** — `PNG`, `JPEG`, or `WEBP` for image save nodes. Mask nodes always use PNG.
- **jpeg_quality** — JPEG quality (1–100), ignored for PNG and WEBP.

## Typical workflow

```
[Load Video Frames]
        │  IMAGE  (N*stride × H × W × C)
        ▼
[Batch Split by Stride (Image)]   stride = video_length
        │  IMAGE list  (one tensor per track)
        ▼
[... per-track processing nodes ...]
        │  IMAGE list
        ▼
[Save Split Batches (Image)]   output_dir = "output/tracks"
```

If you also have masks (e.g. from a segmentation model), run a parallel branch with the Mask variants and connect both lists to their respective save nodes.

## Notes on list behaviour

### `is_output_list` (split nodes)

The four split nodes declare their output with `is_output_list=True`. ComfyUI treats this output as a *list of tensors* rather than a single batched tensor. When you connect it to another node that accepts a single IMAGE/MASK, ComfyUI will iterate over the list automatically (map over each sub-batch). When you connect it to a node that has `is_input_list=True` (like the save nodes here), the entire list is passed in at once.

### `is_input_list` (save nodes)

The two save nodes declare `is_input_list=True` on their schema. This means they receive the **full list** in one call rather than being invoked once per element. All other inputs (`output_dir`, `prefix`, etc.) also arrive as single-element lists and are unpacked internally.
