"""Teacher forcing on the student's already processed ViLaSR visual context."""

from verl.utils.model import extract_multi_modal_inputs
from .oprd import response_rows


def shared_teacher_inputs(batch, max_context):
    ids, mask = batch["input_ids"], batch["attention_mask"].bool()
    lengths = mask.sum(-1)
    if (lengths == 0).any() or (lengths > max_context).any():
        raise ValueError("Empty or overlong shared teacher context")
    indices = mask.flatten().nonzero().squeeze(-1)
    positions = batch["position_ids"]
    if positions.ndim != 3 or positions.shape[1] != 4:
        raise ValueError("ViLaSR OPD requires the student's four-channel packed mRoPE positions")
    packed_positions = positions.transpose(0, 1).reshape(4, -1).index_select(1, indices).unsqueeze(1)
    multimodal = extract_multi_modal_inputs(batch["multi_modal_inputs"])
    # qwen2_vl_forward consumes the pixels/grids, not the actor's packing metadata.
    allowed = {"pixel_values", "image_grid_thw"}
    if not allowed <= multimodal.keys() or "video_grid_thw" in multimodal:
        raise ValueError("ViLaSR OPD requires image pixels and grids, without videos")
    inputs = {key: multimodal[key] for key in allowed}
    inputs.update(
        input_ids=ids.flatten().index_select(0, indices).unsqueeze(0),
        attention_mask=None,
        position_ids=packed_positions,
        use_cache=False,
    )
    rows, labels = response_rows(batch)
    return inputs, rows, labels, lengths
