"""Generation adapter for official ViLaSR evaluation using our validation limits.

Called by the local upstream eval/infer.py patch. This module does not launch
training or generation on import; vLLM and the processor are supplied by it.
"""

from dataclasses import dataclass
from copy import deepcopy
from oprd.agent.contracts import eval_seed
from oprd.agent.vilasr_contract import DrawingSession, FINAL, FOLLOWUP, _ANSWER, clean_segment, score_response


@dataclass(frozen=True)
class EvalLimits:
    prompt_length: int = 8192
    response_length: int = 16384
    max_model_len: int = 24576
    max_action_length: int = 4096
    max_observation_length: int = 8192
    max_tool_turns: int = 12
    max_images: int = 42
    seed: int = 42


def _encode(processor, text, images):
    return processor(text=[text], images=images, return_tensors="pt")["input_ids"][0].tolist()


def engine_prompt_ids(ids, processor):
    """Collapse expanded Qwen vision pads before vLLM expands them itself.

    Keep the fully processed IDs in trajectory state for budget accounting,
    matching the in-training vLLM server's Qwen2.5-VL token deduplication.
    """
    vision_tokens = {getattr(processor, key, None) for key in ("image_token_id", "video_token_id")}
    vision_tokens.discard(None)
    return [
        token
        for index, token in enumerate(ids)
        if not (index and token in vision_tokens and (token == ids[index - 1]))
    ]


def generate_aligned(engine, processor, tokenizer, inputs, sampling_params, limits=EvalLimits()):
    """Return question-major/sample-major records, including invalid trajectories.

    Budgets count processed image tokens and observations exactly as the
    in-training ViLaSRAgentLoop does. Each trajectory has its own drawing state.
    """
    states = []
    for item in inputs:
        prompt = item["prompt"]
        images = list(item["multi_modal_data"]["image"])
        if not images or len(images) > limits.max_images:
            raise ValueError("Initial image count exceeds validation limits")
        ids = _encode(processor, prompt, images)
        if len(ids) > limits.prompt_length:
            raise ValueError(f"Prompt exceeds validation limit: {item['qid']} ({len(ids)} tokens)")
        for sample_index in range(sampling_params.n):
            running_images = list(images)
            states.append(
                dict(
                    item=item,
                    sample_index=sample_index,
                    ids=list(ids),
                    sequence=prompt,
                    prompt_len=len(ids),
                    images=running_images,
                    session=DrawingSession(running_images),
                    segments=[],
                    interactions=[],
                    stop="",
                )
            )
    for turn in range(limits.max_tool_turns + 1):
        active, requests, params = ([], [], [])
        for state in states:
            if state["stop"]:
                continue
            used = len(state["ids"]) - state["prompt_len"]
            available = min(limits.response_length - used, limits.max_model_len - len(state["ids"]))
            if available <= 0:
                state["stop"] = "response_budget"
                continue
            sampling = deepcopy(sampling_params)
            sampling.n = 1
            sampling.max_tokens = min(limits.max_action_length, available)
            sampling.seed = eval_seed(
                state["item"]["data_source"],
                state["item"]["qid"],
                limits.seed,
                turn,
                sample_index=state["sample_index"],
            )
            active.append((state, available))
            params.append(sampling)
            requests.append(
                dict(
                    prompt_token_ids=engine_prompt_ids(state["ids"], processor),
                    multi_modal_data={"image": state["images"]},
                )
            )
        if not requests:
            break
        outputs = engine.generate(prompts=requests, sampling_params=params, use_tqdm=False)
        if len(outputs) != len(active):
            raise RuntimeError("vLLM returned an incomplete evaluation batch")
        for (state, available), sampling, output in zip(active, params, outputs):
            generated = list(output.outputs[0].token_ids)
            text = tokenizer.decode(generated, skip_special_tokens=False)
            state["ids"].extend(generated)
            state["segments"].append(text)
            state["sequence"] += text
            if len(generated) >= sampling.max_tokens:
                state["stop"] = (
                    "action_length_limit" if limits.max_action_length < available else "response_budget"
                )
                continue
            if _ANSWER.fullmatch(clean_segment(text)):
                state["stop"] = "answer"
                continue
            if turn == limits.max_tool_turns:
                state["stop"] = "max_turns_reached"
                continue
            before = len(state["images"])
            try:
                new_images = state["session"].execute(text)
            except (ValueError, KeyError, TypeError, SyntaxError, OverflowError) as exc:
                state["interactions"].append(
                    dict(action=text, obs=str(exc), valid_action=False, observation_accepted=False)
                )
                state["stop"] = "image_limit" if str(exc) == "image_limit" else "tool_error"
                continue
            parts = [
                f"<|vision_start|><|image_pad|><|vision_end|>The index of the given image is {index} (width: {image.width}, height: {image.height}).\n"
                for index, image in enumerate(new_images, before + 1)
            ]
            parts.append(
                FINAL
                if turn + 1 == limits.max_tool_turns or len(state["images"]) >= limits.max_images
                else FOLLOWUP
            )
            end = "" if state["ids"][-1] == tokenizer.eos_token_id else "<|im_end|>"
            observation = (
                end + "\n<|im_start|>user\n" + "".join(parts) + "\n<|im_end|>\n<|im_start|>assistant\n"
            )
            obs_ids = _encode(processor, observation, new_images)
            used = len(state["ids"]) - state["prompt_len"]
            accepted = len(obs_ids) <= min(
                limits.max_observation_length,
                limits.response_length - used,
                limits.max_model_len - len(state["ids"]),
            )
            state["interactions"].append(
                dict(action=text, obs=observation, valid_action=True, observation_accepted=accepted)
            )
            if not accepted:
                del state["images"][before:]
                state["stop"] = "observation_response_budget"
                continue
            state["ids"].extend(obs_ids)
            state["sequence"] += observation
    results = []
    for state in states:
        gold = state["item"]["gold"]
        rule = score_response(state["segments"], state["stop"], gold, state["interactions"])
        results.append(
            dict(
                sequence=state["sequence"],
                sample_index=state["sample_index"],
                qid=state["item"]["qid"],
                model_segments=state["segments"],
                tool_interact_info=state["interactions"],
                traj_stop_reason=state["stop"],
                protocol_valid=not bool(rule["incorrect_format"]),
                validation_rule_score=rule["score"],
                response_tokens=len(state["ids"]) - state["prompt_len"],
            )
        )
    return results
