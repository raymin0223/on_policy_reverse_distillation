"""ViLaSR drawing in root Verl. Observations never receive policy-gradient loss."""

from uuid import uuid4
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from .contracts import eval_seed
from .vilasr_contract import DrawingSession, FOLLOWUP, FINAL, _ANSWER, clean_segment


@register("vilasr_agent")
class ViLaSRAgentLoop(AgentLoopBase):
    async def apply_chat_template(self, messages, **kwargs):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], images=kwargs.get("images"), return_tensors="pt")["input_ids"][
            0
        ].tolist()

    async def run(self, sampling_params, **kwargs):
        cfg = self.config.actor_rollout_ref.rollout
        limits = self.config.actor_rollout_ref.agent_protocol
        validate = bool(kwargs.get("agent_validate", False))
        info = kwargs["extra_info"]
        messages = list(kwargs["raw_prompt"])
        images = (await self.process_vision_info(messages)).get("images") or []
        if not images or len(images) != len(info["images"]):
            raise ValueError("ViLaSR image order/count mismatch")
        prompt = await self.apply_chat_template(messages, images=images)
        if len(prompt) > cfg.prompt_length:
            raise ValueError("Prompt exceeds audited limit")
        ids, running_images = list(prompt), list(images)
        draw_session = DrawingSession(running_images)
        mask, logprobs, segments, interactions = [], [], [], []
        metrics, stop, request = {}, "", uuid4().hex
        max_turns = limits.val_max_tool_turns if validate else limits.train_max_tool_turns
        per_turn = limits.val_max_action_length if validate else limits.train_max_action_length
        for turn in range(max_turns + 1):
            available = min(cfg.response_length - len(mask), cfg.max_model_len - len(ids))
            if available <= 0:
                stop = "response_budget"
                break
            params = dict(sampling_params)
            params.update(n=1, max_tokens=min(per_turn, available), stop=[])
            if validate:
                params["seed"] = eval_seed(
                    kwargs["data_source"],
                    info["qid"],
                    limits.eval_seed,
                    turn,
                    sample_index=int(info.get("eval_sample_index", 0)),
                )
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=f"{request}-{turn}",
                    prompt_ids=ids,
                    sampling_params=params,
                    image_data=running_images,
                )
            generated = list(output.token_ids)
            text = self.tokenizer.decode(generated, skip_special_tokens=False)
            ids.extend(generated)
            mask.extend([1] * len(generated))
            logprobs.extend(output.log_probs or [0.0] * len(generated))
            segments.append(text)
            if len(generated) >= params["max_tokens"]:
                stop = "action_length_limit" if per_turn < available else "response_budget"
                break
            if _ANSWER.fullmatch(clean_segment(text)):
                stop = "answer"
                break
            if turn == max_turns:
                stop = "max_turns_reached"
                break
            before = len(running_images)
            with simple_timer("tool_calls", metrics):
                try:
                    new_images = await self.loop.run_in_executor(None, draw_session.execute, text)
                except (ValueError, KeyError, TypeError, SyntaxError, OverflowError) as exc:
                    interactions.append(
                        {
                            "action": text,
                            "obs": str(exc),
                            "valid_action": False,
                            "observation_accepted": False,
                        }
                    )
                    stop = "image_limit" if str(exc) == "image_limit" else "tool_error"
                    break
            parts = []
            for index, image in enumerate(new_images, before + 1):
                parts.append(
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    f"The index of the given image is {index} (width: {image.width}, height: {image.height}).\n"
                )
            parts.append(FINAL if turn + 1 == max_turns or len(running_images) >= 42 else FOLLOWUP)
            observation = "".join(parts)
            end = "" if ids and ids[-1] == self.tokenizer.eos_token_id else "<|im_end|>"
            observation = end + "\n<|im_start|>user\n" + observation + "\n<|im_end|>\n<|im_start|>assistant\n"
            obs_ids = self.processor(text=[observation], images=new_images, return_tensors="pt")["input_ids"][
                0
            ].tolist()
            accepted = len(obs_ids) <= min(
                limits.max_observation_length, cfg.response_length - len(mask), cfg.max_model_len - len(ids)
            )
            interactions.append(
                {"action": text, "obs": observation, "valid_action": True, "observation_accepted": accepted}
            )
            if not accepted:
                # execute() appended images but the observation is not in the context.
                del running_images[before:]
                stop = "observation_response_budget"
                break
            ids.extend(obs_ids)
            mask.extend([0] * len(obs_ids))
            logprobs.extend([0.0] * len(obs_ids))
        extra = {"model_segments": segments, "tool_interact_info": interactions, "traj_stop_reason": stop}
        return AgentLoopOutput(
            prompt_ids=prompt,
            response_ids=ids[len(prompt) :],
            response_mask=mask,
            response_logprobs=logprobs,
            multi_modal_data={"images": running_images},
            num_turns=len(segments),
            metrics=AgentLoopMetrics(**metrics),
            extra_fields=extra,
        )
