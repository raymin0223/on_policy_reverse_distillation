"""Validate token identity before applying teacher-derived token losses."""

from omegaconf import open_dict


def _as_str_or_none(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def _tokenizer_id_to_token(tokenizer) -> dict[int, str]:
    id_to_token: dict[int, str] = {}
    for token, token_id in tokenizer.get_vocab().items():
        try:
            token_id = int(token_id)
        except (TypeError, ValueError):
            continue
        id_to_token[token_id] = str(token)
    return id_to_token


def _configure_teacher_tokenizer_compatibility(config):
    """Guard OPD when teacher/student tokenizers only differ in added tokens."""
    distillation_cfg = config.actor_rollout_ref.actor.distillation
    policy = (
        str(distillation_cfg.get("teacher_tokenizer_mismatch_policy", "auto_mask_special")).strip().lower()
    )
    if policy in {"", "none", "null", "off", "disabled"}:
        return
    if policy not in {
        "auto_mask_special",
        "strict",
        "allow",
        "retokenize_mask",
        "retokenize",
        "decode_retokenize",
    }:
        raise ValueError(
            f"distillation.teacher_tokenizer_mismatch_policy must be one of ['auto_mask_special', 'strict', 'allow', 'retokenize_mask', 'off'], got {policy!r}."
        )
    student_model_path = _as_str_or_none(config.actor_rollout_ref.model.path)
    teacher_model_path = _as_str_or_none(distillation_cfg.get("teacher_model_path"))
    if student_model_path is None or teacher_model_path is None:
        return
    try:
        from transformers import AutoTokenizer

        trust_remote_code = bool(config.actor_rollout_ref.model.get("trust_remote_code", False))
        student_tokenizer = AutoTokenizer.from_pretrained(
            student_model_path, trust_remote_code=trust_remote_code
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_path, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        if policy == "allow":
            print(f"[distillation] Tokenizer compatibility check skipped: {exc}")
            return
        raise RuntimeError(
            "Failed to compare student/teacher tokenizers for distillation. Set actor_rollout_ref.actor.distillation.teacher_tokenizer_mismatch_policy=allow only if you have verified that tokenizer ids are compatible."
        ) from exc
    student_vocab_size = int(getattr(student_tokenizer, "vocab_size", 0) or 0)
    teacher_vocab_size = int(getattr(teacher_tokenizer, "vocab_size", 0) or 0)
    student_by_id = _tokenizer_id_to_token(student_tokenizer)
    teacher_by_id = _tokenizer_id_to_token(teacher_tokenizer)
    base_limit = min(student_vocab_size, teacher_vocab_size)
    base_mismatched_ids: list[int] = []
    for token_id in range(base_limit):
        if student_by_id.get(token_id) != teacher_by_id.get(token_id):
            base_mismatched_ids.append(token_id)
            if len(base_mismatched_ids) >= 16:
                break
    if student_vocab_size != teacher_vocab_size:
        base_mismatched_ids.append(base_limit)
    all_ids = set(student_by_id) | set(teacher_by_id)
    added_mismatched_ids = sorted(
        (
            token_id
            for token_id in all_ids
            if token_id >= base_limit and student_by_id.get(token_id) != teacher_by_id.get(token_id)
        )
    )
    retokenize_policy = policy in {"retokenize_mask", "retokenize", "decode_retokenize"}
    if base_mismatched_ids and policy != "allow" and (not retokenize_policy):
        raise ValueError(
            f"Student/teacher base tokenizer vocabularies differ, so logit distillation is unsafe. First mismatched ids: {base_mismatched_ids[:16]}. Use the same tokenizer family, set teacher_tokenizer_mismatch_policy=retokenize_mask to rebuild teacher inputs from text and mask nonmatching response-token positions, or set teacher_tokenizer_mismatch_policy=allow only for a manual override."
        )
    if added_mismatched_ids and policy == "strict":
        raise ValueError(
            f"Student/teacher tokenizers have mismatched added/special tokens and strict policy is enabled. Mismatched ids: {added_mismatched_ids[:32]}."
        )
    existing_ids = distillation_cfg.get("teacher_tokenizer_incompatible_token_ids", [])
    if isinstance(existing_ids, str):
        text = existing_ids.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        existing_ids = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    try:
        existing_ids = [int(token_id) for token_id in existing_ids]
    except TypeError:
        existing_ids = []
    incompatible_ids = sorted(set(existing_ids) | set(added_mismatched_ids))
    with open_dict(config):
        distillation_cfg.teacher_tokenizer_incompatible_token_ids = incompatible_ids
        distillation_cfg.teacher_tokenizer_base_vocab_match = 0 if base_mismatched_ids else 1
    print(
        "[distillation] Tokenizer compatibility:",
        {
            "student_model_path": student_model_path,
            "teacher_model_path": teacher_model_path,
            "student_vocab_size": student_vocab_size,
            "teacher_vocab_size": teacher_vocab_size,
            "base_vocab_match": not base_mismatched_ids,
            "masked_incompatible_token_ids": incompatible_ids[:64],
            "num_masked_incompatible_token_ids": len(incompatible_ids),
        },
    )


def _validate_w2s_tokenizer_identity(config) -> None:
    """Require exact token-id identity across all four W2S-OPD policies."""
    distillation_cfg = config.actor_rollout_ref.actor.distillation
    if not bool(distillation_cfg.get("w2s_opd_enabled", False)):
        return
    paths = {
        "current_student": _as_str_or_none(config.actor_rollout_ref.model.path),
        "student_base": _as_str_or_none(distillation_cfg.get("w2s_student_base_model_path")),
        "positive": _as_str_or_none(distillation_cfg.get("w2s_positive_model_path"))
        or _as_str_or_none(distillation_cfg.get("teacher_model_path")),
        "negative": _as_str_or_none(distillation_cfg.get("w2s_negative_model_path")),
    }
    missing = [role for role, path in paths.items() if path is None]
    if missing:
        raise ValueError(f"W2S-OPD requires model paths for {missing}.")
    from transformers import AutoTokenizer

    trust_remote_code = bool(config.actor_rollout_ref.model.get("trust_remote_code", False))
    tokenizers = {
        role: AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
        for role, path in paths.items()
    }
    student_vocab = _tokenizer_id_to_token(tokenizers["current_student"])
    student_vocab_size = int(getattr(tokenizers["current_student"], "vocab_size", 0) or 0)
    for role in ("student_base", "positive", "negative"):
        tokenizer = tokenizers[role]
        role_vocab = _tokenizer_id_to_token(tokenizer)
        role_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        all_ids = sorted(set(student_vocab) | set(role_vocab))
        mismatched_ids = [
            token_id for token_id in all_ids if student_vocab.get(token_id) != role_vocab.get(token_id)
        ]
        if student_vocab_size != role_vocab_size or mismatched_ids:
            raise ValueError(
                f"W2S-OPD requires exact vocabulary/tokenizer identity across current student, student base, positive, and negative models. role={role!r}, student_vocab_size={student_vocab_size}, role_vocab_size={role_vocab_size}, first_mismatched_ids={mismatched_ids[:32]}."
            )
    print("[w2s-opd] Exact four-policy tokenizer identity verified:", paths)
