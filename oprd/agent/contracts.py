import hashlib
from collections import Counter
import uuid


def eval_seed(dataset, question_id, seed=42, turn=0, *, sample_index=0):
    """Stable across checkpoints; sample zero preserves the historical seed."""
    if not isinstance(sample_index, int) or isinstance(sample_index, bool) or sample_index < 0:
        raise ValueError("Evaluation sample_index must be a nonnegative integer")
    text = f"{seed}\x00{dataset}\x00{question_id}\x00{turn}"
    if sample_index:
        text += f"\x00sample:{sample_index}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def occurrence_uids(count):
    """One fresh group per sampled prompt occurrence, even when qid/content repeats."""
    if count != 64:
        raise ValueError(f"Expected 64 prompt occurrences before rollout, got {count}")
    prefix = uuid.uuid4().hex
    return [f"{prefix}:{index}" for index in range(count)]


def validate_training_groups(uids):
    counts = Counter(uids)
    if len(uids) != 512 or len(counts) != 64 or set(counts.values()) != {8}:
        raise ValueError(
            f"Training batch must contain 64 occurrence groups x 8 rollouts; got {len(uids)} trajectories / {len(counts)} groups / counts={sorted(set(counts.values()))}"
        )
