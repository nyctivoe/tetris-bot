import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tetrisformer.model import (
    CACHE_VERSION,
    DEFAULT_SPIN_MODE,
    _apply_placement_fields_to_current_piece,
    _set_engine_root_state,
)
from tetrisEngine import TetrisEngine


def _spin_label_from_stats(clear_stats: dict) -> str:
    spin = clear_stats.get("spin")
    if isinstance(spin, dict) and spin.get("spin_type") == "t-spin":
        return "M" if spin.get("is_mini") else "F"
    return "N"


def _sample_step_refs(cache_dir: str, sample_size: int, seed: int):
    rng = random.Random(seed)
    reservoir = []
    seen = 0

    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(cache_dir, name)
        data = torch.load(path, weights_only=False)
        cache_version = int(data.get("cache_version", 0) or 0)
        if cache_version != CACHE_VERSION:
            raise RuntimeError(
                f"Unsupported cache version in {path}: expected v{CACHE_VERSION}, found v{cache_version or 'missing'}."
            )

        game_id = int(data.get("game_id", 0))
        for step_idx, _step in enumerate(data.get("steps", [])):
            ref = (path, game_id, step_idx)
            seen += 1
            if sample_size <= 0 or len(reservoir) < sample_size:
                reservoir.append(ref)
                continue
            replace_idx = rng.randrange(seen)
            if replace_idx < sample_size:
                reservoir[replace_idx] = ref

    return reservoir, seen


def run_alignment_report(cache_dir: str, sample_size: int = 5000, seed: int = 0) -> dict:
    refs, total_available_steps = _sample_step_refs(cache_dir, sample_size=sample_size, seed=seed)
    refs_by_path = defaultdict(list)
    for path, game_id, step_idx in refs:
        refs_by_path[path].append((game_id, step_idx))

    clear_mismatch_count = 0
    attack_exact_matches = 0
    non_t_spin_positives = 0
    samples_evaluated = 0
    spin_confusion_counts = Counter()
    attack_deltas = []

    for path, path_refs in refs_by_path.items():
        data = torch.load(path, weights_only=False)
        steps = data.get("steps", [])

        for game_id, step_idx in path_refs:
            step = steps[step_idx]
            engine = TetrisEngine(spin_mode=DEFAULT_SPIN_MODE)
            _set_engine_root_state(engine, step)

            placement = step["bfs_placements"][step["expert_match_index"]]
            if not _apply_placement_fields_to_current_piece(engine, placement):
                raise RuntimeError(
                    f"Cached expert placement is invalid for game_id={game_id}, step_idx={step_idx}."
                )

            engine.lock_piece(run_end_phase=False)
            clear_stats = engine.last_clear_stats or {}
            expert = step["expert_replay"]

            actual_cleared = int(clear_stats.get("lines_cleared", 0) or 0)
            actual_attack = int(clear_stats.get("attack", 0) or 0)
            actual_spin = _spin_label_from_stats(clear_stats)

            expected_cleared = int(expert.get("cleared", 0) or 0)
            expected_attack = int(expert.get("attack", 0) or 0)
            expected_spin = str(expert.get("t_spin", "N") or "N")

            samples_evaluated += 1
            if actual_cleared != expected_cleared:
                clear_mismatch_count += 1
            if actual_attack == expected_attack:
                attack_exact_matches += 1
            else:
                attack_deltas.append(
                    {
                        "game_id": game_id,
                        "move_number": int(step.get("move_number", step_idx + 1)),
                        "expected_attack": expected_attack,
                        "actual_attack": actual_attack,
                        "delta": actual_attack - expected_attack,
                    }
                )
            if actual_spin != expected_spin:
                spin_confusion_counts[f"{expected_spin}->{actual_spin}"] += 1

            spin = clear_stats.get("spin")
            if isinstance(spin, dict) and spin.get("piece") not in (None, "T"):
                non_t_spin_positives += 1

    attack_exact_match_rate = (
        float(attack_exact_matches) / float(samples_evaluated) if samples_evaluated > 0 else 0.0
    )
    total_spin_disagreements = int(sum(spin_confusion_counts.values()))
    t_spin_disagreement_rate = (
        float(total_spin_disagreements) / float(samples_evaluated) if samples_evaluated > 0 else 0.0
    )
    attack_deltas.sort(key=lambda item: abs(int(item["delta"])), reverse=True)

    report = {
        "cache_dir": cache_dir,
        "total_available_steps": total_available_steps,
        "sample_size_requested": sample_size,
        "samples_evaluated": samples_evaluated,
        "clear_mismatch_count": clear_mismatch_count,
        "non_t_spin_positives": non_t_spin_positives,
        "spin_confusion_counts": dict(spin_confusion_counts),
        "t_spin_disagreement_rate": t_spin_disagreement_rate,
        "attack_exact_match_rate": attack_exact_match_rate,
        "top_attack_deltas": attack_deltas[:10],
        "thresholds_met": (
            clear_mismatch_count == 0
            and non_t_spin_positives == 0
            and t_spin_disagreement_rate <= 0.005
            and attack_exact_match_rate >= 0.97
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="game_cache_v3")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = run_alignment_report(
        cache_dir=args.cache_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
