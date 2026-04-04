import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tetrisformer.fileParsing import (
    DATA_PATH,
    INDEX_PATH,
    find_bfs_match_index,
    iter_game_frames,
    load_index,
    playfield_to_board,
)
from tetrisEngine import TetrisEngine


CACHE_VERSION = 3
DEFAULT_CACHE_DIR = "game_cache_v3"
DEFAULT_SPIN_MODE = "t_only"


def _safe_piece_str(piece) -> str:
    if piece is None:
        return ""
    text = str(piece)
    if not text or text == "N" or text.lower() == "nan":
        return ""
    return text


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _queue_state_from_row(row: dict) -> dict:
    return {
        "current": _safe_piece_str(row.get("placed")),
        "hold": _safe_piece_str(row.get("hold")) or None,
        "next_queue": str(row.get("next") or ""),
    }


def _pre_state_from_engine(engine: TetrisEngine, row: dict) -> dict:
    return {
        "combo": int(getattr(engine, "combo", 0)),
        "combo_active": bool(getattr(engine, "combo_active", False)),
        "b2b_chain": int(getattr(engine, "b2b_chain", 0)),
        "surge_charge": int(getattr(engine, "surge_charge", 0)),
        "incoming_garbage_total": _safe_int(row.get("incoming_garbage", 0)),
    }


def _expert_replay_from_row(row: dict) -> dict:
    return {
        "x": _safe_int(row.get("x", 0)),
        "y": _safe_int(row.get("y", 0)),
        "r": str(row.get("r") or ""),
        "t_spin": str(row.get("t_spin") or "N") or "N",
        "attack": _safe_int(row.get("attack", 0)),
        "cleared": _safe_int(row.get("cleared", 0)),
    }


def _restore_engine_step_state(
    engine: TetrisEngine,
    *,
    base_board: np.ndarray,
    queue_state: dict,
    incoming_garbage_total: int,
) -> None:
    current = _safe_piece_str(queue_state.get("current"))
    if not current:
        raise RuntimeError("Missing current piece in queue_state.")

    engine.board = base_board.copy()
    engine.current_piece = None
    engine.hold = _safe_piece_str(queue_state.get("hold")) or None
    engine.bag = np.array([], dtype=int)
    engine.bag_size = 0
    engine.game_over = False
    engine.game_over_reason = None
    engine.last_clear_stats = None
    engine.last_spawn_was_clutch = False
    engine.last_end_phase = None
    engine.incoming_garbage = []
    engine.garbage_col = None
    if incoming_garbage_total > 0:
        engine.incoming_garbage.append(
            {"lines": int(incoming_garbage_total), "timer": 0, "col": 0}
        )
    engine.current_piece = engine.spawn_piece(current)
    if engine.current_piece is None:
        raise RuntimeError(f"Failed to spawn piece '{current}'")


def _apply_placement_to_current_piece(engine: TetrisEngine, placement: dict) -> bool:
    piece = engine.current_piece
    if piece is None:
        return False
    piece.position = (
        int(placement.get("x", piece.position[0])),
        int(placement.get("y", piece.position[1])),
    )
    piece.rotation = int(placement.get("rotation", 0)) % 4
    piece.last_action_was_rotation = bool(placement.get("last_was_rot", False))
    last_dir = placement.get("last_rot_dir")
    piece.last_rotation_dir = None if last_dir in (None, 0) else int(last_dir)
    last_kick = placement.get("last_kick_idx")
    piece.last_kick_index = None if last_kick is None else (None if int(last_kick) < 0 else int(last_kick))
    return engine.is_position_valid(piece, position=piece.position, rotation=piece.rotation)


def _advance_engine_with_placement(
    engine: TetrisEngine,
    placement: dict,
    next_piece_kind: str,
) -> bool:
    if not _apply_placement_to_current_piece(engine, placement):
        engine.game_over = True
        engine.game_over_reason = "invalid_placement"
        return False

    engine.lock_piece(run_end_phase=False)
    if engine.game_over:
        return True

    if next_piece_kind:
        spawned = engine.spawn_piece(next_piece_kind)
        if spawned is None:
            engine.current_piece = None
            engine.game_over = True
            engine.game_over_reason = "block_out"
            return True
        can_spawn = engine.is_position_valid(spawned, position=spawned.position)
        if can_spawn:
            engine.current_piece = spawned
            engine.game_over = False
            engine.game_over_reason = None
        else:
            engine.current_piece = None
            engine.game_over = True
            engine.game_over_reason = "block_out"
    else:
        engine.current_piece = None
    return True


def _collect_move_rows(entry: dict, data_path: str) -> list[dict]:
    frames = list(iter_game_frames(entry, data_path=data_path))
    if len(frames) < 2:
        return []

    move_rows = []
    previous = None
    move_number = 0
    for frame in frames:
        if previous is None:
            previous = frame
            continue

        row = previous
        next_row = frame
        previous = frame

        placed = _safe_piece_str(row.get("placed"))
        if not placed:
            continue

        move_number += 1
        move_rows.append(
            {
                "row": row,
                "next_row": next_row,
                "placed": placed,
                "move_number": move_number,
            }
        )
    return move_rows


def build_dataset(
    *,
    data_path: str,
    index_path: str,
    cache_dir: str,
    overwrite: bool,
    limit_games: int,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    entries = load_index(index_path=index_path)
    if limit_games > 0:
        entries = entries[:limit_games]

    print(f"Found {len(entries)} games. Generating BFS cache v{CACHE_VERSION}...")

    for entry in tqdm(entries, desc="Precomputing BFS & Matches"):
        game_id = entry["game_id"]
        save_path = os.path.join(cache_dir, f"{game_id}.pt")
        if os.path.exists(save_path) and not overwrite:
            continue

        move_rows = _collect_move_rows(entry, data_path=data_path)
        if not move_rows:
            continue

        processed_steps = []
        expert_engine = TetrisEngine(spin_mode=DEFAULT_SPIN_MODE)

        for step_idx, step in enumerate(move_rows):
            row = step["row"]
            next_row = step["next_row"]
            placed = step["placed"]
            move_number = step["move_number"]

            base_board = playfield_to_board(row.get("playfield", "")).astype(np.uint8)
            target_board = playfield_to_board(next_row.get("playfield", "")).astype(np.uint8)
            queue_state = _queue_state_from_row(row)
            pre_state = _pre_state_from_engine(expert_engine, row)

            try:
                _restore_engine_step_state(
                    expert_engine,
                    base_board=base_board,
                    queue_state=queue_state,
                    incoming_garbage_total=pre_state["incoming_garbage_total"],
                )
                bfs_results = expert_engine.bfs_all_placements(include_no_place=False)
            except Exception:
                continue

            if not bfs_results:
                continue

            immediate = _safe_int(row.get("immediate_garbage", 0))
            match_idx, _ = find_bfs_match_index(
                bfs_results,
                target_board,
                max_garbage=max(8, immediate),
                shift=True,
            )
            if match_idx < 0:
                continue

            valid_indices = [
                i
                for i, result in enumerate(bfs_results)
                if result.get("board") is not None and result.get("placement") is not None
            ]
            if match_idx not in valid_indices:
                continue

            bfs_boards = np.stack([result["board"] for result in bfs_results]).astype(np.uint8)
            bfs_placements = [result["placement"] for result in bfs_results]
            expert_replay = _expert_replay_from_row(row)

            processed_steps.append(
                {
                    "move_number": move_number,
                    "base_board": base_board,
                    "placed": placed,
                    "queue_state": queue_state,
                    "pre_state": pre_state,
                    "expert_replay": expert_replay,
                    "expert_match_index": int(match_idx),
                    "valid_indices": [int(i) for i in valid_indices],
                    "bfs_boards": bfs_boards,
                    "bfs_placements": bfs_placements,
                    "future_pieces": [
                        _safe_piece_str(move_rows[j]["placed"])
                        for j in range(step_idx + 1, min(len(move_rows), step_idx + 10))
                        if _safe_piece_str(move_rows[j]["placed"])
                    ],
                }
            )

            next_piece_kind = (
                _safe_piece_str(move_rows[step_idx + 1]["placed"])
                if step_idx + 1 < len(move_rows)
                else ""
            )
            if not _advance_engine_with_placement(
                expert_engine,
                bfs_results[match_idx]["placement"],
                next_piece_kind=next_piece_kind,
            ):
                continue

        if not processed_steps:
            continue

        torch.save(
            {
                "cache_version": CACHE_VERSION,
                "game_id": game_id,
                "steps": processed_steps,
            },
            save_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--index-path", default=INDEX_PATH)
    parser.add_argument("--cache-dir", default=os.path.join(_ROOT, DEFAULT_CACHE_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-games", type=int, default=0)
    args = parser.parse_args()

    build_dataset(
        data_path=args.data_path,
        index_path=args.index_path,
        cache_dir=args.cache_dir,
        overwrite=bool(args.overwrite),
        limit_games=int(args.limit_games),
    )


if __name__ == "__main__":
    main()
