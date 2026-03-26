import os
import torch
import numpy as np
from tqdm import tqdm

from fileParsing import load_index, iter_game_frames, playfield_to_board, find_bfs_match_index, DATA_PATH
from tetrisEngine import TetrisEngine

CACHE_DIR = "game_cache_v2"

# --- Helpers reproduced here so the script is completely standalone ---
def _safe_piece_str(p) -> str:
    if p is None: return ""
    s = str(p)
    if not s or s == "N" or s.lower() == "nan": return ""
    return s

def _safe_int(v, default=0):
    try: return int(v)
    except Exception: return default

def _set_engine_root_state(engine, row: dict, base_board: np.ndarray, placed_kind: str):
    engine.board = base_board.copy()
    engine.combo = _safe_int(row.get("combo", 0))
    engine.combo_active = bool(engine.combo > 0)
    engine.b2b_chain = _safe_int(row.get("btb", row.get("b2b_chain", 0)))
    engine.hold = _safe_piece_str(row.get("hold")) or None
    engine.game_over = False
    engine.game_over_reason = None
    engine.last_clear_stats = None
    engine.current_piece = engine.spawn_piece(placed_kind)
    if engine.current_piece is None:
        raise RuntimeError(f"Failed to spawn piece '{placed_kind}'")
    return engine

def build_dataset():
    os.makedirs(CACHE_DIR, exist_ok=True)
    entries = load_index()
    
    print(f"Found {len(entries)} games. Generating BFS cache...")
    
    for entry in tqdm(entries, desc="Precomputing BFS & Matches"):
        game_id = entry["game_id"]
        save_path = os.path.join(CACHE_DIR, f"{game_id}.pt")
        
        if os.path.exists(save_path):
            continue

        frames = list(iter_game_frames(entry, data_path=DATA_PATH))
        if len(frames) < 2:
            continue

        # Group into moves
        move_rows = []
        prev = None
        move_number = 0
        for frame in frames:
            if prev is None:
                prev = frame
                continue
            row, next_row = prev, frame
            prev = frame
            
            placed = _safe_piece_str(row.get("placed"))
            if not placed:
                continue
                
            move_number += 1
            move_rows.append({
                "row": row,
                "next_row": next_row,
                "placed": placed,
                "move_number": move_number,
            })

        if not move_rows:
            continue

        processed_steps = []
        engine = TetrisEngine()

        for step_idx, step in enumerate(move_rows):
            row = step["row"]
            next_row = step["next_row"]
            placed = step["placed"]
            
            base_board = playfield_to_board(row.get("playfield", "")).astype(np.uint8)
            target_board = playfield_to_board(next_row.get("playfield", "")).astype(np.uint8)

            try:
                _set_engine_root_state(engine, row, base_board, placed)
                bfs_results = engine.bfs_all_placements(include_no_place=False)
            except Exception:
                continue

            if not bfs_results:
                continue

            # Find the human match
            immediate = _safe_int(row.get("immediate_garbage", 0))
            match_idx, _ = find_bfs_match_index(
                bfs_results, target_board, max_garbage=max(8, immediate), shift=True
            )
            
            if match_idx < 0:
                continue
                
            valid_indices = [
                i for i, res in enumerate(bfs_results)
                if res.get("board") is not None and res.get("placement") is not None
            ]
            if match_idx not in valid_indices:
                continue

            # Pack it down to save disk space and I/O time
            bfs_boards = np.stack([res["board"] for res in bfs_results]).astype(np.uint8)
            bfs_placements = [res["placement"] for res in bfs_results]

            processed_steps.append({
                "row": row,
                "placed": placed,
                "move_number": step["move_number"],
                "base_board": base_board,
                "expert_match_index": match_idx,
                "valid_indices": valid_indices,
                "bfs_boards": bfs_boards,
                "bfs_placements": bfs_placements,
                # We save future pieces so the Q-rollout still has the ground truth sequence
                "future_pieces": [
                    _safe_piece_str(move_rows[j]["placed"])
                    for j in range(step_idx + 1, min(len(move_rows), step_idx + 10))
                    if _safe_piece_str(move_rows[j]["placed"])
                ]
            })

        if processed_steps:
            torch.save({
                "game_id": game_id,
                "steps": processed_steps
            }, save_path)

if __name__ == "__main__":
    build_dataset()