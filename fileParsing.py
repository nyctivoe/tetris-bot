# fileParsing.py — CSV loading, byte-offset indexing, board parsing, and BFS match alignment.
#
# Primary responsibilities:
#   1. build_index / load_index  — create and read a byte-offset index over data.csv so any
#      game can be seeked to directly without scanning 7.7 GB.
#   2. load_game_frames / iter_game_frames — read one game's rows from data.csv into frame dicts.
#   3. playfield_to_board* — decode the run-length playfield string into a (40×10) numpy board.
#   4. match_bfs_results_to_playfield — identify which BFS candidate the expert actually played
#      and how much incoming garbage arrived between frames.

from tetrisEngine import (
    KIND_TO_PIECE_ID,
    PIECE_ID_TO_KIND,
    TetrisEngine,
    BOARD_WIDTH,
    BOARD_HEIGHT,
)

import argparse
import csv
import os

import numpy as np

# Numba JIT is used to accelerate the inner garbage-matching loop.
# If numba is not installed the decorator is a transparent no-op.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    NUMBA_AVAILABLE = False

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_HERE, "data.csv")
INDEX_PATH = os.path.join(_HERE, "game_index.csv")

# Maps the single-character cell codes used in the CSV playfield string to integer piece IDs.
# "N" = empty, "G" = garbage (id 8), letter pieces map to their canonical IDs.
PLAYFIELD_CHAR_TO_ID = {
    "N": 0,
    "G": 8,
    "I": KIND_TO_PIECE_ID["I"],
    "O": KIND_TO_PIECE_ID["O"],
    "T": KIND_TO_PIECE_ID["T"],
    "S": KIND_TO_PIECE_ID["S"],
    "Z": KIND_TO_PIECE_ID["Z"],
    "J": KIND_TO_PIECE_ID["J"],
    "L": KIND_TO_PIECE_ID["L"],
}

# Same as above but with J and L swapped — used for horizontal-mirror data augmentation.
PLAYFIELD_CHAR_TO_ID_SWAP_JL = {
    "N": 0,
    "G": 8,
    "I": KIND_TO_PIECE_ID["I"],
    "O": KIND_TO_PIECE_ID["O"],
    "T": KIND_TO_PIECE_ID["T"],
    "S": KIND_TO_PIECE_ID["S"],
    "Z": KIND_TO_PIECE_ID["Z"],
    "J": KIND_TO_PIECE_ID["L"],
    "L": KIND_TO_PIECE_ID["J"],
}

# Piece-level swap map for the J↔L augmentation applied to placed/hold/next fields.
PIECE_SWAP_JL = {
    "J": "L",
    "L": "J",
}


def build_index(data_path=DATA_PATH, index_path=INDEX_PATH):
    """Single-pass scan of data.csv that records the byte range of every game.

    The index stores (game_id, won, start_byte, end_byte, row_count) so that
    load_game_frames() can seek directly to any game without reading the full CSV.
    Games are identified by consecutive runs of the same (game_id, won) pair —
    the won flag is included because the same game_id can appear in both won=0
    and won=1 variants in edge-case data.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    with open(data_path, "r", encoding="utf-8", newline="") as f, open(
        index_path, "w", encoding="utf-8", newline=""
    ) as out:
        header = f.readline()
        if not header:
            raise ValueError("Empty CSV file.")

        header_cols = header.rstrip("\n").split(",")
        game_idx = header_cols.index("game_id")
        won_idx = header_cols.index("won")

        writer = csv.writer(out)
        writer.writerow(["game_id", "won", "start", "end", "rows"])

        prev_game = None
        prev_won = None
        start_offset = f.tell()
        row_count = 0
        skipped_rows = 0
        parse_errors = 0
        skipped_game_ids = set()

        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break

            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(game_idx, won_idx):
                skipped_rows += 1
                # Try to extract game_id even from incomplete row for diagnostics
                try:
                    if game_idx < len(parts):
                        gid = int(parts[game_idx])
                        skipped_game_ids.add(gid)
                except (ValueError, IndexError):
                    pass
                continue

            try:
                game_id = int(parts[game_idx])
                won = int(parts[won_idx])
            except (ValueError, IndexError) as e:
                parse_errors += 1
                # Try to extract game_id for diagnostics
                try:
                    if game_idx < len(parts) and parts[game_idx].strip():
                        gid = int(parts[game_idx])
                        skipped_game_ids.add(gid)
                except (ValueError, IndexError):
                    pass
                continue

            if prev_game is None:
                prev_game = game_id
                prev_won = won
                start_offset = offset
                row_count = 1
                continue
            if skipped_game_ids:
                print(f"Skipped game IDs: {sorted(skipped_game_ids)}")

            if game_id != prev_game or won != prev_won:
                writer.writerow([prev_game, prev_won, start_offset, offset, row_count])
                prev_game = game_id
                prev_won = won
                start_offset = offset
                row_count = 1
            else:
                row_count += 1

        if prev_game is not None:
            writer.writerow([prev_game, prev_won, start_offset, f.tell(), row_count])

        if skipped_rows > 0 or parse_errors > 0:
            print(f"Warning: Skipped {skipped_rows} rows (insufficient columns), {parse_errors} rows (parse errors)")


def load_index(index_path=INDEX_PATH):
    """Load the game index CSV into a list of dicts.

    Returns an empty list (not an error) when the index file does not yet exist,
    so callers can check length rather than catching exceptions.
    Each entry has keys: game_id, won, start, end, rows.
    """
    if not os.path.exists(index_path):
        return []
    with open(index_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        entries = []
        for row in reader:
            entries.append(
                {
                    "game_id": int(row["game_id"]),
                    "won": int(row["won"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "rows": int(row["rows"]),
                }
            )
        return entries


def find_game_entry(game_id, won=None, index_path=INDEX_PATH):
    """Return the index entry for a specific game, or None if not found.

    Pass won=1/0 to disambiguate if the same game_id appears for both outcomes.
    """
    entries = load_index(index_path)
    for entry in entries:
        if entry["game_id"] == game_id and (won is None or entry["won"] == won):
            return entry
    return None


def load_game_frames(
    game_id,
    won=None,
    data_path=DATA_PATH,
    index_path=INDEX_PATH,
    max_frames=None,
    swap_jl=False,
):
    """Load all move frames for a game into a list of dicts.

    Uses the byte-offset index to seek directly into data.csv — avoids a full scan.
    swap_jl=True applies the J↔L mirror augmentation to placed/hold/next fields and
    the board character map; used to double effective training data.
    Returns (frames_list, index_entry), or (None, None) if the game is not indexed.
    max_frames caps how many moves are loaded (useful for debugging).
    """
    entry = find_game_entry(game_id, won=won, index_path=index_path)
    if entry is None:
        return None, None

    with open(data_path, "r", encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split(",")
        idx = {name: i for i, name in enumerate(header)}
        f.seek(entry["start"])

        frames = []
        while f.tell() < entry["end"]:
            line = f.readline()
            if not line:
                break
            parts = line.rstrip("\n").split(",")
            if len(parts) < len(header):
                continue

            placed = _get(parts, idx, "placed", "")
            hold = _get(parts, idx, "hold", "")
            next_queue = _get(parts, idx, "next", "")
            if swap_jl:
                placed = _swap_piece_char(placed)
                hold = _swap_piece_char(hold)
                next_queue = _swap_queue(next_queue)

            frame = {
                "game_id": int(parts[idx["game_id"]]),
                "subframe": int(parts[idx["subframe"]]),
                "won": int(parts[idx["won"]]),
                "playfield": _get(parts, idx, "playfield", ""),
                "x": _parse_int(_get(parts, idx, "x", 0)),
                "y": _parse_int(_get(parts, idx, "y", 0)),
                "r": _get(parts, idx, "r", ""),
                "placed": placed,
                "hold": hold,
                "next": next_queue,
                "cleared": _parse_int(_get(parts, idx, "cleared", 0)),
                "garbage_cleared": _parse_int(_get(parts, idx, "garbage_cleared", 0)),
                "attack": _parse_int(_get(parts, idx, "attack", 0)),
                "t_spin": _get(parts, idx, "t_spin", ""),
                "btb": _parse_int(_get(parts, idx, "btb", 0)),
                "combo": _parse_int(_get(parts, idx, "combo", 0)),
                "immediate_garbage": _parse_int(
                    _get(parts, idx, "immediate_garbage", 0)
                ),
                "incoming_garbage": _parse_int(
                    _get(parts, idx, "incoming_garbage", 0)
                ),
                "rating": _parse_float(_get(parts, idx, "rating", 0.0)),
                "glicko": _parse_float(_get(parts, idx, "glicko", 0.0)),
                "glicko_rd": _parse_float(_get(parts, idx, "glicko_rd", 0.0)),
            }

            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break

    return frames, entry


def iter_game_frames(entry, data_path=DATA_PATH, swap_jl=False):
    """Generator version of load_game_frames — yields one frame dict at a time.

    Preferred for training pipelines where holding all frames in memory is wasteful.
    Takes a pre-looked-up index entry rather than a game_id so the caller controls
    the index lookup and can apply filtering before iterating.
    """
    if entry is None:
        return
    with open(data_path, "r", encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split(",")
        idx = {name: i for i, name in enumerate(header)}
        f.seek(entry["start"])

        while f.tell() < entry["end"]:
            line = f.readline()
            if not line:
                break
            parts = line.rstrip("\n").split(",")
            if len(parts) < len(header):
                continue

            placed = _get(parts, idx, "placed", "")
            hold = _get(parts, idx, "hold", "")
            next_queue = _get(parts, idx, "next", "")
            if swap_jl:
                placed = _swap_piece_char(placed)
                hold = _swap_piece_char(hold)
                next_queue = _swap_queue(next_queue)

            frame = {
                "game_id": int(parts[idx["game_id"]]),
                "subframe": int(parts[idx["subframe"]]),
                "won": int(parts[idx["won"]]),
                "playfield": _get(parts, idx, "playfield", ""),
                "x": _parse_int(_get(parts, idx, "x", 0)),
                "y": _parse_int(_get(parts, idx, "y", 0)),
                "r": _get(parts, idx, "r", ""),
                "placed": placed,
                "hold": hold,
                "next": next_queue,
                "cleared": _parse_int(_get(parts, idx, "cleared", 0)),
                "garbage_cleared": _parse_int(_get(parts, idx, "garbage_cleared", 0)),
                "attack": _parse_int(_get(parts, idx, "attack", 0)),
                "t_spin": _get(parts, idx, "t_spin", ""),
                "btb": _parse_int(_get(parts, idx, "btb", 0)),
                "combo": _parse_int(_get(parts, idx, "combo", 0)),
                "immediate_garbage": _parse_int(
                    _get(parts, idx, "immediate_garbage", 0)
                ),
                "incoming_garbage": _parse_int(
                    _get(parts, idx, "incoming_garbage", 0)
                ),
                "rating": _parse_float(_get(parts, idx, "rating", 0.0)),
                "glicko": _parse_float(_get(parts, idx, "glicko", 0.0)),
                "glicko_rd": _parse_float(_get(parts, idx, "glicko_rd", 0.0)),
            }

            yield frame

def playfield_to_board(playfield_str):
    """Decode the CSV playfield string into a (BOARD_HEIGHT × BOARD_WIDTH) numpy int array.

    The playfield string is a flat sequence of single-char cell codes stored bottom-row-first
    (first BOARD_WIDTH chars = bottom row). If its length is not a multiple of BOARD_WIDTH,
    trailing 'N' (empty) chars are appended. Rows are written into the board in reverse so
    that board[0] is the bottom and board[BOARD_HEIGHT-1] is the top, matching the engine.
    Returns a zero board for None / empty / NaN inputs.
    """
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
    if playfield_str is None:
        return board
    s = str(playfield_str).strip()
    if not s or s.lower() == "nan":
        return board
    pad = (-len(s)) % BOARD_WIDTH
    if pad:
        s = s + ("N" * pad)
    rows = [s[i:i + BOARD_WIDTH] for i in range(0, len(s), BOARD_WIDTH)]
    for i, row in enumerate(rows):
        y = BOARD_HEIGHT - 1 - i
        if y < 0:
            break
        for x, ch in enumerate(row):
            value = PLAYFIELD_CHAR_TO_ID.get(ch, 0)
            if value:
                board[y][x] = value
    return board


def playfield_to_board_swapped(playfield_str):
    """Same as playfield_to_board but uses PLAYFIELD_CHAR_TO_ID_SWAP_JL for J↔L augmentation."""
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
    if playfield_str is None:
        return board
    s = str(playfield_str).strip()
    if not s or s.lower() == "nan":
        return board
    pad = (-len(s)) % BOARD_WIDTH
    if pad:
        s = s + ("N" * pad)
    rows = [s[i:i + BOARD_WIDTH] for i in range(0, len(s), BOARD_WIDTH)]
    for i, row in enumerate(rows):
        y = BOARD_HEIGHT - 1 - i
        if y < 0:
            break
        for x, ch in enumerate(row):
            value = PLAYFIELD_CHAR_TO_ID_SWAP_JL.get(ch, 0)
            if value:
                board[y][x] = value
    return board


def _apply_garbage_rows(base_board, target_board, garbage_rows, shift=True):
    """Reconstruct the board state after garbage_rows lines of incoming garbage.

    shift=True (standard): existing rows are pushed upward by garbage_rows, rows that
      overflow the top are lost, and the bottom garbage_rows rows are copied from target_board.
    shift=False: existing rows stay in place; only the bottom garbage_rows rows are
      replaced with those from target_board (used when the board wasn't shifted).
    Returns a new board array; does not mutate inputs.
    """
    if garbage_rows <= 0:
        return base_board.copy()
    new_board = np.zeros_like(base_board)
    if shift:
        if garbage_rows < BOARD_HEIGHT:
            new_board[: BOARD_HEIGHT - garbage_rows] = base_board[garbage_rows:]
        new_board[BOARD_HEIGHT - garbage_rows :] = target_board[
            BOARD_HEIGHT - garbage_rows :
        ]
    else:
        new_board[:, :] = base_board
        new_board[BOARD_HEIGHT - garbage_rows :] = target_board[
            BOARD_HEIGHT - garbage_rows :
        ]
    return new_board


@njit()
def _match_garbage_rows_fast(base_board, target_board, max_garbage, shift):
    """Numba-JIT inner loop for _match_garbage_rows — identical logic, runs ~10–50× faster.

    Tries garbage_rows = 0, 1, 2, … until the non-garbage portion of target_board matches
    base_board shifted (or unshifted). Returns the first matching count, or -1 on failure.
    """
    height = base_board.shape[0]
    width = base_board.shape[1]
    if max_garbage > height:
        max_garbage = height
    if max_garbage < 0:
        max_garbage = 0
    for garbage_rows in range(0, max_garbage + 1):
        limit = height - garbage_rows
        ok = True
        if shift:
            for y in range(limit):
                by = y + garbage_rows
                for x in range(width):
                    if target_board[y, x] != base_board[by, x]:
                        ok = False
                        break
                if not ok:
                    break
        else:
            for y in range(limit):
                for x in range(width):
                    if target_board[y, x] != base_board[y, x]:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            return garbage_rows
    return -1


def _match_garbage_rows(base_board, target_board, max_garbage=None, shift=True):
    """Python dispatcher for the garbage-row count search.

    Determines how many garbage rows arrived between two frames by finding the
    smallest garbage_rows value (0..max_garbage) that makes base_board consistent
    with target_board (after shifting existing rows up by that amount).
    Returns the matched count, or None if no consistent count was found.
    Delegates to the Numba JIT path when available for speed.
    """
    if max_garbage is None:
        max_garbage = BOARD_HEIGHT
    max_garbage = max(0, min(max_garbage, BOARD_HEIGHT))
    if NUMBA_AVAILABLE:
        matched = _match_garbage_rows_fast(base_board, target_board, max_garbage, shift)
        return None if matched < 0 else int(matched)
    for garbage_rows in range(0, max_garbage + 1):
        if shift:
            if np.array_equal(
                target_board[: BOARD_HEIGHT - garbage_rows],
                base_board[garbage_rows:],
            ):
                return garbage_rows
        else:
            if np.array_equal(
                target_board[: BOARD_HEIGHT - garbage_rows],
                base_board[: BOARD_HEIGHT - garbage_rows],
            ):
                return garbage_rows
    return None


def match_bfs_results_to_playfield(
    bfs_results,
    playfield_str,
    max_garbage=None,
    shift=True,
    swap_jl=False,
):
    """Align BFS candidates against the next frame's playfield to identify the expert move.

    For each BFS result (a candidate placement that includes a post-placement board),
    checks whether that board is consistent with the observed next-frame playfield
    (accounting for up to max_garbage rows of incoming garbage).

    Returns a list of dicts — one per matching candidate — each containing:
      - "result":            the original BFS result dict
      - "garbage_rows":      how many garbage rows arrived (0 if none)
      - "board_with_garbage": the candidate board with garbage rows patched in

    Typically only one candidate matches (the expert's actual move). Multiple matches
    can occur when placements produce identical boards.
    """
    target_board = (
        playfield_to_board_swapped(playfield_str)
        if swap_jl
        else playfield_to_board(playfield_str)
    )
    matches = []
    for result in bfs_results:
        base_board = result.get("board")
        if base_board is None:
            continue
        garbage_rows = _match_garbage_rows(
            base_board, target_board, max_garbage=max_garbage, shift=shift
        )
        if garbage_rows is None:
            continue
        board_with_garbage = _apply_garbage_rows(
            base_board, target_board, garbage_rows, shift=shift
        )
        matches.append(
            {
                "result": result,
                "garbage_rows": garbage_rows,
                "board_with_garbage": board_with_garbage,
            }
        )
    return matches


def has_bfs_match_to_board(
    bfs_results,
    target_board,
    max_garbage=None,
    shift=True,
):
    """Return True if any BFS candidate is consistent with target_board (fast existence check).

    Accepts a pre-decoded numpy board rather than a playfield string, unlike
    match_bfs_results_to_playfield. Used to filter frames where no expert move can be found.
    """
    for result in bfs_results:
        base_board = result.get("board")
        if base_board is None:
            continue
        garbage_rows = _match_garbage_rows(
            base_board, target_board, max_garbage=max_garbage, shift=shift
        )
        if garbage_rows is not None:
            return True
    return False


def find_bfs_match_index(
    bfs_results,
    target_board,
    max_garbage=None,
    shift=True,
):
    """Return (index, garbage_rows) for the first BFS candidate matching target_board.

    Returns (-1, None) if no candidate matches. The index is into bfs_results and
    is used downstream to mark the expert's move in the training sample.
    """
    for idx, result in enumerate(bfs_results):
        base_board = result.get("board")
        if base_board is None:
            continue
        garbage_rows = _match_garbage_rows(
            base_board, target_board, max_garbage=max_garbage, shift=shift
        )
        if garbage_rows is not None:
            return idx, int(garbage_rows)
    return -1, None


def _swap_piece_char(ch):
    """Swap J↔L in a single-character piece name; return the char unchanged for all others."""
    if ch is None:
        return ""
    s = str(ch)
    if not s:
        return s
    if len(s) == 1:
        return PIECE_SWAP_JL.get(s, s)
    return s


def _swap_queue(queue):
    """Apply J↔L swap to every character in the next-piece queue string."""
    if queue is None:
        return ""
    s = str(queue)
    if not s:
        return s
    return "".join(PIECE_SWAP_JL.get(ch, ch) for ch in s)


def _parse_int(value, default=0):
    """Safe int cast — returns default on None, empty string, or non-numeric CSV fields."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value, default=0.0):
    """Safe float cast — returns default on None, empty string, or non-numeric CSV fields."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get(parts, index_map, key, default=None):
    """Safe column accessor: looks up the column index by name and returns parts[idx].

    Returns default if the column name is unknown or the row is shorter than expected.
    """
    idx = index_map.get(key)
    if idx is None or idx >= len(parts):
        return default
    return parts[idx]

