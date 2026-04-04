from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pygame
import torch

from beam_search import BeamSearchConfig, beam_search_select, generate_candidates
from mcts import MctsConfig, run_mcts
from model_v2 import TetrisZeroNet, load_checkpoint
from pvp_game import PvpGameConfig, advance_turn_start, run_pvp_turn
from tetrisEngine import BOARD_WIDTH, HIDDEN_ROWS, KIND_TO_PIECE_ID, PIECE_DEFS, PIECE_ID_TO_KIND, VISIBLE_HEIGHT, TetrisEngine


CELL = 22
BOARD_PX_W = BOARD_WIDTH * CELL
BOARD_PX_H = VISIBLE_HEIGHT * CELL
PANEL_W = 220
GAP = 20
MARGIN = 16
TOP_BAR_H = 42
WINDOW_W = MARGIN + PANEL_W + GAP + BOARD_PX_W + GAP + GAP + BOARD_PX_W + GAP + PANEL_W + MARGIN
WINDOW_H = TOP_BAR_H + BOARD_PX_H + 2 * MARGIN
FPS = 60

COL_BG = (17, 22, 28)
COL_PANEL = (25, 32, 39)
COL_GRID = (50, 60, 70)
COL_TEXT = (230, 234, 238)
COL_DIM = (162, 170, 178)
COL_ACCENT = (199, 142, 58)
COL_BORDER = (82, 94, 107)
COL_PENDING = (168, 78, 78)
COL_GO = (77, 151, 89)
COL_PAUSE = (176, 140, 54)
COL_GARBAGE = (96, 96, 96)

PIECE_COLORS = {
    1: (89, 201, 216),
    2: (230, 214, 90),
    3: (163, 104, 212),
    4: (102, 191, 106),
    5: (219, 88, 88),
    6: (90, 128, 209),
    7: (215, 152, 88),
    8: COL_GARBAGE,
}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_paths(
    model: str | None,
    model_a: str | None,
    model_b: str | None,
) -> tuple[str | None, str | None]:
    if model is not None:
        model_a = model if model_a is None else model_a
        model_b = model if model_b is None else model_b
    if model_a is None and model_b is not None:
        model_a = model_b
    if model_b is None and model_a is not None:
        model_b = model_a
    return model_a, model_b


@dataclass
class SearchPlayer:
    name: str
    checkpoint_path: str | None
    device: str
    mcts_cfg: MctsConfig
    model: TetrisZeroNet | None = None
    checkpoint_name: str = "heuristic"
    last_choice: dict[str, Any] | None = None
    last_latency: float = 0.0

    def __post_init__(self) -> None:
        if self.checkpoint_path is None:
            return
        bundle = load_checkpoint(self.checkpoint_path, device=self.device)
        self.model = bundle["model"]
        self.checkpoint_name = Path(self.checkpoint_path).name

    @property
    def mode_label(self) -> str:
        if not self.mcts_cfg.enabled:
            return "beam"
        return "model-mcts" if self.model is not None else "heuristic-mcts"

    def select_action(self, engine: TetrisEngine, opponent_engine: TetrisEngine, move_number: int) -> dict[str, Any]:
        candidates = generate_candidates(engine, include_hold=self.mcts_cfg.beam_cfg.include_hold)
        if not candidates:
            raise RuntimeError(f"{self.name} has no legal candidates.")
        start = time.perf_counter()
        if self.mcts_cfg.enabled:
            result = run_mcts(
                engine,
                opponent_engine,
                move_number,
                self.model,
                candidates,
                cfg=self.mcts_cfg,
            )
            choice = result["selected_candidate"]
            self.last_choice = {
                "mode": self.mode_label,
                "candidate_index": int(choice["candidate_index"]),
                "selected_index": int(result["selected_index"]),
                "visits": result["visits"].copy(),
                "priors": result["priors"].copy(),
                "q_values": result["q_values"].copy(),
            }
        else:
            choice = beam_search_select(
                engine,
                opponent_engine=opponent_engine,
                move_number=move_number,
                cfg=self.mcts_cfg.beam_cfg,
            )
            self.last_choice = {
                "mode": "beam",
                "candidate_index": int(choice["candidate_index"]),
                "score": float(choice.get("sequence_score", choice.get("immediate_score", 0.0))),
            }
        self.last_latency = time.perf_counter() - start
        return dict(choice["placement"])


def build_search_player(
    name: str,
    checkpoint_path: str | None,
    *,
    device: str,
    simulations: int,
    temperature: float,
    max_depth: int,
    visible_queue_depth: int,
    c_puct: float,
    include_hold: bool,
    beam_depth: int,
    beam_width: int,
    disable_mcts: bool = False,
) -> SearchPlayer:
    beam_cfg = BeamSearchConfig(depth=int(beam_depth), width=int(beam_width), include_hold=bool(include_hold))
    mcts_cfg = MctsConfig(
        enabled=not disable_mcts,
        simulations=int(simulations),
        c_puct=float(c_puct),
        temperature=float(temperature),
        beam_cfg=beam_cfg,
        max_depth=int(max_depth),
        visible_queue_depth=int(visible_queue_depth),
    )
    return SearchPlayer(
        name=name,
        checkpoint_path=checkpoint_path,
        device=device,
        mcts_cfg=mcts_cfg,
    )


def _draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, color: tuple[int, int, int], x: int, y: int) -> int:
    surf = font.render(text, True, color)
    surface.blit(surf, (x, y))
    return surf.get_height()


def _draw_mini_piece(surface: pygame.Surface, kind: str, x: int, y: int, cell: int = 14) -> None:
    blocks = PIECE_DEFS[kind]["blocks"]
    size = PIECE_DEFS[kind]["size"]
    pid = KIND_TO_PIECE_ID.get(kind, 0)
    color = PIECE_COLORS.get(pid, COL_TEXT)
    for bx, by in blocks:
        px = x + int((bx - size / 2 + 0.5) * cell)
        py = y + int((by - size / 2 + 0.5) * cell)
        rect = pygame.Rect(px, py, cell - 1, cell - 1)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, tuple(min(255, c + 28) for c in color), rect, 1)


class TetrisZeroPvpUI:
    def __init__(
        self,
        player_a: SearchPlayer,
        player_b: SearchPlayer,
        *,
        seed: int,
        delay_seconds: float,
        cfg: PvpGameConfig,
    ) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("TetrisZero PvP Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 15)
        self.font_small = pygame.font.SysFont("Consolas", 13)
        self.font_big = pygame.font.SysFont("Consolas", 21, bold=True)

        self.player_a = player_a
        self.player_b = player_b
        self.cfg = cfg
        self.seed = int(seed)
        self.delay_seconds = float(delay_seconds)
        self.auto_run = True
        self.turn_timer = 0.0
        self.turn_index = 1
        self.current_side = "a"
        self.turns: list[dict[str, Any]] = []
        self.winner: str | None = None
        self.termination = "in_progress"

        self.engine_a = TetrisEngine(spin_mode="all_spin")
        self.engine_b = TetrisEngine(spin_mode="all_spin")
        self._reset_match()

    def _reset_match(self) -> None:
        master_rng = torch.Generator()
        del master_rng
        import numpy as np

        rng = np.random.default_rng(self.seed)
        self.engine_a = TetrisEngine(spin_mode="all_spin", rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
        self.engine_b = TetrisEngine(spin_mode="all_spin", rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
        self.engine_a.spawn_next(allow_clutch=True)
        self.engine_b.spawn_next(allow_clutch=True)
        self.turn_index = 1
        self.current_side = "a"
        self.turn_timer = 0.0
        self.turns.clear()
        self.winner = None
        self.termination = "in_progress"
        self.player_a.last_choice = None
        self.player_b.last_choice = None
        self.player_a.last_latency = 0.0
        self.player_b.last_latency = 0.0

    def _active_triplet(self) -> tuple[SearchPlayer, TetrisEngine, TetrisEngine]:
        if self.current_side == "a":
            return self.player_a, self.engine_a, self.engine_b
        return self.player_b, self.engine_b, self.engine_a

    def _advance_turn(self) -> None:
        if self.winner is not None:
            return
        if self.turn_index > self.cfg.max_plies:
            self.termination = "max_plies"
            return
        player, active, passive = self._active_triplet()
        planning_active = active.clone()
        planning_passive = passive.clone()
        preview = advance_turn_start(planning_active)
        if preview["terminated"]:
            action: dict[str, Any] = {}
            player.last_latency = 0.0
        else:
            action = player.select_action(planning_active, planning_passive, self.turn_index)
        turn = run_pvp_turn(active, passive, action, self.turn_index, self.cfg)
        turn["player"] = self.current_side
        turn["decision_latency"] = float(player.last_latency)
        self.turns.append(turn)

        if active.game_over:
            self.winner = "b" if self.current_side == "a" else "a"
            self.termination = active.game_over_reason or "top_out"
            return
        if passive.game_over:
            self.winner = self.current_side
            self.termination = passive.game_over_reason or "top_out"
            return

        if self.current_side == "a":
            self.current_side = "b"
        else:
            self.current_side = "a"
            self.turn_index += 1

    def _draw_board(self, engine: TetrisEngine, origin_x: int, origin_y: int) -> None:
        visible = engine.board[HIDDEN_ROWS:]
        for y in range(VISIBLE_HEIGHT):
            for x in range(BOARD_WIDTH):
                rect = pygame.Rect(origin_x + x * CELL, origin_y + y * CELL, CELL - 1, CELL - 1)
                value = int(visible[y, x])
                if value == 0:
                    pygame.draw.rect(self.screen, COL_GRID, rect, 1)
                else:
                    color = PIECE_COLORS.get(value, COL_TEXT)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, tuple(min(255, c + 30) for c in color), rect, 1)
        piece = engine.current_piece
        if piece is not None:
            pid = KIND_TO_PIECE_ID.get(str(piece.kind), 0)
            color = PIECE_COLORS.get(pid, COL_TEXT)
            for bx, by in engine.piece_blocks(piece):
                vis_y = int(by) - HIDDEN_ROWS
                if 0 <= vis_y < VISIBLE_HEIGHT and 0 <= int(bx) < BOARD_WIDTH:
                    rect = pygame.Rect(origin_x + int(bx) * CELL, origin_y + vis_y * CELL, CELL - 1, CELL - 1)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, tuple(min(255, c + 40) for c in color), rect, 1)
        pygame.draw.rect(self.screen, COL_BORDER, (origin_x - 2, origin_y - 2, BOARD_PX_W + 4, BOARD_PX_H + 4), 2)

    def _draw_player_panel(self, player: SearchPlayer, engine: TetrisEngine, x: int, y: int, active: bool) -> None:
        panel_rect = pygame.Rect(x, y, PANEL_W, BOARD_PX_H)
        pygame.draw.rect(self.screen, COL_PANEL, panel_rect)
        pygame.draw.rect(self.screen, COL_ACCENT if active else COL_BORDER, panel_rect, 2)
        cursor_y = y + 10
        cursor_y += _draw_text(self.screen, self.font_big, player.name, COL_TEXT, x + 10, cursor_y)
        cursor_y += 4
        cursor_y += _draw_text(self.screen, self.font_small, player.mode_label, COL_ACCENT, x + 10, cursor_y)
        cursor_y += 4
        cursor_y += _draw_text(self.screen, self.font_small, player.checkpoint_name, COL_DIM, x + 10, cursor_y)
        cursor_y += 18

        stats = [
            f"Pieces: {engine.pieces_placed}",
            f"Lines:  {engine.total_lines_cleared}",
            f"Atk:    {engine.total_attack_sent}",
            f"Cancel: {engine.total_attack_canceled}",
            f"Combo:  {engine.combo}",
            f"B2B:    {engine.b2b_chain}",
            f"Surge:  {engine.surge_charge}",
            f"Pending:{engine.get_pending_garbage_summary()['total_lines']}",
            f"Latency:{player.last_latency * 1000.0:.1f} ms",
        ]
        for line in stats:
            cursor_y += _draw_text(self.screen, self.font_small, line, COL_TEXT, x + 10, cursor_y) + 2

        pending = engine.get_pending_garbage_summary()
        if pending["total_lines"] > 0:
            cursor_y += 4
            cursor_y += _draw_text(self.screen, self.font_small, f"Garbage t={pending['min_timer']}..{pending['max_timer']}", COL_PENDING, x + 10, cursor_y) + 2

        cursor_y += 8
        cursor_y += _draw_text(self.screen, self.font_small, "Hold", COL_TEXT, x + 10, cursor_y)
        hold_kind = None if engine.hold is None else PIECE_ID_TO_KIND.get(int(engine.hold), engine.hold)
        if hold_kind:
            _draw_mini_piece(self.screen, str(hold_kind), x + 42, cursor_y + 18)
        cursor_y += 44
        cursor_y += _draw_text(self.screen, self.font_small, "Next", COL_TEXT, x + 10, cursor_y)
        bag = list(engine.bag[:5]) if engine.bag is not None else []
        for idx, raw_pid in enumerate(bag):
            kind = PIECE_ID_TO_KIND.get(int(raw_pid))
            if kind is None:
                continue
            _draw_mini_piece(self.screen, kind, x + 42, cursor_y + 18 + idx * 28, cell=12)

        info_y = y + BOARD_PX_H - 80
        if player.last_choice is not None:
            if "visits" in player.last_choice:
                visits = player.last_choice["visits"]
                max_visits = int(visits.max()) if len(visits) else 0
                _draw_text(self.screen, self.font_small, f"MCTS sims: {int(visits.sum())}", COL_DIM, x + 10, info_y)
                _draw_text(self.screen, self.font_small, f"Top visits: {max_visits}", COL_DIM, x + 10, info_y + 18)
            elif "score" in player.last_choice:
                _draw_text(self.screen, self.font_small, f"Beam score: {player.last_choice['score']:.2f}", COL_DIM, x + 10, info_y)

    def _draw_top_bar(self) -> None:
        title = f"TetrisZero PvP Viewer  |  Round {self.turn_index}/{self.cfg.max_plies}"
        _draw_text(self.screen, self.font_big, title, COL_TEXT, MARGIN, 10)
        status = "RUNNING" if self.auto_run else "PAUSED"
        status_color = COL_GO if self.auto_run else COL_PAUSE
        _draw_text(self.screen, self.font, status, status_color, WINDOW_W - 220, 14)
        _draw_text(self.screen, self.font_small, f"Delay {self.delay_seconds:.2f}s", COL_DIM, WINDOW_W - 140, 16)
        if self.winner is not None:
            _draw_text(self.screen, self.font, f"Winner: {self.winner.upper()} ({self.termination})", COL_ACCENT, WINDOW_W // 2 - 120, 14)
        elif self.termination == "max_plies":
            _draw_text(self.screen, self.font, "Reached max plies", COL_ACCENT, WINDOW_W // 2 - 70, 14)

    def _draw_controls(self) -> None:
        lines = [
            "Space  Pause/Resume",
            "N      Step one turn",
            "R      Reset match",
            "[ / ]  Slower/Faster",
            "Esc    Quit",
        ]
        base_x = WINDOW_W // 2 - 70
        base_y = WINDOW_H - MARGIN - 16 * len(lines)
        for idx, line in enumerate(lines):
            _draw_text(self.screen, self.font_small, line, COL_DIM, base_x, base_y + idx * 16)

    def _draw(self) -> None:
        self.screen.fill(COL_BG)
        self._draw_top_bar()
        left_panel_x = MARGIN
        left_board_x = left_panel_x + PANEL_W + GAP
        right_board_x = left_board_x + BOARD_PX_W + GAP + GAP
        right_panel_x = right_board_x + BOARD_PX_W + GAP
        board_y = TOP_BAR_H + MARGIN
        self._draw_player_panel(self.player_a, self.engine_a, left_panel_x, board_y, self.current_side == "a" and self.winner is None)
        self._draw_board(self.engine_a, left_board_x, board_y)
        self._draw_board(self.engine_b, right_board_x, board_y)
        self._draw_player_panel(self.player_b, self.engine_b, right_panel_x, board_y, self.current_side == "b" and self.winner is None)
        self._draw_controls()
        pygame.display.flip()

    def _handle_input(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.KEYDOWN:
            return True
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_SPACE:
            self.auto_run = not self.auto_run
        elif event.key == pygame.K_n:
            self._advance_turn()
        elif event.key == pygame.K_r:
            self._reset_match()
        elif event.key == pygame.K_LEFTBRACKET:
            self.delay_seconds = min(5.0, self.delay_seconds + 0.05)
        elif event.key == pygame.K_RIGHTBRACKET:
            self.delay_seconds = max(0.0, self.delay_seconds - 0.05)
        return True

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                running = self._handle_input(event)
                if not running:
                    break
            if not running:
                break
            if self.auto_run and self.winner is None and self.termination != "max_plies":
                self.turn_timer += dt
                if self.turn_timer >= self.delay_seconds:
                    self.turn_timer = 0.0
                    self._advance_turn()
            self._draw()
        pygame.quit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch two TetrisZero agents play a PvP match.")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint to use for both players by default.")
    parser.add_argument("--model-a", type=str, default=None, help="Checkpoint for player A.")
    parser.add_argument("--model-b", type=str, default=None, help="Checkpoint for player B.")
    parser.add_argument("--device", type=str, default=default_device(), help="Torch device for checkpoints.")
    parser.add_argument("--seed", type=int, default=0, help="Match seed.")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between turns.")
    parser.add_argument("--max-plies", type=int, default=400, help="Maximum PvP rounds.")
    parser.add_argument("--simulations", type=int, default=32, help="MCTS simulations per turn.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Visit sampling temperature.")
    parser.add_argument("--max-depth", type=int, default=3, help="MCTS depth limit.")
    parser.add_argument("--visible-queue-depth", type=int, default=5, help="Visible queue depth for search.")
    parser.add_argument("--c-puct", type=float, default=1.5, help="MCTS exploration constant.")
    parser.add_argument("--beam-depth", type=int, default=2, help="Beam fallback depth.")
    parser.add_argument("--beam-width", type=int, default=64, help="Beam fallback width.")
    parser.add_argument("--no-hold", action="store_true", help="Disable hold in search.")
    parser.add_argument("--disable-mcts", action="store_true", help="Use beam search instead of MCTS.")
    args = parser.parse_args(argv)

    model_a, model_b = resolve_model_paths(args.model, args.model_a, args.model_b)
    player_a = build_search_player(
        "Player A",
        model_a,
        device=str(args.device),
        simulations=int(args.simulations),
        temperature=float(args.temperature),
        max_depth=int(args.max_depth),
        visible_queue_depth=int(args.visible_queue_depth),
        c_puct=float(args.c_puct),
        include_hold=not bool(args.no_hold),
        beam_depth=int(args.beam_depth),
        beam_width=int(args.beam_width),
        disable_mcts=bool(args.disable_mcts),
    )
    player_b = build_search_player(
        "Player B",
        model_b,
        device=str(args.device),
        simulations=int(args.simulations),
        temperature=float(args.temperature),
        max_depth=int(args.max_depth),
        visible_queue_depth=int(args.visible_queue_depth),
        c_puct=float(args.c_puct),
        include_hold=not bool(args.no_hold),
        beam_depth=int(args.beam_depth),
        beam_width=int(args.beam_width),
        disable_mcts=bool(args.disable_mcts),
    )
    cfg = PvpGameConfig(max_plies=int(args.max_plies), include_hold=not bool(args.no_hold))
    ui = TetrisZeroPvpUI(player_a, player_b, seed=int(args.seed), delay_seconds=float(args.delay), cfg=cfg)
    ui.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
