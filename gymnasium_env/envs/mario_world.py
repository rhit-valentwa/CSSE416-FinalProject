from __future__ import annotations
import os
from enum import Enum
from typing import Set
import numpy as np
import pygame as pg
import gymnasium as gym
from gymnasium import spaces
from mario_game.data.states.level1 import Level1
from mario_game.data import constants as c
from mario_game.data import setup

JUMP_KEYS = (pg.K_SPACE, pg.K_z, pg.K_a, pg.K_UP, pg.K_w)
RIGHT_KEYS = (pg.K_RIGHT,)
LEFT_KEYS  = (pg.K_LEFT,)
DUCK_KEYS  = (pg.K_DOWN, pg.K_s)

class Actions(Enum):
    NO_ACTION = 0
    RIGHT = 1
    JUMP = 3
    LEFT = 4
    DUCK = 5

COMBO_ACTIONS = [
    set(),                                        # 0: NO_ACTION
    set(RIGHT_KEYS),                              # 1: Right
    set(RIGHT_KEYS) | set(JUMP_KEYS),             # 2: Right+Jump
    set(JUMP_KEYS),                               # 3: Jump
    set(LEFT_KEYS),                               # 4: Left
]


class _KeysProxy: # Mimics pygame.key.get_pressed()
    def __init__(self, pressed=None):
        self._pressed = set(pressed or [])
    def __getitem__(self, keycode: int) -> int:
        return 1 if keycode in self._pressed else 0


class MarioLevelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        width: int = 800,
        height: int = 600,
        max_steps: int = 50_000,
        frame_skip: int = 4,              # repeat each action this many frames
        reward_cfg: dict | None = None,
    ):
        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode
        self.width, self.height = int(width), int(height)
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)

        self.action_space = spaces.Discrete(len(COMBO_ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

        self.rw = {
            "dx_scale": 0.05,
            "score_scale": 0.01,
            "death_penalty": -50.0,
            "win_bonus": 100.0,
            "jump_tap_cost": -0.9,
            "jump_hold_cost": -0.1,
        }
        if reward_cfg:
            self.rw.update(reward_cfg)

        # Headless safety when not rendering a window
        if self.render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pg.init()
        pg.display.init()
        if self.render_mode == "human":
            self.display = pg.display.set_mode((self.width, self.height))
        else:
            self.display = pg.Surface((self.width, self.height))
        self.surface = self.display
        self.clock = pg.time.Clock()

        self.level: Level1 | None = None
        self.persist = None
        self.prev_x = 0
        self.prev_score = 0
        self.step_count = 0
        self.ticks_ms = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.ticks_ms = 0

        self.persist = {
            c.SCORE: 0,
            c.TOP_SCORE: 0,
            c.COIN_TOTAL: 0,
            c.LIVES: 3,
            c.CAMERA_START_X: 0,
        }

        setup.SCREEN = self.surface

        self.level = Level1()
        self.level.startup(current_time=self.ticks_ms, persist=self.persist)

        self.prev_x = self.level.mario.rect.x
        self.prev_score = self.persist[c.SCORE]

        obs = self._frame()
        info = self._info(False, False)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):

        total_reward = 0.0
        terminated = False

        pressed = set(COMBO_ACTIONS[int(action)])
        is_jump_pressed = any(k in pressed for k in JUMP_KEYS)

        for i in range(self.frame_skip):
            self.step_count += 1
            self.ticks_ms += int(1000 / self.metadata["render_fps"])
            self.level.update(self.surface, _KeysProxy(pressed), self.ticks_ms)
            x = self.level.mario.rect.x
            dx = x - self.prev_x
            score = self.persist[c.SCORE]
            dscore = score - self.prev_score

            r = 0.0
            r += self.rw["dx_scale"] * dx
            r += self.rw["score_scale"] * dscore
            if dx < 0:
                r -= 0.02 * abs(dx)

            if is_jump_pressed:
                if i == 0:
                    r += self.rw["jump_tap_cost"]
                else:
                    r += self.rw["jump_hold_cost"]

            mario_dead = self.persist.get(c.MARIO_DEAD, False) or getattr(self.level.mario, "dead", False)
            level_done = bool(getattr(self.level, "done", False))
            if mario_dead:
                r += self.rw["death_penalty"]
                terminated = True
            elif level_done:
                r += self.rw["win_bonus"]
                terminated = True

            total_reward += r
            self.prev_x = x
            self.prev_score = score

            if terminated or self.step_count >= self.max_steps:
                break

        truncated = self.step_count >= self.max_steps
        obs = self._frame()
        info = self._info(terminated, truncated)

        if self.render_mode == "human":
            self.render()

        return obs, float(total_reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pass
            pg.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return self._frame()

    def close(self):
        try:
            pg.display.quit()
            pg.quit()
        except Exception:
            pass

    def _map_action(self, a: int) -> _KeysProxy:
        pressed = set(COMBO_ACTIONS[a])
        return _KeysProxy(pressed)

    def _frame(self) -> np.ndarray:
        rgb = np.transpose(pg.surfarray.array3d(self.surface), (1, 0, 2))
        gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.uint8) # had to look up what a "lumeance conversion" is
        return gray[..., None]

    def _info(self, terminated: bool, truncated: bool) -> dict:
        return {
            "score": int(self.persist.get(c.SCORE, 0)),
            "coins": int(self.persist.get(c.COIN_TOTAL, 0)),
            "x": int(self.level.mario.rect.x) if self.level else 0,
            "y": int(self.level.mario.rect.y) if self.level else 0,
            "terminated": terminated,
            "truncated": truncated,
        }