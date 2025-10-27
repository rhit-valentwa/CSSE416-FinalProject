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
import torch
import torch.nn.functional as F
from collections import deque

JUMP_KEYS = (pg.K_a,)
RIGHT_KEYS = (pg.K_RIGHT,)
LEFT_KEYS  = (pg.K_LEFT,)
DUCK_KEYS  = (pg.K_DOWN, pg.K_s)

COMBO_ACTIONS = [
    set(RIGHT_KEYS),                              # 0: Right
    set(JUMP_KEYS),                             # 1: Jump
    set(LEFT_KEYS),                               # 2: Left
]
GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)



class _KeysProxy: # Mimics pygame.key.get_pressed()
    def __init__(self, pressed=None):
        self._pressed = set(pressed or [])
    def __getitem__(self, keycode: int) -> int:
        return 1 if keycode in self._pressed else 0


class MarioLevelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        width: int = 800,
        height: int = 600,
        max_steps: int = 20000,
        frame_skip: int = 4,
        number_of_sequential_frames: int = 4,
        reward_cfg: dict | None = None,
    ):
        self.render_mode = render_mode
        self.width, self.height = int(width), int(height)
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.number_of_sequential_frames = number_of_sequential_frames

        self.action_space = spaces.MultiBinary(len(COMBO_ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.number_of_sequential_frames, self.height, self.width), dtype=np.uint8)

        self.rw = {
            "dx_scale": 0.05,
            "score_scale": 0.0005,
            "death_penalty": 0,
            "win_bonus": 2.0,
            "jump_tap_cost": 0,
            "jump_hold_cost": 0,
        }
        if reward_cfg:
            self.rw.update(reward_cfg)

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
        self.frame_buf = deque(maxlen=self.number_of_sequential_frames)
        
        # Sticky keys: track currently held action
        self.held_action = None  # Action that's currently being held

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.ticks_ms = 0
        self.held_action = None  # Reset held action

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

        for _ in range(self.number_of_sequential_frames):
            self.frame_buf.append(self._frame())
        
        info = self._info(False, False)
        if self.render_mode == "human":
            self.render()
        return np.stack(self.frame_buf, axis=0), info

    def step(self, action: int):
        total_reward = -0.01
        terminated = False

        action_tuple = tuple(action)
        
        # Update held action if new action is provided
        # Action [0,0,0] means "keep holding current action"
        # if any(action):  # If any key is pressed
        self.held_action = action_tuple
        # If action is [0,0,0] and we have a held action, keep it
        # Otherwise use the current action
        
        if self.held_action is None:
            self.held_action = action_tuple
        
        # Use the held action for this step
        pressed = set()
        for i, v in enumerate(self.held_action):
            if v:
                pressed.update(COMBO_ACTIONS[i])
        self.step_count += 1
        # Execute the held action for frame_skip frames
        r = -0.01
        for i in range(self.frame_skip):
            self.ticks_ms += int(1000 / self.metadata["render_fps"])
            self.level.update(self.surface, _KeysProxy(pressed), self.ticks_ms)
            mario_dead = self.persist.get(c.MARIO_DEAD, False) or getattr(self.level.mario, "dead", False)
            level_done = bool(getattr(self.level, "done", False))
            if mario_dead:
                r += self.rw["death_penalty"]
                terminated = True
            elif level_done:
                r += self.rw["win_bonus"]
                terminated = True
            if terminated or self.step_count >= self.max_steps:
                break

        x = self.level.mario.rect.x
        dx = x - self.prev_x
        score = self.persist[c.SCORE]
        dscore = score - self.prev_score
        r += self.rw["dx_scale"] * dx
        r += self.rw["score_scale"] * dscore
        # print(f"dx: {dx}, dscore: {dscore}, reward: {r}")

        if r > 5 or r < -5:
            r = 0

        total_reward += r
        self.prev_x = x
        self.prev_score = score


        truncated = self.step_count >= self.max_steps
        info = self._info(terminated, truncated)

        self.frame_buf.append(self._frame())

        if self.render_mode == "human":
            self.render()

        return np.stack(self.frame_buf, axis=0), float(r), terminated, truncated, info

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
        # (H, W, 3) uint8 from pygame
        rgb = np.transpose(pg.surfarray.array3d(self.surface), (1, 0, 2))

        # -> float32 [0,1], CHW, add batch
        t = torch.from_numpy(rgb).to(torch.float32).div_(255.0).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        # resize to (60, 80)
        t_small = F.interpolate(t, size=(60, 80), mode='bilinear', align_corners=False)        # (1,3,60,80)

        # grayscale in [0,1]
        w = GRAY_WEIGHTS.to(t_small.device)
        gray = (t_small * w).sum(dim=1, keepdim=True)                                          # (1,1,60,80)

        # return (60,80) float32, normalized
        return gray.squeeze(0).squeeze(0).cpu().numpy()

            # rgb = np.transpose(pg.surfarray.array3d(self.surface), (1, 0, 2))
            # gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.uint8)
            # return gray

    def _info(self, terminated: bool, truncated: bool) -> dict:
        return {
            "score": int(self.persist.get(c.SCORE, 0)),
            "coins": int(self.persist.get(c.COIN_TOTAL, 0)),
            "x": int(self.level.mario.rect.x) if self.level else 0,
            "y": int(self.level.mario.rect.y) if self.level else 0,
            "terminated": terminated,
            "truncated": truncated,
        }