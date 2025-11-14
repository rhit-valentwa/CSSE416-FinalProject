"""
Gymnasium environment for Super Mario Bros using pygame.
Provides frame stacking, reward shaping, and action space conversion.
"""

import os
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
    set(RIGHT_KEYS),  # 0: Right
    set(JUMP_KEYS),   # 1: Jump
    set(LEFT_KEYS),   # 2: Left
]

# RGB to grayscale conversion weights (standard luminance formula)
GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)


class MarioLevelEnv(gym.Env):
    """Gymnasium environment for Mario with frame stacking and custom reward shaping."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        width: int = 800,
        height: int = 600,
        max_steps: int = 20000,
        frame_skip: int = 4,
        number_of_sequential_frames: int = 16,
        reward_cfg: dict | None = None,
    ):
        self.render_mode = render_mode
        self.width, self.height = int(width), int(height)
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.number_of_sequential_frames = number_of_sequential_frames

        self.action_space = spaces.MultiBinary(len(COMBO_ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.number_of_sequential_frames, self.height, self.width), 
            dtype=np.uint8
        )

        # Default reward weights
        self.rw = {
            "dx_scale": 0.1,
            "score_scale": 0.001,
            "death_penalty": -150,
            "win_bonus": 500.0,
            "jump_tap_cost": 0,
            "jump_hold_cost": 0,
            "time_penalty": -0.05,
        }
        if reward_cfg:
            self.rw.update(reward_cfg)

        # Use dummy video driver if not rendering to screen
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
        self.held_action = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.ticks_ms = 0
        self.held_action = None

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
        
        # Fill frame buffer with initial frames
        for _ in range(self.number_of_sequential_frames):
            self.frame_buf.append(self._frame())
        
        info = self._info(False, False)
        if self.render_mode == "human":
            self.render()
        obs = self._get_stacked_frames()
        return obs, info

    def _restart_level(self):
        """Restart level after death while keeping score/lives."""
        setup.SCREEN = self.surface
        self.level = Level1()
        self.level.startup(current_time=self.ticks_ms, persist=self.persist)
        self.persist[c.MARIO_DEAD] = False
        if hasattr(self.level.mario, "dead"):
            self.level.mario.dead = False

    def step(self, action: int):
        total_reward = self.rw["time_penalty"]
        terminated = False
        
        action_tuple = tuple(action)
        self.held_action = action_tuple
        if self.held_action is None:
            self.held_action = action_tuple
        
        # Convert action bits to pressed keys
        pressed = set()
        for i, v in enumerate(self.held_action):
            if v:
                pressed.update(COMBO_ACTIONS[i])
        
        self.step_count += 1
        r = -0.01
        
        # Execute action over multiple frames
        for i in range(self.frame_skip):
            self.ticks_ms += int(1000 / self.metadata["render_fps"])
            self.level.update(self.surface, _KeysProxy(pressed), self.ticks_ms)
            
            mario_dead = self.persist.get(c.MARIO_DEAD, False) or getattr(self.level.mario, "dead", False)
            level_done = bool(getattr(self.level, "done", False))
            st = getattr(self.level.mario, "state", None)
            
            # Check win condition
            if st == c.FLAGPOLE:
                r += self.rw["win_bonus"]
                print("Mario won!")
                terminated = True
                break
            
            # Handle level end states
            if level_done:
                nxt = getattr(self.level, "next", None)
                if nxt == c.LOAD_SCREEN and self.persist.get(c.LIVES, 0) > 0:
                    self._restart_level()
                    self.persist[c.MARIO_DEAD] = False
                    if hasattr(self.level.mario, "dead"):
                        self.level.mario.dead = False
                    self.prev_x = self.level.mario.rect.x
                    self.prev_score = self.persist[c.SCORE]
                    break
                elif nxt == c.TIME_OUT:
                    terminated = True
                    print("Time out!")
                    break
                elif nxt == c.GAME_OVER:
                    r += self.rw["death_penalty"]
                    terminated = True
                    print("Mario died!")
                    break
                else:
                    terminated = True
                    print("Level ended! (else)")
                    break
        
        # Reward for progress and score
        x = self.level.mario.rect.x
        dx = x - self.prev_x
        score = self.persist[c.SCORE]
        dscore = score - self.prev_score
        r += self.rw["dx_scale"] * dx
        r += self.rw["score_scale"] * dscore
        
        total_reward += r
        self.prev_x = x
        self.prev_score = score
        
        truncated = self.step_count >= self.max_steps
        info = self._info(terminated, truncated)
        self.frame_buf.append(self._frame())
        
        if self.render_mode == "human":
            self.render()
        
        obs = self._get_stacked_frames()
        return obs, float(r), terminated, truncated, info
    
    def _get_stacked_frames(self):
        """
        Stack frames with temporal diversity:
        - Always include 4 most recent frames
        - Randomly sample remaining frames from last 64 to reduce temporal correlation
        """
        num_frames = self.number_of_sequential_frames
        buf = list(self.frame_buf)
        n_buf = len(buf)
        
        most_recent = buf[-4:] if n_buf >= 4 else buf[:]
        sample_pool = buf[max(0, n_buf-64):max(0, n_buf-4)] if n_buf > 4 else []
        n_sample = max(0, num_frames - len(most_recent))
        
        if sample_pool and n_sample > 0:
            idxs = np.random.choice(len(sample_pool), size=n_sample, replace=True)
            sampled = [sample_pool[i] for i in idxs]
        else:
            sampled = [buf[0]] * n_sample if buf else []
        
        frames = most_recent + sampled
        return np.stack(frames, axis=0)

    def render(self):
        if self.render_mode == "human":
            frame = self._frame()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pass
            pg.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return self._frame()

    def render_fullframe(self):
        """Return full RGB frame without preprocessing."""
        rgb = np.transpose(pg.surfarray.array3d(self.surface), (1, 0, 2))
        return rgb

    def close(self):
        try:
            pg.display.quit()
            pg.quit()
        except Exception:
            pass

    def _map_action(self, a: int) -> '_KeysProxy':
        pressed = set(COMBO_ACTIONS[a])
        return _KeysProxy(pressed)

    def _frame(self) -> np.ndarray:
        """Process frame: resize to 60x80, convert to grayscale, normalize."""
        rgb = np.transpose(pg.surfarray.array3d(self.surface), (1, 0, 2))
        t = torch.from_numpy(rgb).to(torch.float32).div_(255.0).permute(2, 0, 1).unsqueeze(0)
        t_small = F.interpolate(t, size=(60, 80), mode='bilinear', align_corners=False)
        w = GRAY_WEIGHTS.to(t_small.device)
        gray = (t_small * w).sum(dim=1, keepdim=True)
        return gray.squeeze(0).squeeze(0).cpu().numpy()

    def _info(self, terminated: bool, truncated: bool) -> dict:
        return {
            "score": int(self.persist.get(c.SCORE, 0)),
            "coins": int(self.persist.get(c.COIN_TOTAL, 0)),
            "x": int(self.level.mario.rect.x) if self.level else 0,
            "y": int(self.level.mario.rect.y) if self.level else 0,
            "terminated": terminated,
            "truncated": truncated,
        }


class _KeysProxy:
    """Mimics pygame.key.get_pressed() for action injection."""
    
    def __init__(self, pressed=None):
        self._pressed = set(pressed or [])
    
    def __getitem__(self, keycode: int) -> int:
        return 1 if keycode in self._pressed else 0