"""
Double DQN agent with Dueling architecture, Prioritized Experience Replay,
soft target-network updates, and data augmentation.

What this adds beyond the tutorial (vanilla DQN + MLP + uniform replay):
  1. CNN input  — learns from raw pixels instead of hand-crafted [x, y] state
  2. Double DQN — policy net selects action, target net evaluates it
                  (fixes Q-value overestimation bias)
  3. Dueling    — separate value V(s) and advantage A(s,a) streams
  4. PER        — sample high-TD-error transitions more often
  5. Soft τ-update of target network instead of periodic hard copy
  6. Batch augmentation — random brightness jitter on training samples
"""

from __future__ import annotations

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import config as cfg
from model import DuelingCNN
from replay_buffer import PrioritizedReplayBuffer


class DoubleDQNAgent:

    def __init__(self):
        in_ch = cfg.IN_CHANNELS

        self.policy_net = DuelingCNN(in_ch, cfg.ACTIONS).to(cfg.DEVICE)
        self.target_net = DuelingCNN(in_ch, cfg.ACTIONS).to(cfg.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.LR)

        state_shape = (in_ch, cfg.FRAME_SIZE, cfg.FRAME_SIZE)
        self.memory = PrioritizedReplayBuffer(
            cfg.MEM_SIZE, cfg.MEM_RETAIN, state_shape,
            cfg.REPLAY_START_SIZE,
            alpha=cfg.PER_ALPHA,
            eps=cfg.PER_EPS,
        )

        self.steps   = 0
        self.epsilon = cfg.EPS_START
        self.beta    = cfg.PER_BETA_START

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, state: np.ndarray) -> int:
        # When epsilon > 0 and buffer isn't ready yet: strongly biased random
        # (90 % no-flap) so the bird doesn't die instantly from over-flapping.
        # When epsilon == 0 (test mode) we skip straight to greedy regardless.
        if self.epsilon > 0.0 and not self.memory.ready:
            return int(np.random.choice(2, p=[0.9, 0.1]))

        if np.random.random() < self.epsilon:
            return int(np.random.choice(2, p=[0.9, 0.1]))

        s = torch.FloatTensor(state).unsqueeze(0).to(cfg.DEVICE)
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(s)
        return int(q.argmax(dim=1).item())

    # ── Learning step ─────────────────────────────────────────────────────────

    def learn(self):
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(cfg.BATCH_SIZE, self.beta)

        s  = torch.FloatTensor(states).to(cfg.DEVICE)
        a  = torch.LongTensor(actions).to(cfg.DEVICE)
        r  = torch.FloatTensor(rewards).to(cfg.DEVICE)
        s_ = torch.FloatTensor(next_states).to(cfg.DEVICE)
        d  = torch.FloatTensor(dones).to(cfg.DEVICE)
        w  = torch.FloatTensor(weights).to(cfg.DEVICE)

        # Data augmentation on training batch (random brightness jitter)
        if cfg.AUGMENT:
            brightness = torch.empty(s.shape[0], 1, 1, 1, device=cfg.DEVICE).uniform_(0.85, 1.15)
            s = torch.clamp(s * brightness, 0, 255)

        # Current Q(s, a) from policy network
        self.policy_net.train()
        q_all     = self.policy_net(s)
        q_current = q_all[torch.arange(cfg.BATCH_SIZE), a]

        # Double DQN target:
        #   best_action = argmax_a  Q_policy(s', a)   ← policy net selects
        #   target      = Q_target(s', best_action)   ← target net evaluates
        self.policy_net.eval()
        with torch.no_grad():
            best_actions = self.policy_net(s_).argmax(dim=1)
            q_next       = self.target_net(s_)[torch.arange(cfg.BATCH_SIZE), best_actions]
            q_target     = r + cfg.GAMMA * q_next * (1.0 - d)
        self.policy_net.train()

        td_errors = (q_target - q_current).detach().cpu().numpy()

        # PER-weighted MSE loss
        loss = (w * F.mse_loss(q_current, q_target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), cfg.GRAD_CLIP)
        self.optimizer.step()

        # Update replay priorities with new TD errors
        self.memory.update_priorities(indices, td_errors)

        # Soft-update target network: θ_target = τ·θ + (1-τ)·θ_target
        for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(cfg.TARGET_TAU * p.data + (1.0 - cfg.TARGET_TAU) * tp.data)

        # Linear epsilon decay (starts after buffer is ready)
        self.epsilon = max(
            cfg.EPS_END,
            cfg.EPS_START - (cfg.EPS_START - cfg.EPS_END) * self.steps / cfg.EPS_DECAY,
        )

        # Anneal PER beta from beta_start → 1.0
        self.beta = min(
            cfg.PER_BETA_END,
            cfg.PER_BETA_START
            + (cfg.PER_BETA_END - cfg.PER_BETA_START) * self.steps / cfg.PER_BETA_FRAMES,
        )

        self.steps += 1
        return float(loss), float(td_errors.mean()), float(q_current.max().detach())

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(
            {
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer':  self.optimizer.state_dict(),
                'steps':      self.steps,
                'epsilon':    self.epsilon,
                'beta':       self.beta,
            },
            path,
        )
        print(f"[save] {path}  (steps={self.steps})")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=cfg.DEVICE)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.steps   = ckpt['steps']
        self.epsilon = ckpt['epsilon']
        self.beta    = ckpt['beta']
        print(f"[load] {path}  (steps={self.steps}, ε={self.epsilon:.4f})")

    @staticmethod
    def latest_checkpoint(directory: str) -> str | None:
        ckpts = sorted(glob.glob(os.path.join(directory, '*.pt')))
        return ckpts[-1] if ckpts else None
