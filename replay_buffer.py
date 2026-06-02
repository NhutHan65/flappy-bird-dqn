import numpy as np


class PrioritizedReplayBuffer:
    """
    Replay buffer with two key improvements over the tutorial's uniform buffer:

    1. Protected early-experience segment
       The first `mem_retain` fraction of slots is never overwritten.
       This preserves diverse early experiences and prevents catastrophic
       forgetting of behaviours learned in the initial random phase.

    2. Prioritized Experience Replay (PER)
       Each transition carries a priority p_i = |TD error| + eps.
       Sampling probability: P(i) = p_i^alpha / sum(p_j^alpha).
       Importance-sampling weights w_i correct for the non-uniform
       distribution so the update remains unbiased in expectation.

    States are stored as uint8 to keep memory usage ~700 MB for 50 k frames.
    """

    def __init__(
        self,
        mem_size: int,
        mem_retain: float,
        state_shape: tuple,
        replay_start_size: int,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ):
        self.mem_size          = mem_size
        self.protected_size    = int(mem_retain * mem_size)
        self.writable_size     = mem_size - self.protected_size
        self.replay_start_size = replay_start_size
        self.alpha             = alpha
        self.eps               = eps

        self.states      = np.zeros((mem_size, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((mem_size, *state_shape), dtype=np.uint8)
        self.actions     = np.zeros(mem_size, dtype=np.int64)
        self.rewards     = np.zeros(mem_size, dtype=np.float32)
        self.dones       = np.zeros(mem_size, dtype=np.float32)
        self.priorities  = np.zeros(mem_size, dtype=np.float32)

        self.mem_count     = 0
        self._max_priority = 1.0

    # ── Write ─────────────────────────────────────────────────────────────────

    def _write_index(self) -> int:
        if self.mem_count < self.protected_size:
            return self.mem_count
        writable_pos = (self.mem_count - self.protected_size) % self.writable_size
        return self.protected_size + writable_pos

    def add(self, state, action: int, reward: float, next_state, done: float):
        idx = self._write_index()
        self.states[idx]      = state
        self.next_states[idx] = next_state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.dones[idx]       = done
        # New experiences get max priority so they are visited at least once
        self.priorities[idx]  = self._max_priority
        self.mem_count       += 1

    # ── Sample ────────────────────────────────────────────────────────────────

    def sample(self, batch_size: int, beta: float = 0.4):
        n_valid = min(self.mem_count, self.mem_size)
        raw_p   = self.priorities[:n_valid] ** self.alpha
        probs   = raw_p / raw_p.sum()

        indices = np.random.choice(n_valid, batch_size, p=probs, replace=False)

        # Importance-sampling weights (correct for non-uniform sampling)
        weights = (n_valid * probs[indices]) ** (-beta)
        weights /= weights.max()   # normalise so max weight = 1

        return (
            self.states[indices].astype(np.float32),
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices].astype(np.float32),
            self.dones[indices],
            indices,
            weights.astype(np.float32),
        )

    # ── Priority update ───────────────────────────────────────────────────────

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        new_p = np.abs(td_errors) + self.eps
        self.priorities[indices] = new_p
        self._max_priority = max(self._max_priority, float(new_p.max()))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self.mem_count >= self.replay_start_size

    def __len__(self) -> int:
        return min(self.mem_count, self.mem_size)
