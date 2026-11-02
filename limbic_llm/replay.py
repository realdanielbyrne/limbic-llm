"""Offline replay and systems consolidation (complementary learning systems).

Biology
-------
During slow-wave sleep, sharp-wave ripples replay recently encoded hippocampal
sequences (Wilson & McNaughton, 1994). Interleaving these replays with the
neocortex's slow learning lets new knowledge enter cortical weights without
catastrophic interference (McClelland, McNaughton & O'Reilly, 1995).

Computation
-----------
* ``PrioritizedReplayBuffer`` keeps sequences with priority p_i taken from the
  model's novelty / surprise when they were experienced, samples with
  P(i) ∝ p_i^a, and evicts the lowest-priority trace when full.
* ``Consolidator`` provides
  - ``replay_loss``: an interleaved awake-replay term to add to the
    current-task loss, and
  - ``sleep``: an offline phase that updates only the *slow* parameter group
    (e.g. neocortex layers, persistent memory) at a low learning rate.
  Both use CE on the replayed tokens plus a distillation term
  τ²·KL( softmax(z_stored/τ) ‖ softmax(z/τ) ) toward the logits recorded at
  encoding time (a dark-experience-replay style anchor), which holds the
  function, not just the labels, in place.
* ``orthogonal_projection`` is an optional hook to keep sleep updates out of
  the gradient subspace of previously consolidated tasks (a place to plug
  ICA-identified functional networks / O-LoRA-style constraints).
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 10_000, alpha: float = 0.6, store_logits: bool = True,
                 seed: int | None = None) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.store_logits = store_logits
        self.tokens: list[torch.Tensor] = []
        self.logits: list[torch.Tensor | None] = []
        self.priority = torch.empty(0)
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.tokens)

    @torch.no_grad()
    def add(self, tokens: torch.Tensor, priorities: torch.Tensor,
            logits: torch.Tensor | None = None) -> None:
        """tokens: (B,T); priorities: (B,); logits: (B,T,V) predicted for tokens[:,1:]."""
        tokens, priorities = tokens.detach().cpu(), priorities.detach().float().cpu()
        if logits is not None and self.store_logits:
            logits = logits.detach().to("cpu", torch.float16)
        for b in range(tokens.shape[0]):
            p = float(priorities[b]) + 1e-6
            lg = logits[b] if (logits is not None and self.store_logits) else None
            if len(self.tokens) < self.capacity:
                self.tokens.append(tokens[b])
                self.logits.append(lg)
                self.priority = torch.cat([self.priority, torch.tensor([p])])
            else:
                j = int(self.priority.argmin())
                if p > float(self.priority[j]):
                    self.tokens[j], self.logits[j], self.priority[j] = tokens[b], lg, p

    def sample(self, n: int, device=None):
        if not self.tokens:
            raise RuntimeError("replay buffer is empty")
        probs = self.priority.clamp_min(1e-6) ** self.alpha
        idx = torch.multinomial(probs / probs.sum(), n, replacement=True, generator=self.gen)
        toks = torch.stack([self.tokens[i] for i in idx]).to(device)
        if self.store_logits and all(self.logits[i] is not None for i in idx):
            lg = torch.stack([self.logits[i] for i in idx]).to(device).float()
        else:
            lg = None
        return toks, lg, idx

    def update_priorities(self, idx: torch.Tensor, priorities: torch.Tensor) -> None:
        self.priority[idx] = priorities.detach().float().cpu() + 1e-6


def sequence_priority(novelty: torch.Tensor | None, token_loss: torch.Tensor) -> torch.Tensor:
    """Priority per sequence: mean CA1 novelty if available, else mean loss."""
    if novelty is not None:
        return novelty.detach().mean(1)
    return token_loss.detach().mean(1)


class Consolidator:
    def __init__(
        self,
        model: nn.Module,
        buffer: PrioritizedReplayBuffer,
        slow_params: Iterable[nn.Parameter] | None = None,
        lr: float = 1e-4,
        distill_weight: float = 1.0,
        temperature: float = 2.0,
        loss_mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.buffer = buffer
        self.slow_params = list(slow_params) if slow_params is not None else list(model.parameters())
        self.opt = torch.optim.AdamW(self.slow_params, lr=lr, weight_decay=0.0)
        self.distill_weight = distill_weight
        self.tau = temperature
        self.loss_mask_fn = loss_mask_fn
        self.projector: Callable[[], None] | None = None

    def _loss(self, toks: torch.Tensor, stored_logits: torch.Tensor | None) -> torch.Tensor:
        inp, tgt = toks[:, :-1], toks[:, 1:]
        logits = self.model(inp).logits
        mask = self.loss_mask_fn(tgt) if self.loss_mask_fn else torch.ones_like(tgt, dtype=logits.dtype)
        ce = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt)
        loss = (ce * mask).sum() / mask.sum().clamp_min(1)
        if stored_logits is not None and self.distill_weight > 0:
            t = self.tau
            kl = F.kl_div(F.log_softmax(logits / t, -1), F.softmax(stored_logits / t, -1),
                          reduction="none").sum(-1)
            loss = loss + self.distill_weight * t * t * (kl * mask).sum() / mask.sum().clamp_min(1)
        return loss

    def replay_loss(self, batch_size: int, device=None) -> torch.Tensor:
        """Awake interleaved replay: add the returned loss to the task loss."""
        toks, lg, _ = self.buffer.sample(batch_size, device)
        return self._loss(toks, lg)

    def sleep(self, steps: int = 100, batch_size: int = 16, device=None) -> list[float]:
        """Offline consolidation of replayed traces into the slow parameters."""
        self.model.train()
        history = []
        for _ in range(steps):
            toks, lg, _ = self.buffer.sample(batch_size, device)
            loss = self._loss(toks, lg)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            for p in self.model.parameters():          # freeze fast parameters
                if all(p is not q for q in self.slow_params):
                    p.grad = None
            if self.projector is not None:
                self.projector()
            torch.nn.utils.clip_grad_norm_(self.slow_params, 1.0)
            self.opt.step()
            history.append(loss.item())
        return history


def orthogonal_projection(params: list[nn.Parameter], bases: dict[int, torch.Tensor]) -> Callable[[], None]:
    """Build a hook that removes gradient components inside protected subspaces.

    ``bases[id(p)]`` is an orthonormal (n, k) basis for parameter p's
    flattened-gradient subspace to protect (e.g. top singular vectors of
    gradients from a consolidated task, or ICA-identified functional networks).
    """
    def project() -> None:
        for p in params:
            U = bases.get(id(p))
            if U is None or p.grad is None:
                continue
            g = p.grad.reshape(-1)
            p.grad.copy_((g - U @ (U.T @ g)).view_as(p.grad))
    return project
