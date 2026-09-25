"""Forward noising process and DDIM sampler.

Two schedule families are implemented. The tangent schedule is the one
used throughout; the cosine schedule is retained because replacing one
with the other is the single largest measured effect in this work, and
the mechanism is visible in how the two allocate sampling steps.

Step spacing is a separate choice from the schedule itself. Spacing
uniformly in signal level rather than in timestep index is what puts a
useful number of steps in the low-noise regime where residue identity
is resolved.
"""

import math

import torch


class LatentDiffusion:
    def __init__(
        self,
        n_steps=1000,
        schedule="tan",
        steepness=10.0,
        spacing="signal",
        device=None,
    ):
        self.n_steps = n_steps
        self.schedule = schedule
        self.spacing = spacing
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        t = torch.linspace(0, 1, n_steps + 1, device=self.device)

        if schedule == "tan":
            t_safe = t.clamp(0, 0.9999)
            ac = 1.0 / (1.0 + (steepness ** 2) * torch.tan(math.pi * t_safe / 2) ** 2)
        elif schedule == "cosine":
            s = 0.008
            ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        else:
            raise ValueError(f"unknown schedule '{schedule}'")

        ac = (ac / ac[0]).clamp(min=0.0, max=1.0)
        self.alphas_cumprod = ac
        self.sqrt_alphas = ac.sqrt()
        self.sqrt_one_minus = (1.0 - ac).clamp(min=0).sqrt()

    def _b(self, coef):
        return coef.view(-1, 1, 1)

    def q_sample(self, x0, t_idx):
        noise = torch.randn_like(x0)
        a = self._b(self.sqrt_alphas[t_idx])
        s = self._b(self.sqrt_one_minus[t_idx])
        return a * x0 + s * noise, noise

    def eps_from_x0(self, x_t, x0_hat, t_idx):
        a = self._b(self.sqrt_alphas[t_idx])
        s = self._b(self.sqrt_one_minus[t_idx])
        return (x_t - a * x0_hat) / s.clamp(min=1e-8)

    def make_timesteps(self, n_sample_steps):
        """Descending list of timestep indices for the reverse pass."""
        if self.spacing == "index":
            step = max(1, self.n_steps // n_sample_steps)
            return list(range(self.n_steps - 1, -1, -step))

        sa = self.sqrt_alphas[: self.n_steps]
        sa_asc = torch.flip(sa, dims=[0]).contiguous()
        targets = torch.linspace(
            float(sa[-1]), float(sa[0]), n_sample_steps, device=self.device
        )
        idx_asc = torch.searchsorted(sa_asc, targets).clamp(0, self.n_steps - 1)
        idx = torch.unique(self.n_steps - 1 - idx_asc)
        return torch.flip(idx, dims=[0]).tolist()

    def step_allocation(self, n_sample_steps, t_cut=0.1):
        """How many sampling steps fall below t_cut.

        This is the quantity that separates the two schedule families:
        the tangent schedule places a substantial fraction of its steps
        in the low-noise regime, the cosine schedule almost none.
        """
        ts = self.make_timesteps(n_sample_steps)
        below = sum(1 for t in ts if t < t_cut * self.n_steps)
        return {"total": len(ts), "below_t_cut": below, "t_cut": t_cut}

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        mask,
        n_sample_steps=250,
        temperature=1.0,
        clamp_val=3.0,
        use_self_cond=True,
    ):
        """Deterministic DDIM reverse pass. The mask carries the per-sample
        target length, so nothing is generated at padded positions."""
        model.eval()
        B = shape[0]
        mask_f = mask.unsqueeze(-1).float()

        x = torch.randn(shape, device=self.device) * temperature * mask_f
        ts = self.make_timesteps(n_sample_steps)

        x_self_cond = None
        x0_hat = None

        for i, t_val in enumerate(ts):
            t_idx = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            t_cont = t_idx.float() / self.n_steps

            x0_hat = model(x, t_cont, mask, x_self_cond if use_self_cond else None)
            x0_hat = x0_hat.clamp(-clamp_val, clamp_val) * mask_f
            x_self_cond = x0_hat

            t_prev = ts[i + 1] if i + 1 < len(ts) else 0
            eps_hat = self.eps_from_x0(x, x0_hat, t_idx)
            x = (
                self.sqrt_alphas[t_prev] * x0_hat
                + self.sqrt_one_minus[t_prev] * eps_hat
            ) * mask_f

        return x0_hat * mask_f
