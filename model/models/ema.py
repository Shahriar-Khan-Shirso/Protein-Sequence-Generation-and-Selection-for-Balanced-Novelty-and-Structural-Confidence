"""Exponential moving average of the denoiser weights.

Generation uses the averaged weights. Under cosine annealing with warm
restarts the instantaneous weights at the end of a cycle are not
necessarily the best point on the trajectory.
"""

import copy

import torch


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self):
        for sp, mp in zip(self.shadow.parameters(), self.model.parameters()):
            sp.data.copy_(self.decay * sp.data + (1 - self.decay) * mp.data)
        for sb, mb in zip(self.shadow.buffers(), self.model.buffers()):
            sb.data.copy_(mb.data)

    def get_model(self):
        return self.shadow

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)
