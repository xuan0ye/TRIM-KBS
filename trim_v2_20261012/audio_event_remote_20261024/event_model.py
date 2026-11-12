"""Equal-capacity protected adapters. No labels enter the forward computation."""
import torch
from torch import nn


class EventCorrection(nn.Module):
    def __init__(self, context_dim=128, event_dim=128):
        super().__init__()
        self.ta = self.route(2*context_dim+event_dim+2)
        self.tav = self.route(3*context_dim+event_dim+2)
        for route in (self.ta, self.tav):
            nn.init.zeros_(route["delta"][-1].weight)
            nn.init.zeros_(route["delta"][-1].bias)

    @staticmethod
    def route(width):
        return nn.ModuleDict(dict(
            delta=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,2)),
            gate=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,1))))

    def forward(self, base_logits, text, audio, visual, event, audio_present,
                visual_present, event_present=None):
        if event_present is None:
            event_present = torch.ones_like(audio_present)
        available = (audio_present.bool() & event_present.bool()).float()[:,None]
        ta_x = torch.cat((text,audio,event,base_logits[:,1]),dim=-1)
        tav_x = torch.cat((text,audio,visual*visual_present[:,None],event,base_logits[:,3]),dim=-1)
        ta = self.ta["gate"](ta_x).sigmoid() * 2*torch.tanh(self.ta["delta"](ta_x)/2)
        tav = self.tav["gate"](tav_x).sigmoid() * 2*torch.tanh(self.tav["delta"](tav_x)/2)
        full_delta = torch.where(visual_present[:,None].bool(), tav, ta)
        output = torch.stack((base_logits[:,0],base_logits[:,1]+available*ta,
                              base_logits[:,2],base_logits[:,3]+available*full_delta),dim=1)
        return output
