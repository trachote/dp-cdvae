from models.gnn.schnet import SchNet
from torch import nn

class SchNetEncoder(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.schnet = SchNet(cfg)

    def forward(self, batch, embed_node=True, transform=False):
        h = self.schnet(batch, embed_node, transform)
        return h

