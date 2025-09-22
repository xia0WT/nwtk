import torch
from .config import th_int

CUTOFF_RADIUS=2.5
"""
convert a molecular to a graph
"""
class GetMolEdge(object):
    def __init__(self, 
                 structure
                ):
        
        self.structure=structure
        self.edge_info=torch.tensor((), dtype=th_int)

    #def __call__(self):
        #self.edge_info()
        
    def get_edge_connectivity(self):
        
        for i, site in enumerate(self.structure):
            
            neibor_index=torch.tensor([nn.index for nn in self.structure.get_neighbors(self.structure[i], CUTOFF_RADIUS)], dtype=th_int)
            self_index=torch.tensor((), dtype=th_int).new_full(neibor_index.shape, i)
            edge_info_part=torch.stack((neibor_index, self_index), 0)
            
            self.edge_info=torch.cat((edge_info_part, self.edge_info), 1)
            
        return self.edge_info
