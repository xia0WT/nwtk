import torch
#import numpy
from torch.nn import functional
from .config import th_float

class GetMolNode(object):
    def __init__(self, 
                 structure):
        
        self.structure=structure
        self.coor = torch.tensor(structure.cart_coords, dtype=th_float)
        self.relative_vector = self.coor[:,None]-self.coor 
        #self.edge_info=torch.tensor((), dtype=th_int)
    
    def acos_safe(self, x, eps=1e-6):

        return torch.where(abs(x) <= 1-eps,
                        torch.acos(x),
                        torch.acos(x - torch.sign(x)*eps))
        
    def get_node_distance(self): # return distance matrix
        
        dist_matrix=torch.norm(self.relative_vector , dim=2 ,p=2)
        return dist_matrix
        
    def get_node_angle(self):  #the first dimension is the center atom
        
        relative_vector_n = functional.normalize(self.relative_vector ,dim =2)
        relative_vector_T =relative_vector_n.permute(0,2,1)
        rec = torch.matmul(relative_vector_n ,relative_vector_T )
        return rec#self.acos_safe(rec)


