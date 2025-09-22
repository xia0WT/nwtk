import dgl
import math
import numpy as np

class CreateGraph(nn.Module):
    def __init__(self, partition, graph_dim=10, seed=42):
        super().__init__()
        self.nrr = np.random.RandomState(seed)
        self.graph_dim = graph_dim
        kind = len(atom_list)
        a_=np.array([],dtype=np.int32)
        for idx in range(len(atom_list)):
            a_ = np.r_[a_, np.full(( int(partition[idx]*graph_dim **2 ), ), list(atom_list.values())[idx])]
        self.nrr.shuffle(a_)
        self.a_ = a_

    def forward(self):
        graph_dim = self.graph_dim
        scale = lambda x : [x-graph_dim+1, x+graph_dim, x+1] \
                    if (x+1) %graph_dim == 0 and x < graph_dim**2 - graph_dim \
                    else ( [x+1, x-graph_dim**2+graph_dim, x-graph_dim**2+graph_dim+1] \
                          if x >= graph_dim**2 - graph_dim and x != graph_dim**2 - 1
                          else ( [graph_dim**2-graph_dim, graph_dim-1, 0] \
                                if x == graph_dim**2 - 1
                                else [x+1, x+graph_dim, x+graph_dim+1]))

        u = np.linspace(0, graph_dim **2, graph_dim **2, endpoint=False, dtype=np.int32)
        v = np.array(list(map(scale, u)), dtype=np.int32).flatten()
        
        g = dgl.graph((u.repeat(3), v), idtype=th_int)
        bg = dgl.to_bidirected(g)
        self.nodedata(bg)
        
        return bg

    def nodedata(self, graph):
        graph.ndata["node_type"] = torch.tensor(self.a_, dtype = th_int)
        graph.ndata["node_neighbour"] = torch.tensor([[ self.a_[idx]  for idx in graph.predecessors(i)] \
                                                    for i in range(self.graph_dim**2)], dtype = th_int)
    def edgedata(self, graph):
        pass