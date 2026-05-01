import numpy as np
import itertools
import networkx as nx

def D_to_NXgraph(D, G_directed):
    if G_directed == False:
        G = nx.Graph()
    elif G_directed == True:
        G = nx.DiGraph()
    nodes = range(D.shape[1])
    rows, cols = np.where(D > 0)
    edge_pos = zip(rows.tolist(), cols.tolist())
    edges = [(ind[0],ind[1],{'weight': D[ind[0],ind[1]]}) for ind in edge_pos]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def find_ones_pos(lst):
    return [i for i, x in enumerate(lst) if x == 1]


class InciGraph_Statistics:
    def __init__(self, InciMat, row_node_label=None, col_node_label=None):
        self.InciMat = InciMat
        self.row_node_label = row_node_label if row_node_label is not None else {i: i for i in range(len(InciMat))}
        self.col_node_label = col_node_label if col_node_label is not None else {j: j for j in range(len(InciMat[0]))}


class AdjGraph_Statistics(InciGraph_Statistics):
    def __init__(self, edge_weight_option, InciMat=None, AdjMat=None, AdjMatRev=None, row_node_label=None, col_node_label=None):
        if InciMat is None and AdjMat is None and AdjMatRev is None:
            raise ValueError('At least one of InciMat, AdjMat, or AdjMatRev must be provided')
        if InciMat is not None:
            super().__init__(InciMat, row_node_label, col_node_label)
        else:
            self.InciMat = None
            self.row_node_label = row_node_label
            self.col_node_label = col_node_label
        self.edge_weight_option = edge_weight_option
        self.AdjMat = AdjMat
        self.AdjMatRev = AdjMatRev

    def FlattenEdgeWeight_Reverse(self, col_i):
        if self.edge_weight_option == "Weighted":
            return 1/sum(self.InciMat[:,col_i])
        elif self.edge_weight_option == "Uniform":
            return 1
        else:
            raise ValueError('Edge weight is not defined')
 
    def FlattenEdgeWeight(self, row_i):
        if self.edge_weight_option == "Weighted":
            return 1/sum(self.InciMat[row_i,:])
        elif self.edge_weight_option == "Uniform":
            return 1
        else:
            raise ValueError('Edge weight is not defined')
 

    #takes incidence matrix of a hypergraph and flatten it out to a graph in the column direction
    def FlattenHyper(self):
        row_dim, col_dim = self.InciMat.shape[0], self.InciMat.shape[1]
        AdjMat = np.zeros((col_dim,col_dim))
        for r_i in range(row_dim):
            row = self.InciMat[r_i,:]
            ones = find_ones_pos(row)
            combs = list(itertools.combinations(ones,2))
            for comb in combs:
                i,j = comb[0],comb[1]
                AdjMat[i,j] += self.FlattenEdgeWeight(r_i)
                AdjMat[j,i] += self.FlattenEdgeWeight(r_i)
        self.AdjMat = AdjMat
        if self.col_node_label is None:
            self.col_node_label = {j: j for j in range(col_dim)}


    #takes incidence matrix of a hypergraph and flatten it in the row direction
    def FlattenHyper_Reverse(self):
        row_dim, col_dim = self.InciMat.shape[0], self.InciMat.shape[1]
        AdjMat = np.zeros((row_dim,row_dim))
        for c_i in range(col_dim):
            col = self.InciMat[:,c_i]
            ones = find_ones_pos(col)
            combs = list(itertools.combinations(ones,2))
            for comb in combs:
                i,j = comb[0],comb[1]
                AdjMat[i,j] += self.FlattenEdgeWeight_Reverse(c_i)
                AdjMat[j,i] += self.FlattenEdgeWeight_Reverse(c_i)
        self.AdjMatRev = AdjMat
        if self.row_node_label is None:
            self.row_node_label = {i: i for i in range(row_dim)}


class NetworkX_Statistics(AdjGraph_Statistics):
    def __init__(self, edge_weight_option, G_directed, InciMat=None, AdjMat=None, AdjMatRev=None, row_node_label=None, col_node_label=None):
        super().__init__(edge_weight_option, InciMat, AdjMat, AdjMatRev, row_node_label, col_node_label)
        self.G_directed = G_directed
        self.G = D_to_NXgraph(self.AdjMat, G_directed) if self.AdjMat is not None else None
        self.GRev = D_to_NXgraph(self.AdjMatRev, G_directed) if self.AdjMatRev is not None else None

    def FlattenHyper(self):
        super().FlattenHyper()
        self.G = D_to_NXgraph(self.AdjMat, self.G_directed)

    def FlattenHyper_Reverse(self):
        super().FlattenHyper_Reverse()
        self.GRev = D_to_NXgraph(self.AdjMatRev, self.G_directed)



