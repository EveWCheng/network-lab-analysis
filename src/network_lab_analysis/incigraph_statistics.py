import numpy as np
import itertools
import networkx as nx

class InciGraph_Statistics:
    def __init__(self, edge_weight_option, InciMat, row_index_to_name=None, col_index_to_name=None):
        self.InciMat = InciMat
        self.edge_weight_option = edge_weight_option
        self.row_index_to_name = row_index_to_name if row_index_to_name is not None else {i: i for i in range(len(InciMat))}
        self.col_index_to_name = col_index_to_name if col_index_to_name is not None else {j: j for j in range(len(InciMat[0]))}

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
        return AdjMat

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
        return AdjMat

    def flatten(self,flatten_dir):
        if flatten_dir == "col":
            return self.FlattenHyper()
        elif flatten_dir == "row":
            return self.FlattenHyper_Reverse()
        else:
            print("Warning: no adj matrix is produced")



