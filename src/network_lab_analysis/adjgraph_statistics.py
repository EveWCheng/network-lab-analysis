class NetworkX_Statistics:
    def __init__(self, AdjMat, G_directed, index_to_name=None):
        self.AdjMat = AdjMat
        self.G_directed = G_directed
        self.G = D_to_NXgraph(self.AdjMat, G_directed) if self.AdjMat is not None else None

    def average_max_flow(self,node_to_size=None):
        G = self.G
        cap_data = {e: G.edges[e]['weight'] for e in G.edges}
        nx.set_edge_attributes(G, cap_data,'capacity')
        maxflow = 0
        den = 0
        iter_func = itertools.permutations
        for u, v in iter_func(G, 2):
            if node_to_size == None:
                maxflow += nx.maximum_flow_value(G, u, v)
            else:
                """scaled by the number of nodes in the source,see the network as a traffic network"""
                maxflow += node_to_size(u)*nx.maximum_flow_value(G, u, v)
            den += 1
        average_flow = maxflow/den
        return average_flow

    def transitivity(self):
        return nx.transitivity(self.G)


#from distance matrix to networkx object
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

