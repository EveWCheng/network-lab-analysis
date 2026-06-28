import csv
import networkx as nx
import numpy as np
import os
import tomllib
from pathlib import Path

class SetUp:
    def __init__(self):
        config_path = Path(os.path.abspath(__file__)).parent.parent / "config.toml"
        with open(config_path, "rb") as f:
            _config = tomllib.load(f)
        config_dir = config_path.parent
        for section in _config.values():
            for key, value in section.items():
                if key.endswith("_dir"):
                    setattr(self, key, str((config_dir / value).resolve()))
                else:
                    setattr(self, key, value)

    def incimat_path(self, labels):
        return Path(self.incimat_dir) / Path(*labels) / f"{'_'.join(labels)}.txt"

    def class_dict_path(self, labels):
        return Path(self.class_dict_dir) / Path(*labels) / "classifier_index.csv"

    def mp_dict_path(self, labels):
        return Path(self.mp_dict_dir) / Path(*labels) / "mp_index.csv"

    def bouquet_output_path(self, poi):
        out_dir = Path(self.bouquet_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"bouquet_{poi}.csv"


def read_incidence_matrix(incimat_path):
    return np.loadtxt(incimat_path)


def read_classifier_index(classifier_index_path):
    index_to_class = {}
    with open(classifier_index_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            index_to_class[int(row[0])] = row[1]
    return index_to_class


def read_mp_index(mp_index_path):
    index_to_mp = {}
    with open(mp_index_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            index_to_mp[int(row[0])] = row[1]
    return index_to_mp


def flatten_hyper(hypermat):
    return hypermat.T @ hypermat


def find_bridge_node(i, j, hypermat):
    for r in range(hypermat.shape[0]):
        if hypermat[r, i] != 0 and hypermat[r, j] != 0:
            return r
    return None


def classifier_betweenness_percentiles(adjmat):
    """Returns {classifier_index: betweenness_percentile} for the classifier co-occurrence graph."""
    rows, cols = np.where(np.triu(adjmat > 0, k=1))
    G = nx.Graph()
    G.add_nodes_from(range(adjmat.shape[0]))
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return _percentile_ranks(nx.betweenness_centrality(G))


def bouquet_measure(i, j, adjmat):
    """
    Returns (shortest_path_length, n_shortest_paths) between classifiers i and j,
    excluding the direct i-j edge. Returns (np.inf, 0) if no alternative path exists.
    """
    rows, cols = np.where(np.triu(adjmat > 0, k=1))
    G = nx.Graph()
    G.add_nodes_from(range(adjmat.shape[0]))
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    G.remove_edge(i, j)
    try:
        paths = list(nx.all_shortest_paths(G, source=i, target=j))
        return len(paths[0]) - 1, len(paths)
    except nx.NetworkXNoPath:
        return np.inf, 0


def build_mp_graphs(poi, setup, years=np.arange(1947, 2020, dtype=np.int64)):
    """Returns {year: G_mp} where G_mp is the MP-MP co-occurrence graph (hypermat @ hypermat.T)."""
    result = {}
    for year in years:
        labels = [poi, str(year)]
        imat_path = setup.incimat_path(labels)
        if not imat_path.exists():
            continue
        hypermat = read_incidence_matrix(imat_path)
        mp_adjmat = hypermat @ hypermat.T
        mp_rows, mp_cols = np.where(np.triu(mp_adjmat > 0, k=1))
        G_mp = nx.Graph()
        G_mp.add_nodes_from(range(hypermat.shape[0]))
        G_mp.add_edges_from(zip(mp_rows.tolist(), mp_cols.tolist()))
        result[year] = G_mp
    return result


def _percentile_ranks(values_dict):
    values = list(values_dict.values())
    return {k: sum(v <= val for v in values) / len(values) * 100 for k, val in values_dict.items()}


def mp_effective_size_percentiles(poi, setup, years=np.arange(1947, 2020, dtype=np.int64), graphs=None):
    """Returns {year: {mp_index: percentile_rank}} of effective size within that year's distribution."""
    if graphs is None:
        graphs = build_mp_graphs(poi, setup, years)
    return {year: _percentile_ranks(nx.effective_size(G)) for year, G in graphs.items()}


def mp_constraint_percentiles(poi, setup, years=np.arange(1947, 2020, dtype=np.int64), graphs=None):
    """Returns {year: {mp_index: percentile_rank}} of constraint within that year's distribution."""
    if graphs is None:
        graphs = build_mp_graphs(poi, setup, years)
    return {year: _percentile_ranks(nx.constraint(G)) for year, G in graphs.items()}


def mp_betweenness_percentiles(poi, setup, years=np.arange(1947, 2020, dtype=np.int64), graphs=None):
    """Returns {year: {mp_index: percentile_rank}} of betweenness centrality within that year's distribution."""
    if graphs is None:
        graphs = build_mp_graphs(poi, setup, years)
    return {year: _percentile_ranks(nx.betweenness_centrality(G)) for year, G in graphs.items()}


def detect_bouquets(poi, setup, threshold=0, measure_threshold=0, min_count=None, years=np.arange(1947, 2020, dtype=np.int64)):
    """
    For each year and the given party (poi), find pairs of classifiers (i, j) where
    exactly one MP holds both — and both classifiers are large enough (smaller > threshold * no_people).
    Bouquets are further filtered by measure_threshold: only kept when bouquet_measure > measure_threshold,
    where bouquet_measure is the shortest path between i and j excluding the direct i-j edge.
    Writes results to output/bouquets/bouquet_{poi}.csv.
    """
    out_path = setup.bouquet_output_path(poi)
    graphs = build_mp_graphs(poi, setup, years)
    eff_size_pcts = mp_effective_size_percentiles(poi, setup, graphs=graphs)
    constraint_pcts = mp_constraint_percentiles(poi, setup, graphs=graphs)
    betweenness_pcts = mp_betweenness_percentiles(poi, setup, graphs=graphs)
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["year", "class_1", "class_2", "no_1", "no_2", "mp_id", "no_people", "shortest_path", "n_shortest_paths", "mp_eff_size_pct", "mp_constraint_pct", "mp_betweenness_pct", "class_1_betweenness_pct", "class_2_betweenness_pct"])
        for year in years:
            labels = [poi, str(year)]
            imat_path = setup.incimat_path(labels)
            if not imat_path.exists():
                continue
            hypermat = read_incidence_matrix(imat_path)
            no_people = hypermat.shape[0]
            index_to_class = read_classifier_index(setup.class_dict_path(labels))
            index_to_mp = read_mp_index(setup.mp_dict_path(labels))
            adjmat = flatten_hyper(hypermat)
            class_btwn = classifier_betweenness_percentiles(adjmat)
            indices = np.argwhere(adjmat == 1)
            for i, j in indices:
                no_1, no_2 = int(adjmat[i, i]), int(adjmat[j, j])
                size_ok = no_2 > min_count if min_count is not None else no_2 > threshold * no_people
                if no_1 > no_2 and size_ok:
                    shortest_path, n_shortest = bouquet_measure(i, j, adjmat)
                    if shortest_path > measure_threshold and n_shortest < 2:
                        bridge = find_bridge_node(i, j, hypermat)
                        mp_id = index_to_mp.get(bridge, "unknown") if bridge is not None else "unknown"
                        eff_size_pct = eff_size_pcts[year][bridge] if bridge is not None else None
                        constraint_pct = constraint_pcts[year][bridge] if bridge is not None else None
                        betweenness_pct = betweenness_pcts[year][bridge] if bridge is not None else None
                        writer.writerow([year, index_to_class[i], index_to_class[j], no_1, no_2, mp_id, no_people, shortest_path, n_shortest, eff_size_pct, constraint_pct, betweenness_pct, class_btwn[i], class_btwn[j]])
    print(f"Bouquets written to {out_path}")


def main():
    setup = SetUp()
    for poi in ["LP", "ALP"]:
        detect_bouquets(poi, setup, threshold=0.05, measure_threshold=1)


main()
