import csv
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


def detect_bouquets(poi, setup, threshold=0.1, years=np.arange(1947, 2020, dtype=np.int64)):
    """
    For each year and the given party (poi), find pairs of classifiers (i, j) where
    exactly one MP holds both — and both classifiers are large enough (smaller > threshold * no_people).
    Writes results to data/bouquets/bouquet_{poi}.csv.
    """
    out_path = setup.bouquet_output_path(poi)
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["year", "class_1", "class_2", "no_1", "no_2", "mp_id", "no_people"])
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
            indices = np.argwhere(adjmat == 1)
            for i, j in indices:
                no_1, no_2 = int(adjmat[i, i]), int(adjmat[j, j])
                if no_1 > no_2 and no_2 > threshold * no_people:
                    bridge = find_bridge_node(i, j, hypermat)
                    mp_id = index_to_mp.get(bridge, "unknown") if bridge is not None else "unknown"
                    writer.writerow([year, index_to_class[i], index_to_class[j], no_1, no_2, mp_id, no_people])
                    print(f"  {year}: {index_to_class[i]} ({no_1}) -- {index_to_class[j]} ({no_2}) via {mp_id}")
    print(f"Bouquets written to {out_path}")


def test(poi="ALP", year=1965):
    setup = SetUp()
    labels = [poi, str(year)]
    hypermat = read_incidence_matrix(setup.incimat_path(labels))
    print("hypermat shape:", hypermat.shape)
    detect_bouquets(poi, setup, years=np.array([year]))


def main():
    setup = SetUp()
    for poi in ["LP", "ALP"]:
        detect_bouquets(poi, setup)


test()
