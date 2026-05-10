import csv
import numpy as np
import itertools
import os
import tomllib
from pathlib import Path
import pandas as pd
import networkx as nx

#labels: [poi, year]

class SetUp:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "../config.toml"), "rb") as f:
            _config = tomllib.load(f)
        for section in _config.values():
            for key, value in section.items():
                setattr(self, key, value)


def Filter(ele):
    return ele not in ["NA", "NK", "other"]

def DeleteNANK(l):
    return list(filter(Filter, l))

def real_file_name(labels,setup):
    fn = Path(setup.raw_data_dir) / f"{'_'.join(labels)}.csv"
    return fn


def sim_file_name(labels,n,sim_option, setup):
    fn = Path(setup.sim_dir) / sim_option / Path(*labels) / f"simulation_{'_'.join(labels)}_{n}.csv"
    return fn

def import_data(fn,setup):
    df = pd.read_csv(fn,usecols=setup.headers,keep_default_na=False)
    no_people = len(list(df[setup.id_].unique()))
    if len(list(df[setup.id_]))-no_people != 0:
        print("warning: repeated people")
    return df,no_people

def write_dict_to_file(keys,values,dir_name,fn):
    with open(Path(dir_name) / fn, 'w') as f:
        writer = csv.writer(f, delimiter='\t')              
        writer.writerows(zip(keys,values))

def class_to_index_gen(df,labels,setup):
    classes = []
    for col in setup.attributes:
        classes += list(DeleteNANK(df[col].unique()))
    seen = set()
    dupes = [c for c in classes if c in seen or seen.add(c)]
    if dupes:
        print(f"warning: duplicate classes across columns: {dupes}")
    class_to_index = dict(zip(classes,range(len(classes))))

    dir_name = Path(setup.class_dict_dir) / Path(*labels)
    os.makedirs(dir_name,mode=0o777,exist_ok=True)
    write_dict_to_file(class_to_index.values(),class_to_index.keys(),dir_name,"classifier_index.csv")
    return class_to_index

def index_to_mp(df,labels,setup):
    mp_list = [i[5:] for i in list(df[setup.id_].unique())]
    dir_name = Path(setup.mp_dict_dir) / Path(*labels)
    os.makedirs(dir_name,mode=0o777,exist_ok=True)
    write_dict_to_file(range(len(mp_list)),mp_list,dir_name,"mp_index.csv")

def incidence_matrix(class_to_index,no_people,df,labels,setup):
    IMat = np.zeros((no_people,len(class_to_index.keys())))
    id_col = setup.id_
    attr_cols = setup.attributes
    seen_ids = set()
    row_no = 0
    for _,row in df.iterrows():
        id_ = row[id_col]
        if id_ not in seen_ids:
            for col in attr_cols:
                if Filter(row[col]):
                    IMat[row_no,class_to_index[row[col]]] = 1
            seen_ids.add(id_)
            row_no+=1
    dir_name = Path(setup.incimat_dir) / Path(*labels)
    os.makedirs(dir_name,mode=0o777,exist_ok=True)
    np.savetxt(dir_name / f"{'_'.join(labels)}.txt",IMat,fmt='%.4e')


def main():
    setup = SetUp() 
    for poi in ["LP","ALP"]:
        for year in np.arange(1947,2020,dtype=np.int64):
            if setup.sim_real == "sim":
                pass
            elif setup.sim_real == "real":
                labels = [poi,str(year)]
                fn = real_file_name(labels,setup)
                df,no_people = import_data(fn,setup)
                class_to_index = class_to_index_gen(df,labels,setup)
                incidence_matrix(class_to_index,no_people,df,labels,setup)


main()


