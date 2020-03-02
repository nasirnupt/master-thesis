import pickle
import numpy as np
import sys
import os
import pandas as pd

if __name__ == "__main__":
    base_dir = "./data/experiment_results"
    path = os.path.join(base_dir,sys.argv[1])
    path = os.path.join(path,sys.argv[2])
    values_map = {"f1":0,"precision":1,"recall":2}

    with open(path, "rb") as f:
        dataset = pickle.load(f)

    dataset_list = dataset.keys()
    res_pro = {}
    for name in dataset_list:
        data = dataset[name]
        data_ver_list = data.keys()
        res = []
        for ver_list in data_ver_list:
            res.append(round(data[ver_list][values_map[sys.argv[-1]]],2))
        res_pro[name.split("-")[0]] = res
    df = pd.DataFrame(res_pro)
    print(df.to_markdown())









