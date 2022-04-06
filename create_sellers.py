import csv
import json
import pandas as pd
import os
import numpy as np
from datamarket import agg_dataset, split_mask, select_features

def get_sellers_from_dir(dirname, tbl_name, limit=1):
    res = []
    fkname = dirname + os.path.sep + 'join.json'
    datadir = dirname + os.path.sep + 'data/'
    #we can use this map
    taxicol2proc = np.load('taxi_processed/data/feature_dic_map_taxi.npy', allow_pickle=True).item()
    taxifile2proc = np.load('taxi_processed/data/dic_map_taxi.npy', allow_pickle=True).item()
    
    with open(fkname, 'r') as fkfh:
        fkmap = json.load(fkfh)
    
    ind = 0
    for fname in fkmap:
        if ind > limit:
            break
        f = taxifile2proc[fname + '.csv']
        sfcols = set()
        fcols = []
        newdf = pd.read_csv(os.path.join(datadir, f))
        join_matches = fkmap[fname]
        join_key_s = set()
        jk_map = {}
        for jm in join_matches:
            join_key = jm['right_columns_names'][0][0]
            lj_key = jm['left_columns_names'][0][0]
            jk_map[join_key] = lj_key
            join_key_s.add(join_key)
        join_keys = list(join_key_s)
        
        for jk in join_keys:
            jtup = (fname + '.csv', jk)
            ljk = jk_map[jk]
            col = taxicol2proc[jtup]
            #change the name in df
            newdf = newdf.rename({col : ljk}, axis='columns')
            sfcols.add(ljk)
        
        fcols = list(sfcols)
        
        new_tbl = agg_dataset()
        new_tbl.load(newdf, [], fcols, tbl_name + str(ind))
        new_tbl.find_features()
        new_tbl.remove_redundant_columns()
        new_tbl.compute_agg(True)
        ind += 1
        if len(fcols) == 1:
            res.append((new_tbl, fcols[0]))
        elif len(fcols) > 1:
            res.append((new_tbl, tuple(fcols)))
    
    return res
new_sellers = get_sellers_from_dir('taxi_processed', 'cur_tbl')
# read base_data from 
buyer_ds = pd.read_csv("taxi/base_data.csv")

# train test split
msk = split_mask(len(buyer_ds)) < 0.8
taxi_train = buyer_ds[msk].copy()
taxi_test = buyer_ds[~msk].copy()

taxi_train_data = agg_dataset()
taxi_train_data.load(taxi_train, ["n. trips", "n. collisions"], ["datetime"], "taxi")
taxi_train_data.process_target("n. collisions")
taxi_train_data.to_numeric_and_impute_all()
taxi_train_data.remove_redundant_columns()
taxi_train_data.compute_agg()

taxi_test_data = agg_dataset()
taxi_test_data.load(taxi_test, ["n. trips", "n. collisions"], ["datetime"], "taxi")
taxi_test_data.process_target("n. collisions")
taxi_test_data.to_numeric_and_impute_all()
taxi_test_data.remove_redundant_columns()
taxi_test_data.compute_agg()

# find m best datasets to augment
bought = set()
m = 1
y = "n. collisions"

for i in range(m):
    best_seller = None
    best_seller_attrs = []
    best_dimension = None
    best_r2 = 0

    for sellerdata, dimension in new_sellers:
        # check if current seller has been bought
        if sellerdata.name in bought:
            continue
        print(sellerdata.name)

        # find the attributes and r2 of augmenting
        cur_atts, final_r2 = select_features(taxi_train_data, taxi_test_data, sellerdata, dimension, 4, y)
#         cur_atts, final_r2 = select_features(gender_train, gender_test, sellerdata, dimension,10)

        if final_r2 >= best_r2:
            best_seller = sellerdata
            best_dimension = dimension
            best_seller_attrs = cur_atts
            best_r2 = final_r2


    print(best_seller.name, best_seller_attrs, best_r2)


