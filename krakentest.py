import csv
import json
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from datamarket import agg_dataset, split_mask, select_features

def linear_sanitycheck(dirname):
    base_t = {}
    bname = ''
    sellers = []
    with open(os.path.join(dirname, 'schema.csv'), 'r') as fh:
        reader = csv.reader(fh)
        for row in reader:
            print(row)
            bname = os.path.join(dirname, row[0])
            base_t[row[2]] = {}
            jks = row[1]
            ljk = jks.split('--')[0]
            rjk = jks.split('--')[1]
            base_t[row[2]]['left_k'] = ljk
            base_t[row[2]]['right_k'] = rjk
    
    buyer_ds = pd.read_csv(bname, header=0)
    #buyer_ds.rename(columns={'f4k' : '4index', 'f109k' : '109index'})
    targ_val = {'fail' : 0, 'nofail' : 1}
    buyer_ds['result'] = [targ_val[item] for item in buyer_ds['result']]
    
    # train test split
    msk = split_mask(len(buyer_ds)) < 0.8
    b_train = buyer_ds[msk].copy()
    b_test = buyer_ds[~msk].copy()
    
    for k in base_t:
        fname = os.path.join(dirname, k)
        newdf = pd.read_csv(fname, header=0)
        newdf.set_index(base_t[k]['right_k'])
        b_train.set_index(base_t[k]['left_k'])
        b_test.set_index(base_t[k]['left_k'])
        joindf = b_train.merge(newdf, left_index=True, right_index=True)
        #print(joindf.columns)
        #now, train regression
        excols = ['f4k', 'f109k']
        excols += [c for c in joindf.columns if 'event_id' in c]
        X = joindf.loc[:, ~joindf.columns.isin(excols)]
        y = joindf['result']
        jtestdf = b_test.merge(newdf, left_index=True, right_index=True)
        X_test = jtestdf.loc[:, ~jtestdf.columns.isin(excols)]
        y_test = jtestdf['result']
        reg = LinearRegression().fit(X, y)
        print(reg.coef_)
        print(k + ': ' + str(reg.score(X_test, y_test)))
        
linear_sanitycheck('../ardads/arda-datasets/datasets/kraken')

def use_kraken(dirname, tbl_name, limit=1, m=1):
    t_files = []
    base_t = {}
    bname = ''
    sellers = []
    with open(os.path.join(dirname, 'schema.csv'), 'r') as fh:
        reader = csv.reader(fh)
        for row in reader:
            print(row)
            bname = os.path.join(dirname, row[0])
            base_t[row[2]] = {}
            jks = row[1]
            ljk = jks.split('--')[0]
            rjk = jks.split('--')[1]
            base_t[row[2]]['left_k'] = ljk
            base_t[row[2]]['right_k'] = rjk
    ind = 0
    for k in base_t:
        fname = os.path.join(dirname, k)
        newdf = pd.read_csv(fname, header=0)
        new_tbl = agg_dataset()
        #new_tbl.load(newdf, [], ["Provider City"], tbl_name + str(i))
        new_tbl.load(newdf, [], [base_t[k]['right_k']], k)
        new_tbl.find_features()
        new_tbl.remove_redundant_columns()
        new_tbl.compute_agg(True)
        #res.append((new_tbl, "Provider City"))
        sellers.append((new_tbl, base_t[k]['right_k']))
        ind += 1
    
    buyer_ds = pd.read_csv(bname, header=0)
    buyer_ds.rename(columns={'f4k' : '4index', 'f109k' : '109index'})
    targ_val = {'fail' : 0, 'nofail' : 1}
    buyer_ds['result'] = [targ_val[item] for item in buyer_ds['result']]

    # train test split
    msk = split_mask(len(buyer_ds)) < 0.8
    taxi_train = buyer_ds[msk].copy()
    taxi_test = buyer_ds[~msk].copy()

    taxi_train_data = agg_dataset()
    taxi_train_data.load(taxi_train, [], [['event_id', '4index', '109index']], "krakenbase")
    taxi_train_data.process_target('result')
    taxi_train_data.to_numeric_and_impute_all()
    taxi_train_data.remove_redundant_columns()
    taxi_train_data.compute_agg()

    taxi_test_data = agg_dataset()
    taxi_test_data.load(taxi_test, [], [['event_id', '4index', '109index']], "krakenbase")
    taxi_test_data.process_target("result")
    taxi_test_data.to_numeric_and_impute_all()
    taxi_test_data.remove_redundant_columns()
    taxi_test_data.compute_agg()

    # find m best datasets to augment
    bought = set()
    y = "result"

    for i in range(m):
        best_seller = None
        best_seller_attrs = []
        best_dimension = None
        best_r2 = 0

        for sellerdata, dimension in sellers:
            #if 'kr_sensor' in sellerdata.name:
            #    continue
            # check if current seller has been bought
            if sellerdata.name in bought:
                continue
            print("Sellerdata name: {}".format(sellerdata.name))

            # find the attributes and r2 of augmenting
            cur_atts, final_r2 = select_features(taxi_train_data, taxi_test_data, sellerdata, dimension, 4, y)
            print(final_r2)
    #         cur_atts, final_r2 = select_features(gender_train, gender_test, sellerdata, dimension,10)

            if final_r2 >= best_r2:
                best_seller = sellerdata
                best_dimension = dimension
                best_seller_attrs = cur_atts
                best_r2 = final_r2


        print(best_seller.name, best_seller_attrs, best_r2)

#use_kraken('../ardads/arda-datasets/datasets/kraken', 'cur_tbl', limit=1, m=1)

