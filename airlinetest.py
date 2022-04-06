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
    with open(os.path.join(dirname, 'schema.csv'), 'r') as fh:
        reader = csv.reader(fh)
        for row in reader:
            #print(row)
            bname = os.path.join(dirname, row[0])
            base_t[row[2]] = {}
            jks = row[1]
            ljks = jks.split('--')[0]
            rjks = jks.split('--')[1]
            if '*:' in ljks:
                ljk = ljks.split('*:')
                for i,e in enumerate(ljk):
                    if e == 'airport_code':
                        ljk[i] = 'airport'
                    if e == 'time':
                        ljk[i] = 'date'
                print(ljk)
            else:
                ljk = ljks
                if ljk == 'airport_code':
                    ljk = 'airport'
                if ljk == 'time':
                    ljk = 'date'
            
            if '*:' in rjks:
                rjk = rjks.split('*:')
                for i,e in enumerate(rjk):
                    if e == 'airport_code':
                        rjk[i] = 'airport'
                    if e == 'time':
                        rjk[i] = 'date'
                print(rjk)
                        
            else:
                rjk = rjks
                if rjk == 'airport_code':
                    rjk = 'airport'
                if rjk == 'time':
                    rjk = 'date'
            
            
            base_t[row[2]]['left_k'] = ljk
            base_t[row[2]]['right_k'] = rjk
    
    print(base_t)
    
    buyer_ds = pd.read_csv(bname, header=0)
    #buyer_ds.rename(columns={'f4k' : '4index', 'f109k' : '109index'})
    #targ_val = {'fail' : 0, 'nofail' : 1}
    #buyer_ds['result'] = [targ_val[item] for item in buyer_ds['result']]
    
    # train test split
    msk = split_mask(len(buyer_ds)) < 0.8
    b_train = buyer_ds[msk].copy()
    b_test = buyer_ds[~msk].copy()
    
    for k in base_t:
        print(k)
        fname = os.path.join(dirname, k)
        newdf = pd.read_csv(fname, header=0)
        newdf.set_index(base_t[k]['right_k'])
        newdf = newdf.dropna(axis=0)
        b_train.set_index(base_t[k]['left_k'])
        b_test.set_index(base_t[k]['left_k'])
        joindf = b_train.merge(newdf, left_index=True, right_index=True)
        #print(joindf.columns)
        #now, train regression
        excols = ['departure_time', 'origin', 'destination', 'airline', 'flight', 'plane', 'airport', 'date']
        #excols += [c for c in joindf.columns if 'event_id' in c]
        excols += ['delay']
        X = joindf.loc[:, ~joindf.columns.isin(excols)]
        
        y = joindf['delay']
        jtestdf = b_test.merge(newdf, left_index=True, right_index=True)
        X_test = jtestdf.loc[:, ~jtestdf.columns.isin(excols)]
        y_test = jtestdf['delay']
        reg = LinearRegression().fit(X, y)
        #print(reg.coef_)
        print(k + ': ' + str(reg.score(X_test, y_test)))
linear_sanitycheck('../ardads/arda-datasets/datasets/airline')

def use_airline(dirname, tbl_name, limit=1, m=1):
    t_files = []
    base_t = {}
    bname = ''
    sellers = []
    with open(os.path.join(dirname, 'schema.csv'), 'r') as fh:
        reader = csv.reader(fh)
        for row in reader:
            #print(row)
            bname = os.path.join(dirname, row[0])
            base_t[row[2]] = {}
            jks = row[1]
            ljks = jks.split('--')[0]
            rjks = jks.split('--')[1]
            if '*:' in ljks:
                ljk = ljks.split('*:')
                for i,e in enumerate(ljk):
                    if e == 'airport_code':
                        ljk[i] = 'airport'
                    if e == 'time':
                        ljk[i] = 'date'
                print(ljk)
            else:
                ljk = ljks
                if ljk == 'airport_code':
                    ljk = 'airport'
                if ljk == 'time':
                    ljk = 'date'
            
            if '*:' in rjks:
                rjk = rjks.split('*:')
                for i,e in enumerate(rjk):
                    if e == 'airport_code':
                        rjk[i] = 'airport'
                    if e == 'time':
                        rjk[i] = 'date'
                print(rjk)
                        
            else:
                rjk = rjks
                if rjk == 'airport_code':
                    rjk = 'airport'
                if rjk == 'time':
                    rjk = 'date'
            
            
            base_t[row[2]]['left_k'] = ljk
            base_t[row[2]]['right_k'] = rjk
    
    print(base_t)
    
    buyer_ds = pd.read_csv(bname, header=0)
    #buyer_ds.rename(columns={'f4k' : '4index', 'f109k' : '109index'})
    #targ_val = {'fail' : 0, 'nofail' : 1}
    #buyer_ds['result'] = [targ_val[item] for item in buyer_ds['result']]
    
    # train test split
    msk = split_mask(len(buyer_ds)) < 0.8
    b_train = buyer_ds[msk].copy()
    b_test = buyer_ds[~msk].copy()
    
    ind = 0
    for k in base_t:
        fname = os.path.join(dirname, k)
        newdf = pd.read_csv(fname, header=0)
        newdf = newdf.dropna(axis=0)
            
        new_tbl = agg_dataset()
        #new_tbl.load(newdf, [], ["Provider City"], tbl_name + str(i))
        new_tbl.load(newdf, [], [base_t[k]['right_k']], k)
        new_tbl.find_features()
        new_tbl.remove_redundant_columns()
        new_tbl.compute_agg(True)
        #res.append((new_tbl, "Provider City"))
        sellers.append((new_tbl, base_t[k]['right_k']))
        ind += 1

    b_train_data = agg_dataset()
    b_train_data.load(b_train, [], ['origin', 'destination', ['departure_time', 'origin']], "airlinebase")
    b_train_data.process_target('delay')
    b_train_data.to_numeric_and_impute_all()
    b_train_data.remove_redundant_columns()
    b_train_data.compute_agg()

    b_test_data = agg_dataset()
    b_test_data.load(b_test, [], ['origin', 'destination', ['departure_time', 'origin']], "airlinebase")
    b_test_data.process_target("delay")
    b_test_data.to_numeric_and_impute_all()
    b_test_data.remove_redundant_columns()
    b_test_data.compute_agg()

    # find m best datasets to augment
    bought = set()
    y = "delay"

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
            #print("Sellerdata name: {}".format(sellerdata.name))
            #print("Dimension name: {}".format(dimension))

            # find the attributes and r2 of augmenting
            cur_atts, final_r2 = select_features(b_train_data, b_test_data, sellerdata, dimension, 4, y)
            print(final_r2)
    #         cur_atts, final_r2 = select_features(gender_train, gender_test, sellerdata, dimension,10)

            if final_r2 >= best_r2:
                best_seller = sellerdata
                best_dimension = dimension
                best_seller_attrs = cur_atts
                best_r2 = final_r2


        print(best_seller.name, best_seller_attrs, best_r2)

use_airline('../ardads/arda-datasets/datasets/airline', 'cur_tbl', limit=1, m=1)





