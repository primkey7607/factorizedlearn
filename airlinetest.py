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



