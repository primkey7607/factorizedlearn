import csv
import json
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from datamarket import agg_dataset, split_mask, select_features

def linear_sanitycheck(dirname, tbl_name, bname):
    new_sellers = []
    t_files = []
    
    for f in os.listdir(dirname):
        if f.endswith('.csv'):
            t_files.append(os.path.join(dirname, f))
    
    for i,t in enumerate(t_files):
        
        # try:
        #     newdf = pd.read_csv(t)
        # except UnicodeDecodeError:
        #     newdf = pd.read_csv(t, encoding='cp1252')
        newdf = pd.read_csv(t, nrows=100)
        covidcols = [c for c in newdf.columns if 'COVID' in c]
        #print(covidcols)
        if 'Residents Total Confirmed COVID-19' not in newdf.columns:
            print("Target Not found!")
        
        new_sellers.append((t,newdf))
        
        #new_tbl.load(newdf, [], ["Provider City"], tbl_name + str(i))
        #res.append((new_tbl, "Provider City"))
    
    buyer_ds = pd.read_csv(bname, nrows=1200, encoding='cp1252')
    print(buyer_ds)
    #buyer_ds = buyer_ds.dropna(axis=1, how='all')
    #buyer_ds = buyer_ds.dropna(axis=0)
    buyer_ds = buyer_ds.select_dtypes(include=[np.number])
    buyer_ds.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # train test split
    msk = split_mask(len(buyer_ds)) < 0.8
    b_train = buyer_ds[msk].copy()
    b_test = buyer_ds[~msk].copy()
    
    numerics = ['float16', 'float32', 'float64']
    excols = ['Residents Total Confirmed COVID-19']
    excols += [c for c in b_train.columns if 'COVID' in c]
    
    X_b = b_train.loc[:, ~b_train.columns.isin(excols)]
    y_b = b_train['Residents Total Confirmed COVID-19']
    print(X_b)
    print(y_b)
    
    X_btest = b_test.loc[:, ~b_test.columns.isin(excols)]
    y_btest = b_test['Residents Total Confirmed COVID-19']
    print(X_btest)
    print(y_btest)
    
    
    reg = LinearRegression().fit(X_b, y_b)
    print('Original: ' + str(reg.score(X_btest, y_btest)))
    
    #and now, we'll union each table with this and see how much performance improves.
    for i,t in enumerate(t_files):
        newdf = pd.read_csv(t, nrows=1000, encoding='cp1252')
        newdf = newdf.select_dtypes(include=[np.number])
        newdf.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        X_new = newdf.loc[:, ~newdf.columns.isin(excols)]
        y_new = newdf['Residents Total Confirmed COVID-19']
        unionX = pd.concat([X_b, X_new])
        uniony = pd.concat([y_b, y_new])
        reg = LinearRegression().fit(unionX, uniony)
        print(t + ': ' + str(reg.score(X_btest, y_btest)))
    
    
    
    # for k,newdf in new_sellers:
    #     newdf.set_index('Federal Provider Number')
    #     newdf = newdf.dropna(axis=0)
    #     b_train.set_index('Federal Provider Number')
    #     b_test.set_index('Federal Provider Number')
    #     joindf = b_train.merge(newdf, left_index=True, right_index=True)
    #     print(joindf.columns)
    #     #now, train regression
    #     excols = joindf.select_dtypes(['number']).columns
    #     #excols += [c for c in joindf.columns if 'event_id' in c]
    #     excols += ['Residents Total Confirmed COVID-19']
    #     X = joindf.loc[:, ~joindf.columns.isin(excols)]
        
    #     y = joindf['Residents Total Confirmed COVID-19']
    #     jtestdf = b_test.merge(newdf, left_index=True, right_index=True)
    #     X_test = jtestdf.loc[:, ~jtestdf.columns.isin(excols)]
    #     y_test = jtestdf['Residents Total Confirmed COVID-19']
    #     reg = LinearRegression().fit(X, y)
    #     #print(reg.coef_)
    #     print(k + ': ' + str(reg.score(X_test, y_test)))

linear_sanitycheck('../healthds', 'cur_tbl', '../healthds/COVID-19 Nursing Home Data 12.26.2021.csv')
    

def create_and_execute_market(dirname, tbl_name, limit=1, m=1):
    res = []
    t_files = []
    
    for f in os.listdir(dirname):
        if f.endswith('.csv'):
            t_files.append(os.path.join(dirname, f))
    
    for i,t in enumerate(t_files):
        
        # try:
        #     newdf = pd.read_csv(t)
        # except UnicodeDecodeError:
        #     newdf = pd.read_csv(t, encoding='cp1252')
        newdf = pd.read_csv(t, encoding='cp1252', nrows=100)
        print(newdf.columns)
        if 'Provider City' not in newdf.columns:
            continue
        
        new_tbl = agg_dataset()
        #new_tbl.load(newdf, [], ["Provider City"], tbl_name + str(i))
        new_tbl.load(newdf, [], ["Provider Zip Code"], tbl_name + str(i))
        new_tbl.find_features()
        new_tbl.remove_redundant_columns()
        new_tbl.compute_agg(True)
        #res.append((new_tbl, "Provider City"))
        res.append((new_tbl, "Provider Zip Code"))
    
    return res
    

def run_healthbuyerds(bname):
    new_sellers = create_and_execute_market('../healthds', 'cur_tbl')
    # read base_data from 
    buyer_ds = pd.read_csv(bname, nrows=100)

    # train test split
    msk = split_mask(len(buyer_ds)) < 0.8
    taxi_train = buyer_ds[msk].copy()
    taxi_test = buyer_ds[~msk].copy()

    taxi_train_data = agg_dataset()
    taxi_train_data.load(taxi_train, ["Resident Access to Testing in Facility", "Residents Weekly Admissions COVID-19"], ["Provider City", "Provider Zip Code"], "taxi")
    taxi_train_data.process_target('Residents Total Confirmed COVID-19')
    taxi_train_data.to_numeric_and_impute_all()
    taxi_train_data.remove_redundant_columns()
    taxi_train_data.compute_agg()

    taxi_test_data = agg_dataset()
    taxi_test_data.load(taxi_test, ["Resident Access to Testing in Facility", "Residents Weekly Admissions COVID-19"], ["Provider City", "Provider Zip Code"], "taxi")
    taxi_test_data.process_target("Residents Total Confirmed COVID-19")
    taxi_test_data.to_numeric_and_impute_all()
    taxi_test_data.remove_redundant_columns()
    taxi_test_data.compute_agg()

    # find m best datasets to augment
    bought = set()
    m = 1
    y = "Residents Total Confirmed COVID-19"

    for i in range(m):
        best_seller = None
        best_seller_attrs = []
        best_dimension = None
        best_r2 = 0

        for sellerdata, dimension in new_sellers:
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

#run_healthbuyerds('../healthds/COVID-19 Nursing Home Data 12.26.2021.csv')
#run_healthbuyerds('../fsds/ALLAML.mat')


