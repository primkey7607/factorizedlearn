from datamarket import *
import json
import time

#sellers should be accessible to all buyers.

with open('market_data.json', 'r') as openfile:
    market_data = json.load(openfile)

sellers = []

for data in market_data:
    file, dimension, name = data
    df = pd.read_csv(file)
    aggdata = agg_dataset()
    aggdata.load(df, [], dimension, name)
    aggdata.find_features()
    aggdata.remove_redundant_columns()
    aggdata.compute_agg(True)
    
    sellers.append((aggdata, dimension[0]))

def find_optimal_plan(bfile, enc=False):
    # read buyer dataset
    # gender 
    if enc:
        buyer = pd.read_csv(bfile, encoding='cp1252')
    else:
        buyer = pd.read_csv(bfile)
    # ethnic 
    # buyer = pd.read_csv("2013-2017_School_ELA_Results_-_Ethnic.csv")
    # survery
    # buyer = pd.read_csv("NH_SurveySummary_Mar2022.csv", encoding='cp1252')

    # buyer = pd.read_csv("NH_CovidVaxProvider_20220320.csv", encoding='cp1252')

    # train test split
    if 'gender' in bfile:
        msk = split_mask(len(buyer)) < 0.8
        buyer_train = buyer[msk].copy()
        buyer_test = buyer[~msk].copy()
    
        buyer_train_data = agg_dataset()
        buyer_train_data.load(buyer_train, ["Number Tested", "Mean Scale Score"], ["DBN", ["DBN","Grade"], "Year", "Category"], "buyer")
        buyer_train_data.process_target("Mean Scale Score")
        # buyer_train_data.load(buyer_train, ['Total Number of Health Deficiencies','Total Number of Fire Safety Deficiencies'], ["Federal Provider Number", 'Location', 'Processing Date'], "buyer")
        # buyer_train_data.process_target('Total Number of Health Deficiencies')
        # buyer_train_data.load(buyer_train, ['Percent Vaccinated Residents'], ["Federal Provider Number", 'Provider State'], "buyer")
        # buyer_train_data.process_target('Percent Vaccinated Residents')
        buyer_train_data.to_numeric_and_impute_all()
        buyer_train_data.remove_redundant_columns()
        buyer_train_data.compute_agg()
    
        buyer_test_data = agg_dataset()
        buyer_test_data.load(buyer_test, ["Number Tested", "Mean Scale Score"], ["DBN", ["DBN","Grade"], "Year", "Category"], "buyer")
        buyer_test_data.process_target("Mean Scale Score")
        # buyer_test_data.load(buyer_test, ['Total Number of Health Deficiencies','Total Number of Fire Safety Deficiencies'], ["Federal Provider Number", 'Location', 'Processing Date'], "buyer")
        # buyer_test_data.process_target('Total Number of Health Deficiencies')
        # buyer_test_data.load(buyer_test, ['Percent Vaccinated Residents'], ["Federal Provider Number", 'Provider State'], "buyer")
        # buyer_test_data.process_target('Percent Vaccinated Residents')
        buyer_test_data.to_numeric_and_impute_all()
        buyer_test_data.remove_redundant_columns()
        buyer_test_data.compute_agg()
        y = 'Mean Scale Score'
    
    if 'SurveySummary' in bfile:
        msk = split_mask(len(buyer)) < 0.8
        buyer_train = buyer[msk].copy()
        buyer_test = buyer[~msk].copy()
        
        buyer_train_data = agg_dataset()
        buyer_train_data.load(buyer_train, ['Total Number of Health Deficiencies','Total Number of Fire Safety Deficiencies'], ["Federal Provider Number", 'Location', 'Processing Date'], "buyer")
        buyer_train_data.process_target('Total Number of Health Deficiencies')
        # buyer_train_data.load(buyer_train, ['Percent Vaccinated Residents'], ["Federal Provider Number", 'Provider State'], "buyer")
        # buyer_train_data.process_target('Percent Vaccinated Residents')
        buyer_train_data.to_numeric_and_impute_all()
        buyer_train_data.remove_redundant_columns()
        buyer_train_data.compute_agg()
    
        buyer_test_data = agg_dataset()
        buyer_test_data.load(buyer_test, ['Total Number of Health Deficiencies','Total Number of Fire Safety Deficiencies'], ["Federal Provider Number", 'Location', 'Processing Date'], "buyer")
        buyer_test_data.process_target('Total Number of Health Deficiencies')
        # buyer_test_data.load(buyer_test, ['Percent Vaccinated Residents'], ["Federal Provider Number", 'Provider State'], "buyer")
        # buyer_test_data.process_target('Percent Vaccinated Residents')
        buyer_test_data.to_numeric_and_impute_all()
        buyer_test_data.remove_redundant_columns()
        buyer_test_data.compute_agg()
        y = 'Total Number of Health Deficiencies'
    
    m = 3

    for i in range(m):
        best_seller = None
        best_seller_attrs = []
        best_dimension = None
        best_r2 = 0

        for sellerdata, dimension in sellers:
            print(sellerdata.name)
    #         pd.merge(buyer_train_data.data, sellerdata.data, left_on=dimension, right_on=dimension)
            # check if current seller has been bought
            if sellerdata.name in buyer_train_data.datasets:
                continue

            # find the attributes and r2 of augmenting
            cur_atts, final_r2 = select_features(buyer_train_data, buyer_test_data, sellerdata, dimension, 6, y)
    #         cur_atts, final_r2 = select_features(buyer_train, buyer_test, sellerdata, dimension,10)
            print(cur_atts, final_r2)
            if final_r2 > best_r2:
                best_seller = sellerdata
                best_dimension = dimension
                best_seller_attrs = cur_atts
                best_r2 = final_r2


        print(best_seller.name, best_seller_attrs, best_r2)
        
        if len([x for x in best_seller_attrs if x in best_seller.X]) == 0:
            buyer_train_data.datasets.add(best_seller)
            buyer_test_data.datasets.add(best_seller)
        else:
            buyer_train_data.absorb(best_seller, best_dimension, best_seller_attrs + [buyer_train_data.name + ":" + y])
            buyer_test_data.absorb(best_seller, best_dimension, best_seller_attrs + [buyer_train_data.name + ":" + y])

num_reps = 2
start = time.time()
for i in range(num_reps):
    #find_optimal_plan('gender.csv')
    find_optimal_plan("NH_SurveySummary_Mar2022.csv", enc=True)
end = time.time()
tot_runtime = end - start
    


