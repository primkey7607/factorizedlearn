from datamarket import *
from request_cache import RequestCache
import pickle as pkl
import json
import time

#sellers should be accessible to all buyers.

with open('market_data.json', 'r') as openfile:
    market_data = json.load(openfile)

sellers = []

#ind = 0
#limit = 10
for data in market_data:
    # if ind > limit:
    #     break
    # else:
    #     ind += 1
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
    aug_plan = []

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
            #...but obviously, if we can't join with a dataset, we need to avoid it, right?
            if dimension not in buyer_train_data.agg_dimensions:
                continue
            cur_atts, final_r2 = select_features(buyer_train_data, buyer_test_data, sellerdata, dimension, 6, y)
    #         cur_atts, final_r2 = select_features(buyer_train, buyer_test, sellerdata, dimension,10)
            print(cur_atts, final_r2)
            if final_r2 > best_r2:
                best_seller = sellerdata
                best_dimension = dimension
                best_seller_attrs = cur_atts
                best_r2 = final_r2
        
        print(best_seller.name, best_seller_attrs, best_r2)
        aug_plan.append((best_dimension, best_seller_attrs))

        # if best_seller != None:
        #     print(best_seller.name, best_seller_attrs, best_r2)
        # else:
        #     print("No best seller found...")
        
        if len([x for x in best_seller_attrs if x in best_seller.X]) == 0:
            buyer_train_data.datasets.add(best_seller)
            buyer_test_data.datasets.add(best_seller)
        else:
            buyer_train_data.absorb(best_seller, best_dimension, best_seller_attrs + [buyer_train_data.name + ":" + y])
            buyer_test_data.absorb(best_seller, best_dimension, best_seller_attrs + [buyer_train_data.name + ":" + y])
    return aug_plan
    

def cache_optimal_plan(bfile, cache_state=None):
    #initialize the cache...
    if cache_state == None:
        cache = RequestCache(1)
    else:
        cache = cache_state
        print(cache_state)
    
    buyerdf = pd.read_csv(bfile)
    b_schema = tuple(buyerdf.columns.tolist())
    aug_plan = cache.read_el(b_schema)
    print("Cache count before: {}".format(cache.cnt))
    if aug_plan != None:
        print("Number of augmentations to make: {}".format(len(aug_plan)))
        best_r2 = 0.0
        use_aug = True
        for jk, attrs in aug_plan:
            s_matches = [s[0] for s in sellers if jk == s[1]]
            msk = split_mask(len(buyerdf)) < 0.8
            buyer_train = buyerdf[msk].copy()
            buyer_test = buyerdf[~msk].copy()
        
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
            
            for j, s_match in enumerate(s_matches):
                if j > 0:
                    break
                # find the attributes and r2 of augmenting
                cur_atts, final_r2 = select_features(buyer_train_data, buyer_test_data, s_match, jk, 4, y)
          
                #this is a check to see if our r2 is really improving.
                #if we find out it's not, then this augmentation plan is no good, and we need to move to 
                if final_r2 > best_r2:
                  best_r2 = final_r2
                
                buyer_train_data.absorb(s_match, jk, cur_atts + [buyer_train_data.name + ":" + y])
                buyer_test_data.absorb(s_match, jk, cur_atts + [buyer_train_data.name + ":" + y])
        
        #once we've executed the plan, check the r2
        if final_r2 <= 0:
            use_aug = False
    
        if not use_aug:
            print("Augmentation plan failed--finding optimal plan manually")
            opt_aug = find_optimal_plan(bfile)
            print("Cache count just before: {}".format(cache.cnt))
            cache.add_el(b_schema, opt_aug)
            print("Cache count immediately after: {}".format(cache.cnt))
    else:
        opt_aug = find_optimal_plan(bfile)
        cache.add_el(b_schema, opt_aug)
        print("Cache count first time after: {}".format(cache.cnt))
    
    return cache
        

num_reps = 2
# start = time.time()
# for i in range(num_reps):
#     find_optimal_plan('gender.csv')
#     #find_optimal_plan("NH_SurveySummary_Mar2022.csv", enc=True)
# end = time.time()
# tot_runtime = end - start
# print("original gender runtime: {}".format(tot_runtime))

tot_ctime = 0.0
for i in range(num_reps):
    if i == 0:
        c_start = time.time()
        new_state = cache_optimal_plan('gender.csv')
        c_end = time.time()
        print("Before-cache runtime: {}".format(c_end - c_start))
        tot_ctime += c_end - c_start
        pkl.dump(new_state, open("cache_state.pkl", "wb+"))
    else:
        #we read the file in here so we don't hav
        with open('cache_state.pkl', 'rb') as f:
            cache_state = pkl.load(f)
        c_start = time.time()
        cache_optimal_plan('gender.csv', cache_state)
        c_end = time.time()
        print("Cached runtime: {}".format(c_end - c_start))
        tot_ctime += c_end - c_start
print("cache gender runtime: {}".format(tot_ctime))


