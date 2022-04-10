from datamarket import *
import json

# read buyer dataset
# gender 
buyer = pd.read_csv("gender.csv")
# ethnic 
# buyer = pd.read_csv("2013-2017_School_ELA_Results_-_Ethnic.csv")
# survery
# buyer = pd.read_csv("NH_SurveySummary_Mar2022.csv", encoding='cp1252')

# buyer = pd.read_csv("NH_CovidVaxProvider_20220320.csv", encoding='cp1252')

# train test split
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

# model performance before augmentation with linear regression
train_cov = buyer_train_data.covariance
test_cov = buyer_test_data.covariance
X = ['Number Tested']
y = 'Mean Scale Score'

# X = ['Total Number of Fire Safety Deficiencies']
# y = 'Total Number of Health Deficiencies'

# X = []
# y = 'Percent Vaccinated Residents'

parameter = linear_regression(train_cov, ["buyer:" + x for x in X], "buyer:" + y)
print("R2 score is:" + str(r2(test_cov, ["buyer:" + x for x in X], "buyer:" + y, parameter)))

categories = ['DBN', 'Year', "Grade", "Category"]
# categories = ["Federal Provider Number", 'Location', 'Processing Date']
# categories = ["Federal Provider Number", 'Provider State']
for cate in categories:
    buyer = buyer.astype({cate: 'category'})

buyer[y] = pd.to_numeric(buyer[y], errors='coerce')
buyer.dropna(subset=[y],inplace=True)
msk = split_mask(len(buyer)) < 0.8
buyer_train = buyer[msk].copy()
buyer_test = buyer[~msk].copy()

time_budget = 600

# model performance with autoML 
X_train = buyer_train[X + categories]
y_train = buyer_train[y]
X_test = buyer_test[X + categories]
y_test = buyer_test[y]

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=time_budget,
    per_run_time_limit=int(time_budget/3),
    memory_limit=6072
)
automl.fit(X_train, y_train, X_test, y_test, dataset_name='buyer')

now_n = datetime.now()
current_time = now_n.strftime("%H:%M:%S")
print("Current Time =", current_time)
automl.sprint_statistics()

train_predictions = automl.predict(X_train)
print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
test_predictions = automl.predict(X_test)
print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))
print(automl.leaderboard())
poT = automl.performance_over_time_
print(poT)
poT.plot(
    x='Timestamp',
    kind='line',
    legend=True,
    title='Auto-sklearn accuracy over time',
    grid=True,
)

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




