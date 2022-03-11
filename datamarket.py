import pandas as pd
import numpy as np
import random
import statistics
import math
from dateutil import parser
from datetime import datetime
from functools import reduce
import random
from random import choices
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import *
import autosklearn.regression
import sklearn.datasets
import sklearn.metrics
import autosklearn.regression


# Given a gram matrix semi-ring, normalize it
def normalize(cov):
    cols = []
    
    if isinstance(cov, pd.DataFrame):
        cols = cov.columns
    # this is for the final semiring, which is reduced to a single np array
    else:
        cols = list(cov.axes[0])
        
    for col in cols:
        if col != 'cov:c':
            cov[col] = cov[col]/cov['cov:c']
    
    if isinstance(cov, pd.DataFrame):
        cov.rename(columns={'cov:c':'cov:c_old'}, inplace=True)
        
    cov['cov:c'] = 1
    return cov

class agg_dataset:
    def set_meta(self, data, X, dimensions, name):
        self.data = data
        self.dimensions = dimensions
        self.X =  X
        self.name = name
        self.datasets = set()
        self.datasets.add(self.name)
    
    # process target variable
    # unlike features, we want to *remove* all tuples whose target variable is null
    def process_target(self, y):
        # don't impute y
        self.to_numeric(y)
        self.remove_null(y)
    
    # dimensions is a list of list of attributes 
    # dedup_dimensions get the set of all attributes in dimensions
    def get_deduplicated_dimension(self):
        dedup_dimensions = set()
        for d in self.dimensions:
            if isinstance(d, list):
                dedup_dimensions.update(d)
            else:
                dedup_dimensions.add(d)
        self.dedup_dimensions = list(dedup_dimensions)
    
    # remove redundant columns not for ml
    def remove_redundant_columns(self):
        # project out attributes except x, y, dim
        self.data.drop(self.data.columns.difference(self.X  + self.dedup_dimensions), axis=1, inplace=True)
        # Note that the commented codes is not efficient, as it creates a view
        # self.data = self.data[self.X  + self.dedup_dimensions]
    
    # load data (in the format of dataframe)
    # user provides dimensions (these dimensions will be pre-aggregated)
    # name should be unique across datasets
    def load(self, data, X, dimensions, name):
        self.data = data
        self.dimensions = dimensions
        self.X = X
        self.name = name
        
        self.get_deduplicated_dimension()
        
        # all the datasets it contains
        # this is useful for buyer when it has augmented by many seller datasets
        self.datasets = set()
        self.datasets.add(self.name)
    
    # compute the semi-ring aggregation for each dimension
    # norm is "whether the semi-ring aggregation is normalized"
    def compute_agg(self, norm = False):
        # build semi-ring structure
        self.lift(self.name, self.X)
        
        self.agg_dimensions = dict()
        
        for d in self.dimensions:
            if isinstance(d, list):
                self.agg_dimensions[tuple(d)] = self.data[list(filter(lambda col: col.startswith("cov:"), self.data.columns)) + d].groupby(d).sum()
            else:
                self.agg_dimensions[d] = self.data[list(filter(lambda col: col.startswith("cov:"), self.data.columns)) + [d]].groupby(d).sum()
            
        if norm:
            for d in self.agg_dimensions.keys():
                self.agg_dimensions[d] = normalize(self.agg_dimensions[d])
            
        self.covariance = normalize(self.data[list(filter(lambda col: col.startswith("cov:"), self.data.columns))].sum())
        
        self.X = [self.name + ':' + x for x in self.X]
    
    # iterate all attributes and find candidate features
    def find_features(self):
        atts = []
        for att in self.data.columns:
            if att in self.dedup_dimensions or att in self.X:
                continue
 
            if self.is_features(att, 0.3, 5):
                atts.append(att)
                self.impute_mean(att)
                
        self.X += atts
    
    # try to parse the column into feature and decide whether it is a feature
    # decided by two criterions: 1. percentage of missing value is below missing_threshold 2. # of distinct values is larger than distinct_threshold
    def is_features(self, att, missing_threshold, distinct_threshold):
        # parse attribute to numeric
        self.to_numeric(att)
        
        col = self.data[att]
        missing = sum(np.isnan(col))/len(self.data)
        distinct = len(col.unique())
        
        if missing < missing_threshold and distinct > distinct_threshold:
            return True
        else:
            return False
    
    def to_numeric_and_impute_all(self):
        for att in self.X:
            self.to_numeric(att)
            self.impute_mean(att)
    
    # this is the function to transform an attribute to number
    def to_numeric(self, att):
        # parse attribute to numeric
        self.data[att] = pd.to_numeric(self.data[att],errors="coerce")
    
    def impute_mean(self, att):
        mean_value=self.data[att].mean()
        self.data[att].fillna(value=mean_value, inplace=True)
    
    def remove_null(self, att):
        self.data.dropna(subset=[att],inplace=True)
    
    def standardize_all(self):
        for att in self.X:
            self.standardize(att)
    
    def standardize(self, att):
        self.data[att] = (self.data[att] - self.data[att].mean())/(self.data[att].std())
        
    def transform(self):
        for att in self.X:
            self.transform(att)
    
    # for now, only square
    # please comment out codes if you want more transformations
    def transform(self, att):
#         self.data["log" + att] = np.log(self.data[att])
        self.data["sq" + att] = np.square(self.data[att])
#         self.data["cbr" + att] = np.cbrt(self.data[att])
#         atts.append("log" + att)
        self.X.append("sq" + att)
#         atts.append("cbr" + att)
    
    # build gram matrix semi-ring
    def lift(self, tablename, attributes):
        self.data['cov:c'] = 1

        for i in range(len(attributes)):
            for j in range(i, len(attributes)):
                self.data['cov:Q:' + tablename + ":" + attributes[i] + ","+ tablename + ":" + attributes[j]] = self.data[attributes[i]] * self.data[attributes[j]]

        for attribute in attributes:
            self.data= self.data.rename(columns = {attribute:'cov:s:' + tablename + ":" + attribute})
    
    def absorb(self, agg_data, dimension, attrs):
        
        if agg_data.name in self.datasets:
            print("already absorbed this data")
            return
            
        self.data = connect(self, agg_data, dimension, True, attrs)
        
        for d in self.dimensions:
            if isinstance(d, list):
                self.agg_dimensions[tuple(d)] = self.data[list(filter(lambda col: col.startswith("cov:"), self.data.columns)) + d].groupby(d).sum()
            else:
                self.agg_dimensions[d] = self.data[list(filter(lambda col: col.startswith("cov:"), self.data.columns)) + [d]].groupby(d).sum()
            
        self.covariance = normalize(self.data[list(filter(lambda col: col.startswith("cov:"), self.data.columns))].sum())
        
        self.X = self.X + attrs
        self.datasets.add(agg_data.name)
    
# return the coefficients of features and a constant 
def linear_regression(cov_matrix, features, result):
    a = np.empty([len(features) + 1, len(features) + 1])
    b = np.empty(len(features) + 1)
    
    for i in range(len(features)):
        for j in range(len(features)):
            if 'cov:Q:' + features[i] + ","+ features[j] in cov_matrix:
                a[i][j] = cov_matrix['cov:Q:' + features[i] + ","+ features[j]]
            else:
                a[i][j] = cov_matrix['cov:Q:' + features[j] + ","+ features[i]]
    
    for i in range(len(features)):
        a[i][len(features)] = cov_matrix['cov:s:' + features[i]]
        a[len(features)][i] = cov_matrix['cov:s:' + features[i]]
        if 'cov:Q:' + result + "," + features[i] in cov_matrix:
            b[i] = cov_matrix['cov:Q:' + result + "," + features[i]]
        else:
            b[i] = cov_matrix['cov:Q:' + features[i] + "," + result]
    
    b[len(features)] = cov_matrix['cov:s:' + result]
    
    a[len(features)][len(features)] = cov_matrix['cov:c']
#     print(a,b)
    return np.linalg.solve(a, b)

def square_error(cov_matrix, features, result, parameter):
    se = cov_matrix['cov:Q:'  + result + "," + result]
    
#     print(se)
    for i in range(len(features)):
        for j in range(len(features)):
            if 'cov:Q:'  + features[i] + "," + features[j] in cov_matrix:
                se += parameter[i] * parameter[j] * cov_matrix['cov:Q:'  + features[i] + "," + features[j]]
            else:    
                se += parameter[j] * parameter[i] * cov_matrix['cov:Q:'  + features[j] + "," + features[i]]
#             print(se, 'cov:Q:'  + features[i] + "," + features[j])
   
    
    for i in range(len(features)):
        se += 2 * parameter[i] * parameter[-1] * cov_matrix['cov:s:'  + features[i]]
        if 'cov:Q:' + result + "," + features[i] in cov_matrix:
            se -= 2 * parameter[i] *  cov_matrix['cov:Q:' + result + "," + features[i]]
        else:
            se -= 2 * parameter[i] *  cov_matrix['cov:Q:' + features[i] + "," + result]
    
#     print(se)
    se -= 2 * parameter[-1] * cov_matrix['cov:s:'  + result]
    se += cov_matrix['cov:c'] * parameter[-1] * parameter[-1]

    return se

def total_sum_of_square(cov_matrix, result):
    return cov_matrix['cov:Q:'  + result + "," + result] - cov_matrix['cov:s:'  + result] * cov_matrix['cov:s:'  + result] / cov_matrix['cov:c']

def mean_squared_error(cov_matrix, features, result, parameter):
    return square_error(cov_matrix, features, result, parameter)/cov_matrix['cov:c']


def r2(cov_matrix, features, result, parameter):
    result =  1 - square_error(cov_matrix, features, result, parameter)/total_sum_of_square(cov_matrix, result)
    if result > 2:
        # overflow
        return -1
    return result

def adjusted_r2(cov_matrix, features, result, parameter):
    return 1 - (cov_matrix['cov:c']-1)*(1 - r2(cov_matrix, features, result, parameter))/(cov_matrix['cov:c'] - len(parameter) - 1)

# left_inp is whether we join of index of aggdata1 (index has good performance. however no index during absorption)
# specify right_attrs if for right table, only part of attributes are involved
def connect(aggdata1, aggdata2, dimension, left_inp = False, right_attrs = []):
    
    if isinstance(dimension, list):
        dimension = tuple(dimension)
    
    if left_inp:
        agg1 = aggdata1.data
    else:
        agg1 = aggdata1.agg_dimensions[dimension]
        
    agg2 = aggdata2.agg_dimensions[dimension]
    
    left_attributes = aggdata1.X
    left_tablename = aggdata1.name
    right_attributes = aggdata2.X
    right_tablename = aggdata2.name
    
    # if you only want to augment part of attributes (that are predictive)
    if len(right_attrs) > 0:
        kept_cols = []
        for col in agg2.columns:
            # cov has multiple names
            names = col[6:].split(",")
            match = True
            for name in names:
                if name not in right_attrs:
                    match = False
            if match:
                kept_cols.append(col)
        agg2 = agg2[kept_cols + ['cov:c']]
        right_attributes = right_attrs
    
    # wheter join on index
    if left_inp:
        join = pd.merge(agg1.set_index(dimension), agg2, how='left', left_index=True, right_index=True)
    else:
        join = pd.merge(agg1, agg2, how='left', left_index=True, right_index=True)
#         join = pd.merge(agg1, agg2, how='inner', left_index=True, right_index=True)
    join = join.drop('cov:c_y', 1)
    join = join.rename(columns = {'cov:c_x':'cov:c'})
    
    right_cov = aggdata2.covariance
    
    # fill in nan
    for att2 in right_attributes:
        join['cov:s:' + att2].fillna(value=right_cov['cov:s:' + att2], inplace=True)
        join['cov:s:' + att2] *= join['cov:c']
    
    for i in range(len(right_attributes)):
        for j in range(i, len(right_attributes)):
            if 'cov:Q:' + right_attributes[i] + "," + right_attributes[j] in join:
                join['cov:Q:' + right_attributes[i] + "," + right_attributes[j]].fillna(value=right_cov['cov:Q:' + right_attributes[i] + "," + right_attributes[j]], inplace=True)
                join['cov:Q:' + right_attributes[i] + "," + right_attributes[j]] *= join['cov:c']
            else:
                join['cov:Q:' + right_attributes[j] + "," + right_attributes[i]].fillna(value=right_cov['cov:Q:' + right_attributes[j] + "," + right_attributes[i]], inplace=True)
                join['cov:Q:' + right_attributes[j] + "," + right_attributes[i]] *= join['cov:c']
            
    
    
    for att1 in left_attributes:
        for att2 in right_attributes:
            if 'cov:Q:' + att1 + "," + att2 in join:
                join['cov:Q:' + att1 + "," + att2] = join['cov:s:' + att1] * join['cov:s:' + att2]/join['cov:c']
            else:
                join['cov:Q:' + att2 + "," + att1] = join['cov:s:' + att2] * join['cov:s:' + att1]/join['cov:c']
    
    
    return join

def select_features(train, test, seller, dimension, k, target):
    join_test = connect(test, seller, dimension)
    join_train = connect(train, seller, dimension)

    cur_atts = []
    join_train_cov = join_train.sum()
    join_test_cov = join_test.sum()
    final_r2 = 0
    
    for i in range(k):
        best_r2 = 0
        best_att = -1
        for att in train.X + seller.X:
            if att in cur_atts or att == train.name + ":" + target:
                continue
            # maybe singular
            try:
                parameter = linear_regression(join_train_cov, cur_atts + [att], train.name + ":" + target)
            except:
                continue
            cur_r2 = r2(join_test_cov, cur_atts + [att], train.name + ":" + target, parameter)
    #         print(cur_r2, att)
            if cur_r2 > best_r2:
                best_r2 = cur_r2
                best_att = att
        if best_r2 == 0 or best_r2 < final_r2:
            break
        cur_atts = cur_atts + [best_att]
        final_r2 = best_r2
#         print(i, best_r2, cur_atts)
    return cur_atts, final_r2


# return the coefficients of features and a constant 
def ridge_linear_regression(cov_matrix, features, result, alpha):
    a = np.empty([len(features) + 1, len(features) + 1])
    b = np.empty(len(features) + 1)
    
    for i in range(len(features)):
        for j in range(len(features)):
            if 'cov:Q:' + features[i] + ","+ features[j] in cov_matrix:
                a[i][j] = cov_matrix['cov:Q:' + features[i] + ","+ features[j]]
            else:
                a[i][j] = cov_matrix['cov:Q:' + features[j] + ","+ features[i]]
        if i == j:
            a[i][i] += alpha
    
    for i in range(len(features)):
        a[i][len(features)] = cov_matrix['cov:s:' + features[i]]
        a[len(features)][i] = cov_matrix['cov:s:' + features[i]]
        if 'cov:Q:' + result + "," + features[i] in cov_matrix:
            b[i] = cov_matrix['cov:Q:' + result + "," + features[i]]
        else:
            b[i] = cov_matrix['cov:Q:' + features[i] + "," + result]
    
    b[len(features)] = cov_matrix['cov:s:' + result]
    
    a[len(features)][len(features)] = cov_matrix['cov:c']
#     print(a,b)
    return np.linalg.solve(a, b)

def split_mask(length):
    return np.random.rand(length)

def cross_mask(length, classes):
    return np.random.choice(classes, length)