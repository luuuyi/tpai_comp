import numpy as np
import pandas as pd
import common
import datetime
import pprint
import xgboost as xgb 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import logging

def read_from_file(file_name, chunk_size=50000):
    reader = pd.read_csv(file_name, iterator=True)
    chunks = []
    mark = True
    while mark:
        try:
            df = reader.get_chunk(chunk_size)
            chunks.append(df)
        except:
            print "Iterator Stop..."
            mark = False
    df = pd.concat(chunks,ignore_index=True)
    return df

def xgb_model_select(file_name):  
    train_df = read_from_file(file_name)
    selected_train_df = train_df.filter(regex='label|creativeID|positionID|connectionType|telecomsOperator|adID|camgaignID|advertiserID|appID|appPlatform|sitesetID|positionType|age|gender|education|marriageStatus|haveBaby')
    train_np = selected_train_df.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]

    print 'Select Model...'
    start_time  = datetime.datetime.now()
    xgb_clf = xgb.XGBRegressor() 
    parameters = {'n_estimators': [120, 100, 140], 'max_depth':[3,5,7,9]}
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=10, n_jobs=-1)
    print("parameters:")
    pprint.pprint(parameters)
    grid_search.fit(X, y)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters=grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    end_time = datetime.datetime.now()
    print 'Select Done..., Time Cost: %d' % ((end_time - start_time).seconds)

def gbdt_select_model(file_name):
    train_df = read_from_file(file_name)
    #featrue 16
    selected_train_df = train_df.filter(regex='label|creativeID|positionID|connectionType|telecomsOperator|adID|camgaignID|advertiserID|appID|appPlatform|sitesetID|positionType|age|gender|education|marriageStatus|haveBaby')
    train_np = selected_train_df.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]

    print 'Select Model...'
    start_time  = datetime.datetime.now()
    gbdt = GradientBoostingRegressor() 
    parameters = {'n_estimators': [100, 120], 'max_depth':[4, 5, 6]}
    grid_search = GridSearchCV(estimator=gbdt, param_grid=parameters, cv=10, n_jobs=-1)
    print("parameters:")
    pprint.pprint(parameters)
    grid_search.fit(X, y)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters=grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    end_time = datetime.datetime.now()
    print 'Select Done..., Time Cost: %d' % ((end_time - start_time).seconds)

if __name__ == '__main__':
    xgb_model_select(common.PROCESSED_TRAIN_CSV)