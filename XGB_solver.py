import numpy as np
import pandas as pd
import common
import datetime
import xgboost as xgb 
from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

def read_from_file(file_name, chunk_size=500000):
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

def test():
    iris = load_iris()  
    xgb_model = xgb.XGBRegressor(n_estimators=300000, max_depth=2)
    xgb_model.fit(iris.data[:120],iris.target[:120])

    predict = xgb_model.predict(iris.data[:120])
    print mean_squared_error(iris.target[:120], predict)

    pred = xgb_model.predict(iris.data[120:])
    print mean_squared_error(iris.target[120:], pred)

def merge_and_select_ad(ori_train_df):
    #ad.csv
    processd_ad_df = read_from_file(common.PROCESSED_AD_CSV)
    selected_features_ad = ['creativeID','adID','camgaignID','advertiserID','appID','appPlatform']
    selected_ad_df = processd_ad_df[selected_features_ad]
    df = pd.merge(ori_train_df, selected_ad_df, how='left', on='creativeID')

    #app_categories.csv
    processd_app_df = read_from_file(common.PROCESSED_APP_CATEGORIES_CSV)
    selected_features_app = ['appID','appCategoryFirst','appCategorySecond']
    selected_app_df = processd_app_df[selected_features_app]
    df = pd.merge(df, selected_app_df, how='left', on='appID')

    return df

def merge_and_select_user(ori_train_df):
    #user.csv
    processd_user_df = read_from_file(common.PROCESSED_USER_CSV)
    selected_features_user = ['userID','age','gender','education','marriageStatus','haveBaby','hometownProvID','hometownCityID','residenceProvID','residenceCityID']
    selected_user_df = processd_user_df[selected_features_user]
    df = pd.merge(ori_train_df, selected_user_df, how='left', on='userID')

    #user_app_actions.csv
    '''processd_actions_df = read_from_file(common.PROCESSED_APP_CATEGORIES_CSV)
    selected_features_app = ['appID','appCategory','appCategoryFirst','appCategorySecond']
    selected_app_df = processd_app_df[selected_features_app]
    df = pd.merge(df, selected_app_df, how='left', on='appID')'''

    #user_installedapps.csv
    processd_installed_df = read_from_file(common.PROCESSED_USER_INSTALLEDAPPS_CSV)
    df = pd.merge(df, processd_installed_df, how='left', on='userID')

    return df

def merge_and_select_other(ori_train_df):
    #position.csv
    processd_pos_df = read_from_file(common.PROCESSED_POSITION_CSV)
    selected_features_pos = ['positionID','sitesetID','positionType']
    selected_pos_df = processd_pos_df[selected_features_pos]
    df = pd.merge(ori_train_df, selected_pos_df, how='left', on='positionID')

    return df

def test_merge_appcount(file_name):
    ori_df = read_from_file(file_name)
    ori_df.drop(['clickTime', 'clickTimeDay', 'clickTimeHour', 'clickTimeMinu'], axis=1, inplace=True)

    processd_installed_df = read_from_file(common.ORIGIN_USER_INSTALLEDAPPS_CSV)
    tmp_seri = processd_installed_df['userID'].value_counts().sort_index()
    tmp_df = pd.DataFrame({'userID':list(tmp_seri.index), 'appCount':list(tmp_seri.values)})
    df = pd.merge(ori_df, tmp_df, how='left', on='userID')
    print df.isnull().sum()

def fix_missing_appcounts(df, model):
    app_df = df[['appCount','age','gender','education','marriageStatus','haveBaby']]
    unknown_app = app_df[app_df.appCount.isnull()].as_matrix()
    predicted_app = model.predict(unknown_app[:, 1:])
    df.loc[ (df.appCount.isnull()), 'appCount' ] = predicted_app
    return df

def train_model_for_appcounts(df):
    app_df = df[['appCount','age','gender','education','marriageStatus','haveBaby']]
    known_app = app_df[app_df.appCount.notnull()].as_matrix()
    unknown_app = app_df[app_df.appCount.isnull()].as_matrix()
    y = known_app[:, 0]
    X = known_app[:, 1:]

    print 'Train Xgboost Model(For Missing AppCount)...'
    start_time  = datetime.datetime.now()
    xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=3)
    xgb_reg.fit(X, y)
    end_time = datetime.datetime.now()
    print 'Training Done..., Time Cost: %d' % ((end_time - start_time).seconds)

    predicted_app = xgb_reg.predict(unknown_app[:, 1:])
    df.loc[ (df.appCount.isnull()), 'appCount' ] = predicted_app 

    return df, xgb_reg

def merge_features_to_use(file_name):
    ori_df = read_from_file(file_name)
    ori_df.drop(['clickTime', 'clickTimeDay', 'clickTimeHour', 'clickTimeMinu'], axis=1, inplace=True)
    print "Merge And Select Ad Data..."
    ori_df = merge_and_select_ad(ori_df)
    print "Done"
    print "Merge And Select User Data..."
    ori_df = merge_and_select_user(ori_df)
    print "Done"
    print "Merge And Select Other Data..."
    ori_df = merge_and_select_other(ori_df)
    print "Done"
    ori_df.drop(['adID', 'camgaignID', 'advertiserID', 'appID', 'creativeID', 'userID', 'positionID', 'hometownProvID', 'hometownCityID', 'residenceProvID', 'residenceCityID'], axis=1, inplace=True)
    return ori_df

def generate_XGB_model(train_df):
    train_df.drop(['conversionTime'], axis=1, inplace=True)
    print 'Train And Fix Missing App Count Value...'
    train_df, xgb_appcount = train_model_for_appcounts(train_df)
    joblib.dump(xgb_appcount, 'XGB_missing.model')
    print 'Done'
    print train_df.info()
    print train_df.describe()
    print train_df.isnull().sum()
    train_np = train_df.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]
    print 'Train Xgboost Model...'
    start_time  = datetime.datetime.now()
    xbg_clf = xgb.XGBRegressor(n_estimators=100, max_depth=6)
    xbg_clf.fit(X,y)
    end_time = datetime.datetime.now()
    print 'Training Done..., Time Cost: %d' % ((end_time - start_time).seconds)
    model_df = pd.DataFrame({'columns':list(train_df.columns)[1:], 'values':xbg_clf.feature_importances_})
    print model_df
    return xbg_clf

def use_model_to_predict(test_df, model):
    test_df.drop(['label'], axis=1, inplace=True)
    print 'Fix Missing App Count Value...'
    model_miss = joblib.load('XGB_missing.model')
    test_df = fix_missing_appcounts(test_df, model_miss)
    print 'Done'
    print test_df.info()
    print test_df.describe()
    print test_df.isnull().sum()
    test_np = test_df.as_matrix()
    X = test_np[:, 1:]
    print 'Use Model To Predict...'
    predicts = model.predict(X)
    result = pd.DataFrame({'instanceID':test_df['instanceID'].as_matrix(), 'prob':predicts})
    #print predicts#, predicts.min(axis=0), predicts.max(axis=0), predicts.sum(axis=1)
    return result

def filter_some_feature(train_df):
    df = train_df[train_df.connectionType == 1]
    return df

def train_to_predict(train_file_name, test_file_name, out_put):
    train_df = merge_features_to_use(train_file_name)
    XGB_clf = generate_XGB_model(train_df)
    test_df = merge_features_to_use(test_file_name)
    result = use_model_to_predict(test_df, XGB_clf)
    result.to_csv(out_put, index=False)
    print 'Save Model...'
    joblib.dump(XGB_clf, 'XGB.model')

'''def read_model_to_predict(model, test_file_name, out_put):
    clf = joblib.load(model)
    result = use_model_to_predict(test_file_name, clf)
    result.to_csv(out_put, index=False)'''

if __name__ == '__main__':
    train_to_predict(common.PROCESSED_TRAIN_CSV, common.PROCESSED_TEST_CSV, common.SUBMISSION_CSV)
    #merge_features_to_use(common.PROCESSED_TRAIN_CSV)
    #merge_features_to_use(common.PROCESSED_TEST_CSV)
    #test_merge_appcount(common.PROCESSED_TRAIN_CSV)