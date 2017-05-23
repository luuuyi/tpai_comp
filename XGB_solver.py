import numpy as np
import pandas as pd
import common
import datetime
import xgboost as xgb 
from sklearn.datasets import load_iris
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
    #print iris
    #print iris['target'].shape 
    '''train_data = xgb.DMatrix(iris.data[:100], label=iris.target[:100]) 
    val_data = xgb.DMatrix(iris.data[100:120], label=iris.target[100:120]) 
    watch_list = [(train_data, 'train'), (val_data, 'val')]
    test_data = xgb.DMatrix(iris.data[120:], label=iris.target[120:]) '''

    xgb_model = xgb.XGBRegressor(n_estimators=300000, max_depth=2)
    xgb_model.fit(iris.data[:120],iris.target[:120])

    #Save GBDT Model
    #joblib.dump(gbdt, 'GBDT.model') 

    predict = xgb_model.predict(iris.data[:120])
    print mean_squared_error(iris.target[:120], predict)
    '''total_err = 0
    for i in range(len(predict)):
        print predict[i],iris.target[i]
        err = predict[i] - iris.target[i]
        total_err += err * err
    print 'Training Error: %f' % (total_err / len(predict))'''

    pred = xgb_model.predict(iris.data[120:])
    print mean_squared_error(iris.target[120:], pred)
    '''error = 0
    for i in range(len(pred)):
        print pred[i],iris.target[i+120]
        err = pred[i] - iris.target[i+120]
        error += err * err
    print 'Test Error: %f' % (error / len(pred))'''

def merge_and_select_ad(ori_train_df):
    #ad.csv
    processd_ad_df = read_from_file(common.PROCESSED_AD_CSV)
    selected_features_ad = ['creativeID','adID','camgaignID','advertiserID','appID','appPlatform','appPlatform_1','appPlatform_2']
    selected_ad_df = processd_ad_df[selected_features_ad]
    df = pd.merge(ori_train_df, selected_ad_df, how='left', on='creativeID')

    #app_categories.csv
    processd_app_df = read_from_file(common.PROCESSED_APP_CATEGORIES_CSV)
    selected_features_app = ['appID','appCategory','appCategoryFirst','appCategorySecond']
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
    processd_installed_df = read_from_file(common.ORIGIN_USER_INSTALLEDAPPS_CSV)
    tmp_seri = processd_installed_df['userID'].value_counts().sort_index()
    tmp_df = pd.DataFrame({'userID':list(tmp_seri.index), 'appCount':list(tmp_seri.values)})
    tmp_df['canCount'] = 1
    df = pd.merge(df, tmp_df, how='left', on='userID')
    df = df.fillna(0)

    return df

def merge_and_select_other(ori_train_df):
    #position.csv
    processd_pos_df = read_from_file(common.PROCESSED_POSITION_CSV)
    selected_features_pos = ['positionID','sitesetID','positionType']
    selected_pos_df = processd_pos_df[selected_features_pos]
    df = pd.merge(ori_train_df, selected_pos_df, how='left', on='positionID')

    return df

def merge_features_to_use(file_name):
    ori_df = read_from_file(file_name)
    print "Merge And Select Ad Data..."
    ori_df = merge_and_select_ad(ori_df)
    print "Done"
    print "Merge And Select User Data..."
    ori_df = merge_and_select_user(ori_df)
    print "Done"
    print "Merge And Select Other Data..."
    ori_df = merge_and_select_other(ori_df)
    print "Done"
    return ori_df

def generate_XGB_model(train_df):
    #featrue 18
    selected_train_df = train_df.filter(regex='label|creativeID|positionID|telecomsOperator|adID|camgaignID|advertiserID|appID|appPlatform|sitesetID|positionType|age|gender|education|marriageStatus|haveBaby|hometown|residence')
    train_np = selected_train_df.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]
    print 'Train Xgboost Model...'
    start_time  = datetime.datetime.now()
    xbg_clf = xgb.XGBRegressor(n_estimators=100, max_depth=6)
    xbg_clf.fit(X,y)
    end_time = datetime.datetime.now()
    print 'Training Done..., Time Cost: '
    print (end_time - start_time).seconds

    return xbg_clf

def use_model_to_predict(test_df, model):
    selected_test_df = test_df.filter(regex='creativeID|positionID|telecomsOperator|adID|camgaignID|advertiserID|appID|appPlatform|sitesetID|positionType|age|gender|education|marriageStatus|haveBaby|hometown|residence')
    test_np = selected_test_df.as_matrix()
    print 'Use Model To Predict...'
    #model_df =pd.DataFrame({'coef':model.coef_.T[:,0], 'columns':list(selected_test_df.columns)}) 
    #print model_df.describe()
    #print model_df.info()
    predicts = model.predict(test_np)
    result = pd.DataFrame({'instanceID':test_df['instanceID'].as_matrix(), 'prob':predicts})
    #print predicts#, predicts.min(axis=0), predicts.max(axis=0), predicts.sum(axis=1)
    return result

def filter_some_feature(train_df):
    df = train_df[train_df.connectionType == 1]
    return df

def train_to_predict(train_file_name, test_file_name, out_put):
    train_df = read_from_file(train_file_name)
    train_df = filter_some_feature(train_df)

    XGB_clf = generate_XGB_model(train_df)

    test_df = read_from_file(test_file_name)
    ret = pd.DataFrame({'instanceID':test_df['instanceID'].as_matrix()})
    test_df = filter_some_feature(test_df)

    result = use_model_to_predict(test_df, XGB_clf)
    ret = pd.merge(ret, result, how='left', on='instanceID')
    ret = ret.fillna(0)
    ret.to_csv(out_put, index=False)

'''def read_model_to_predict(model, test_file_name, out_put):
    clf = joblib.load(model)
    result = use_model_to_predict(test_file_name, clf)
    result.to_csv(out_put, index=False)'''

if __name__ == '__main__':
    #train_to_predict(common.PROCESSED_TRAIN_CSV, common.PROCESSED_TEST_CSV, common.SUBMISSION_CSV)
    merge_features_to_train(common.ORIGIN_TRAIN_CSV)