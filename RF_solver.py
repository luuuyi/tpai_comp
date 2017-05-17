import numpy as np
import pandas as pd
import common
import datetime
from sklearn.ensemble import RandomForestRegressor  
from sklearn.datasets import load_iris
from sklearn import cross_validation, metrics
from sklearn.externals import joblib


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

def test():
    iris=load_iris()  
    #print iris
    #print iris['target'].shape  
    rf=RandomForestRegressor() 
    rf.fit(iris.data[:150],iris.target[:150])
    predict = rf.predict(iris.data[:150])
    print predict
    print iris.target[:150]
    #metrics.roc_auc_score(predict, iris.target[:150])

def generate_RF_model(file_name):
    train_df = read_from_file(file_name)
    selected_train_df = train_df.filter(regex='label|creativeID|positionID|connectionType|telecomsOperator|adID|camgaignID|advertiserID|appID|appPlatform|sitesetID|positionType|age|gender|education|marriageStatus|haveBaby|hometown|residence')
    train_np = selected_train_df.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]
    print 'Train Random Forest Regression Model...'
    start_time  = datetime.datetime.now()
    rf = RandomForestRegressor(n_estimators=25, n_jobs=-1)#, class_weight='balanced')
    rf.fit(X,y)
    end_time = datetime.datetime.now()
    print 'Training Done..., Time Cost: '
    print (end_time-start_time).seconds

    print 'Save Model...'
    joblib.dump(rf, 'RF.model')
    return rf

def use_model_to_predict(test_file_name, model):
    test_df = read_from_file(test_file_name)
    selected_test_df = test_df.filter(regex='creativeID|positionID|connectionType|telecomsOperator|adID|camgaignID|advertiserID|appID|appPlatform|sitesetID|positionType|age|gender|education|marriageStatus|haveBaby|hometown|residence')
    test_np = selected_test_df.as_matrix()
    print 'Use Model To Predict...'
    #model_df =pd.DataFrame({'coef':model.coef_.T[:,0], 'columns':list(selected_test_df.columns)}) 
    #print model_df.describe()
    #print model_df.info()
    predicts = model.predict(test_np)
    result = pd.DataFrame({'instanceID':test_df['instanceID'].as_matrix(), 'prob':predicts})
    #print predicts#, predicts.min(axis=0), predicts.max(axis=0), predicts.sum(axis=1)
    return result

def train_to_predict(train_file_name, test_file_name, out_put):
    RF_clf = generate_RF_model(train_file_name)
    result = use_model_to_predict(test_file_name, RF_clf)
    result.to_csv(out_put, index=False)

def read_model_to_predict(model, test_file_name, out_put):
    clf = joblib.load(model)
    result = use_model_to_predict(test_file_name, clf)
    result.to_csv(out_put, index=False)

if __name__ == '__main__':
    #test()
    #train_to_predict(common.PROCESSED_TRAIN_CSV, common.PROCESSED_TEST_CSV, common.SUBMISSION_CSV)
    read_model_to_predict('./RF.model', common.PROCESSED_TEST_CSV, common.SUBMISSION_CSV)