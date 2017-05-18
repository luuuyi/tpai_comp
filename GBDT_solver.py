import numpy as np
import pandas as pd
import common
import datetime
import pprint
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

def test():
    iris = load_iris()  
    #print iris
    #print iris['target'].shape  
    gbdt=GradientBoostingRegressor(n_estimators=1000, max_depth=4) 
    gbdt.fit(iris.data[:120],iris.target[:120])

    #Save GBDT Model
    joblib.dump(gbdt, 'GBDT.model') 

    predict = gbdt.predict(iris.data[:120])
    total_err = 0
    for i in range(len(predict)):
        print predict[i],iris.target[i]
        err = predict[i] - iris.target[i]
        total_err += err * err
    print 'Training Error: %f' % (total_err / len(predict))

    pred = gbdt.predict(iris.data[120:])
    error = 0
    for i in range(len(pred)):
        print pred[i],iris.target[i+120]
        err = pred[i] - iris.target[i+120]
        error += err * err
    print 'Test Error: %f' % (error / len(pred))

def test_model_select_by_param():
    iris = load_iris()    
    gbdt=GradientBoostingRegressor() 
    parameters = {'n_estimators': [1000, 5000], 'max_depth':[3,4]}
    grid_search = GridSearchCV(estimator=gbdt, param_grid=parameters, cv=10, n_jobs=-1)
    print("parameters:")
    pprint.pprint(parameters)
    grid_search.fit(iris.data[:150],iris.target[:150])
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters=grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def generate_GBDT_model(file_name):
    train_df = read_from_file(file_name)
    #featrue 18
    selected_train_df = train_df.filter(regex='label|creativeID|positionID|connectionType|telecomsOperator|adID|camgaignID|advertiserID|appID|appPlatform|sitesetID|positionType|age|gender|education|marriageStatus|haveBaby|hometown|residence')
    train_np = selected_train_df.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]
    print 'Train Gradient Boosting Regression Model...'
    start_time  = datetime.datetime.now()
    gbdt = GradientBoostingRegressor(n_estimators=10000, max_depth=18) #, class_weight='balanced')
    gbdt.fit(X,y)
    end_time = datetime.datetime.now()
    print 'Training Done..., Time Cost: '
    print (end_time - start_time).seconds

    print 'Save Model...'
    joblib.dump(gbdt, 'GBDT.model')
    return gbdt

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
    GBDT_clf = generate_RF_model(train_file_name)
    result = use_model_to_predict(test_file_name, GBDT_clf)
    result.to_csv(out_put, index=False)

def read_model_to_predict(model, test_file_name, out_put):
    clf = joblib.load(model)
    result = use_model_to_predict(test_file_name, clf)
    result.to_csv(out_put, index=False)

if __name__ == '__main__':
    test()
    #iris = load_iris() 
    #print iris, len(iris.target)