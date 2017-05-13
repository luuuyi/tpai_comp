from sklearn import linear_model
import numpy as np
import pandas as pd
import common

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

def generate_LR_model(file_name):
    train_df = read_from_file(file_name)
    selected_train_df = train_df.filter(regex='label|connectionType|telecomsOperator|appPlatform_.*|sitesetID_.*|positionType_.*|gender_.*|education_.*')
    train_np = selected_train_df.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]
    print 'Train Logistic Regression Model...'
    clf = linear_model.LogisticRegression(penalty='l1',C=1.0,tol=1e-6)
    clf.fit(X,y)
    print 'Training Done...'
    return clf

def use_model_to_predict(test_file_name, model):
    test_df = read_from_file(test_file_name)
    selected_test_df = test_df.filter(regex='connectionType|telecomsOperator|appPlatform_.*|sitesetID_.*|positionType_.*|gender_.*|education_.*')
    test_np = selected_test_df.as_matrix()
    print 'Use Model To Predict...'
    predicts = model.predict(test_np)
    print predicts
    return predicts

def train_to_predict(train_file_name, test_file_name):
    LR_clf = generate_LR_model(train_file_name)
    predicts = use_model_to_predict(test_file_name, LR_clf)

if __name__ == '__main__':
    train_to_predict(common.PROCESSED_TRAIN_CSV, common.PROCESSED_TEST_CSV)