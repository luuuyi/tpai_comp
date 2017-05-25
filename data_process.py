import numpy as np
import pandas as pd
import common
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.preprocessing as preprocessing

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

def process_ad_file(src_file_name,dst_file_name):
    origin_ad_df = read_from_file(src_file_name)
    dummy_app_platform = pd.get_dummies(origin_ad_df['appPlatform'], prefix='appPlatform')
    processd_ad_df = pd.concat([origin_ad_df,dummy_app_platform],axis=1)
    
    processd_ad_df.to_csv(dst_file_name, index=False)

def process_position_file(src_file_name,dst_file_name):
    origin_pos_df = read_from_file(src_file_name)
    dummy_siteset = pd.get_dummies(origin_pos_df['sitesetID'], prefix='sitesetID')
    dummy_pos_type = pd.get_dummies(origin_pos_df['positionType'], prefix='positionType')
    processd_pos_df = pd.concat([origin_pos_df,dummy_siteset,dummy_pos_type], axis=1)

    processd_pos_df.to_csv(dst_file_name, index=False)

def process_user_file(src_file_name,dst_file_name):
    origin_user_df = read_from_file(src_file_name)
    origin_user_df['hometownProvID'] = origin_user_df['hometown'].as_matrix() / 100
    origin_user_df['hometownCityID'] = origin_user_df['hometown'].as_matrix() % 100
    origin_user_df['residenceProvID'] = origin_user_df['residence'].as_matrix() / 100
    origin_user_df['residenceCityID'] = origin_user_df['residence'].as_matrix() % 100
    dummy_gender = pd.get_dummies(origin_user_df['gender'], prefix='gender')
    dummy_education = pd.get_dummies(origin_user_df['education'], prefix='education')
    dummy_marr_status = pd.get_dummies(origin_user_df['marriageStatus'], prefix='marriageStatus')
    dummy_have_baby = pd.get_dummies(origin_user_df['haveBaby'], prefix='haveBaby')
    processd_user_df = pd.concat([origin_user_df, dummy_gender, dummy_education, dummy_marr_status, dummy_have_baby], axis=1)

    processd_user_df.to_csv(dst_file_name, index=False)

def process_app_category_file(src_file_name,dst_file_name):
    origin_app_cate_df = read_from_file(src_file_name)
    origin_app_cate_df['appCategoryFirst'] = origin_app_cate_df['appCategory'].as_matrix() / 100
    origin_app_cate_df['appCategorySecond'] = origin_app_cate_df['appCategory'].as_matrix() % 100

    origin_app_cate_df.to_csv(dst_file_name, index=False)

def process_user_app_actions_file(src_file_name,dst_file_name):
    origin_user_app_actions_df = read_from_file(src_file_name)
    origin_user_app_actions_df['installTimeDay'] = origin_user_app_actions_df['installTime'].as_matrix() / 10000
    origin_user_app_actions_df['installTimeHour'] = (origin_user_app_actions_df['installTime'].as_matrix() % 10000) / 100
    origin_user_app_actions_df['installTimeMinu'] = origin_user_app_actions_df['installTime'].as_matrix() % 100

    origin_user_app_actions_df.to_csv(dst_file_name, index=False)

def process_user_installed_app_file(src_file_name,dst_file_name):
    origin_user_installed_app_df = read_from_file(src_file_name)
    tmp_seri = origin_user_installed_app_df['userID'].value_counts().sort_index()
    tmp_df = pd.DataFrame({'userID':list(tmp_seri.index), 'appCount':list(tmp_seri.values)})

    tmp_df.to_csv(dst_file_name, index=False)

def process_train_file(src_file_name, dst_file_name,):
    ori_train_df = read_from_file(src_file_name)
    ori_train_df['clickTimeDay'] = ori_train_df['clickTime'].as_matrix() / 10000
    ori_train_df['clickTimeHour'] = (ori_train_df['clickTime'].as_matrix() % 10000) / 100
    ori_train_df['clickTimeMinu'] = ori_train_df['clickTime'].as_matrix() % 100
    
    ori_train_df.to_csv(dst_file_name, index=False)

def process_test_file(src_file_name, dst_file_name):
    ori_test_df = read_from_file(src_file_name)
    ori_test_df['clickTimeDay'] = ori_test_df['clickTime'].as_matrix() / 10000
    ori_test_df['clickTimeHour'] = (ori_test_df['clickTime'].as_matrix() % 10000) / 100
    ori_test_df['clickTimeMinu'] = ori_test_df['clickTime'].as_matrix() % 100

    ori_test_df.to_csv(dst_file_name,index=False)

if __name__ == '__main__':
    process_ad_file(common.ORIGIN_AD_CSV, common.PROCESSED_AD_CSV)
    process_position_file(common.ORIGIN_POSITION_CSV, common.PROCESSED_POSITION_CSV)
    process_user_file(common.ORIGIN_USER_CSV, common.PROCESSED_USER_CSV)
    process_app_category_file(common.ORIGIN_APP_CATEGORIES_CSV, common.PROCESSED_APP_CATEGORIES_CSV)
    process_user_app_actions_file(common.ORIGIN_USER_APP_ACTIONS_CSV, common.PROCESSED_USER_APP_ACTIONS_CSV)
    process_user_installed_app_file(common.ORIGIN_USER_INSTALLEDAPPS_CSV, common.PROCESSED_USER_INSTALLEDAPPS_CSV)
    process_train_file(common.ORIGIN_TRAIN_CSV, common.PROCESSED_TRAIN_CSV)
    process_test_file(common.ORIGIN_TEST_CSV, common.PROCESSED_TEST_CSV)