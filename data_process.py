import numpy as np
import pandas as pd
import common
import matplotlib.pyplot as plt
from sklearn import linear_model

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
    #print origin_ad_df.info()
    #print origin_ad_df.describe()
    dummy_app_platform = pd.get_dummies(origin_ad_df['appPlatform'], prefix='appPlatform')
    processd_ad_df = pd.concat([origin_ad_df,dummy_app_platform],axis=1)
    #print processd_ad_df.info()
    #print processd_ad_df.describe()

    processd_ad_df.to_csv(dst_file_name, index=False)

def process_position_file(src_file_name,dst_file_name):
    origin_pos_df = read_from_file(src_file_name)
    #print origin_pos_df.info()
    #print origin_pos_df.describe()
    dummy_siteset = pd.get_dummies(origin_pos_df['sitesetID'], prefix='sitesetID')
    dummy_pos_type = pd.get_dummies(origin_pos_df['positionType'], prefix='positionType')
    processd_pos_df = pd.concat([origin_pos_df,dummy_siteset,dummy_pos_type], axis=1)
    #print processd_pos_df.info()
    #print processd_pos_df.describe()

    processd_pos_df.to_csv(dst_file_name, index=False)

def process_user_file(src_file_name,dst_file_name):
    origin_user_df = read_from_file(src_file_name)
    #print origin_user_df.info()
    #print origin_user_df.describe()
    dummy_gender = pd.get_dummies(origin_user_df['gender'], prefix='gender')
    dummy_education = pd.get_dummies(origin_user_df['education'], prefix='education')
    dummy_marr_status = pd.get_dummies(origin_user_df['marriageStatus'], prefix='marriageStatus')
    dummy_have_baby = pd.get_dummies(origin_user_df['haveBaby'], prefix='haveBaby')
    processd_user_df = pd.concat([origin_user_df, dummy_gender, dummy_education, dummy_marr_status, dummy_have_baby], axis=1)
    #print processd_user_df.info()
    #print processd_user_df.describe()

    processd_user_df.to_csv(dst_file_name, index=False)

def process_train_file(src_file_name, dst_file_name, is_skip_regenerate=False):
    ori_train_df = read_from_file(src_file_name)
    #print ori_train_df.info()
    #print ori_train_df.describe()
    if not is_skip_regenerate:
        print "Process Ad File..."
        process_ad_file(common.ORIGIN_AD_CSV,common.PROCESSED_AD_CSV)
        print "Process Position File..."
        process_position_file(common.ORIGIN_POSITION_CSV, common.PROCESSED_POSITION_CSV)
        print "Process User File..."
        process_user_file(common.ORIGIN_USER_CSV, common.PROCESSED_USER_CSV)
    
    #merge data frame
    print "Merge Data..."
    processed_ad_df = read_from_file(common.PROCESSED_AD_CSV)
    processed_pos_df = read_from_file(common.PROCESSED_POSITION_CSV)
    processed_user_df = read_from_file(common.PROCESSED_USER_CSV)
    merge_train_data = pd.merge(ori_train_df,processed_ad_df,how='left',on='creativeID')
    merge_train_data = pd.merge(merge_train_data, processed_pos_df,how='left', on='positionID')
    merge_train_data = pd.merge(merge_train_data, processed_user_df,how='left', on='userID')
    merge_train_data.drop(['appPlatform','sitesetID','positionType','gender','education','marriageStatus','haveBaby'], axis=1, inplace=True)
    #print merge_train_data.info()
    #print merge_train_data.describe()

    merge_train_data.to_csv(dst_file_name,index=False)

def process_test_file(src_file_name, dst_file_name, is_skip_regenerate=False):
    ori_test_df = read_from_file(src_file_name)
    #print ori_train_df.info()
    #print ori_train_df.describe()
    if not is_skip_regenerate:
        print "Process Ad File..."
        process_ad_file(common.ORIGIN_AD_CSV,common.PROCESSED_AD_CSV)
        print "Process Position File..."
        process_position_file(common.ORIGIN_POSITION_CSV, common.PROCESSED_POSITION_CSV)
        print "Process User File..."
        process_user_file(common.ORIGIN_USER_CSV, common.PROCESSED_USER_CSV)
    
    #merge data frame
    print "Merge Data..."
    processed_ad_df = read_from_file(common.PROCESSED_AD_CSV)
    processed_pos_df = read_from_file(common.PROCESSED_POSITION_CSV)
    processed_user_df = read_from_file(common.PROCESSED_USER_CSV)
    merge_test_data = pd.merge(ori_test_df,processed_ad_df,how='left',on='creativeID')
    merge_test_data = pd.merge(merge_test_data, processed_pos_df,how='left', on='positionID')
    merge_test_data = pd.merge(merge_test_data, processed_user_df,how='left', on='userID')
    merge_test_data.drop(['appPlatform','sitesetID','positionType','gender','education','marriageStatus','haveBaby'], axis=1, inplace=True)
    #print merge_train_data.info()
    #print merge_train_data.describe()

    merge_test_data.to_csv(dst_file_name,index=False)


def analysis_ad_data(file_name):
    ad_df = read_from_file(file_name)
    print ad_df.info()
    print ad_df.describe()

    #plot section
    plt.figure(figsize=(1,3))

    plt.subplot(131)
    ad_df['appPlatform'].value_counts().plot(kind='pie')
    plt.title('appPlatform')

    plt.subplot(132)
    ad_df['appID'].value_counts().plot(kind='bar')
    plt.title('appID')

    plt.subplot(133)
    ad_df['advertiserID'].plot(kind='kde')
    plt.title('advertiserID')

    plt.show()

def analysis_pos_data(file_name):
    pos_df = read_from_file(file_name)
    print pos_df.info()
    print pos_df.describe()

    #plot section
    plt.figure(figsize=(1,3))

    plt.subplot(131)
    pos_df['positionID'].value_counts().plot()
    plt.title('positionID')

    plt.subplot(132)
    pos_df['sitesetID'].value_counts().plot(kind='bar')
    plt.title('sitesetID')

    plt.subplot(133)
    pos_df['positionType'].value_counts().plot(kind='bar')
    plt.title('positionType')

    plt.show()

def analysis_user_data(file_name):
    user_df = read_from_file(file_name)
    print user_df.info()
    print user_df.describe()

    #plot section
    plt.figure(figsize=(2,3))

    plt.subplot(231)
    age_ser = user_df['age'].value_counts()
    age_ser.plot(use_index=False)
    plt.legend(loc='best')
    plt.title('age')

    plt.subplot(232)
    user_df['gender'].value_counts().plot(kind='pie')
    plt.title('gender')

    plt.subplot(233)
    user_df['education'].value_counts().plot(kind='bar')
    plt.title('education')

    plt.subplot(234)
    user_df['marriageStatus'].value_counts().plot(kind='pie')
    plt.title('marriageStatus')

    plt.subplot(224)
    user_df['haveBaby'].value_counts().plot(kind='bar')
    plt.title('haveBaby')

    plt.show()

if __name__ == '__main__':
    #process_ad_file(common.ORIGIN_AD_CSV, common.PROCESSED_AD_CSV)
    #process_position_file(common.ORIGIN_POSITION_CSV, common.PROCESSED_POSITION_CSV)
    #process_user_file(common.ORIGIN_USER_CSV, common.PROCESSED_USER_CSV)
    #analysis_ad_data(common.ORIGIN_AD_CSV)
    #analysis_pos_data(common.ORIGIN_POSITION_CSV)
    #analysis_user_data(common.ORIGIN_USER_CSV)
    process_train_file(common.ORIGIN_TRAIN_CSV, common.PROCESSED_TRAIN_CSV,True)
    process_test_file(common.ORIGIN_TEST_CSV, common.PROCESSED_TEST_CSV, True)