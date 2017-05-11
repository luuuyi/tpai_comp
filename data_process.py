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

def process_ad_file(src_file_name,dst_file_name):
    origin_ad_df = read_from_file(src_file_name)
    #print origin_ad_df.info()
    #print origin_ad_df.describe()
    dummy_app_platform = pd.get_dummies(origin_ad_df['appPlatform'], prefix='appPlatform')
    processd_ad_df = pd.concat([origin_ad_df,dummy_app_platform],axis=1)
    #print processd_ad_df.info()
    #print processd_ad_df.describe()

    processd_ad_df.to_csv(dst_file_name)

def process_position_file(src_file_name,dst_file_name):
    origin_pos_df = read_from_file(src_file_name)
    #print origin_pos_df.info()
    #print origin_pos_df.describe()
    dummy_siteset = pd.get_dummies(origin_pos_df['sitesetID'], prefix='sitesetID')
    dummy_pos_type = pd.get_dummies(origin_pos_df['positionType'], prefix='positionType')
    processd_pos_df = pd.concat([origin_pos_df,dummy_siteset,dummy_pos_type], axis=1)
    #print processd_pos_df.info()
    #print processd_pos_df.describe()

    processd_pos_df.to_csv(dst_file_name)

def process_user_file(src_file_name,dst_file_name):
    origin_user_df = read_from_file(src_file_name)
    #print origin_user_df.info()
    #print origin_user_df.describe()
    dummy_gender = pd.get_dummies(origin_user_df['gender'], prefix='gender')
    dummy_education = pd.get_dummies(origin_user_df['education'], prefix='education')
    dummy_marr_status = pd.get_dummies(origin_user_df['marriageStatus'], prefix='marriageStatus')
    dummy_have_baby = pd.get_dummies(origin_user_df['haveBaby'], prefix='haveBaby')
    processd_user_df = pd.concat([origin_user_df, dummy_gender, dummy_education, dummy_marr_status, dummy_have_baby], axis=1)
    print processd_user_df.info()
    print processd_user_df.describe()

    processd_user_df.to_csv(dst_file_name)

if __name__ == '__main__':
    #process_ad_file(common.ORIGIN_AD_CSV, common.PROCESSED_AD_CSV)
    #process_position_file(common.ORIGIN_POSITION_CSV, common.PROCESSED_POSITION_CSV)
    process_user_file(common.ORIGIN_USER_CSV, common.PROCESSED_USER_CSV)