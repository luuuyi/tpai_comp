import common
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def read_from_file(file_name, chunk_size=100000):
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

def analysis_connec_type(file_name):
    df = read_from_file(file_name)
    #print df.info()
    #print df.describe()

    label_0 = df.connectionType[df.label == 0].value_counts()
    label_1 = df.connectionType[df.label == 1].value_counts()

    plt_df = pd.DataFrame({'No':label_0, 'Yes':label_1})
    plt_df.plot(kind='bar', stacked=True)
    plt.title('Depend On ConnectionType')
    plt.xlabel('ConnectionType') 
    plt.ylabel('Numbers')
    plt.show()

def analysis_telecom(file_name):
    df = read_from_file(file_name)
    #print df.info()
    #print df.describe()

    label_0 = df.telecomsOperator[df.label == 0].value_counts()
    label_1 = df.telecomsOperator[df.label == 1].value_counts()

    plt_df = pd.DataFrame({'No':label_0, 'Yes':label_1})
    plt_df.plot(kind='bar', stacked=True)
    plt.title('Depend On telecomsOperator')
    plt.xlabel('telecomsOperator') 
    plt.ylabel('Numbers')
    plt.show()

def analysis_appPlatform(file_name):
    df = read_from_file(file_name)
    #print df.info()
    #print df.describe()

    platform_1 = df.label[df.appPlatform_1 == 1].value_counts()
    platform_2 = df.label[df.appPlatform_2 == 1].value_counts()

    plt_df = pd.DataFrame({'platform1':platform_1, 'platform2':platform_2})
    plt_df.plot(kind='bar', stacked=True)
    plt.title('Depend On Platform')
    plt.xlabel('Label') 
    plt.ylabel('Numbers')
    plt.show()

def analysis_siteset_and_posType(file_name):
    df = read_from_file(file_name)
    #print df.info()
    #print df.describe()

    siteset_0 = df.label[df.sitesetID_0 == 1].value_counts()
    siteset_1 = df.label[df.sitesetID_1 == 1].value_counts()
    siteset_2 = df.label[df.sitesetID_2 == 1].value_counts()

    pos_type_0 = df.label[df.positionType_0 == 1].value_counts()
    pos_type_1 = df.label[df.positionType_1 == 1].value_counts()
    pos_type_2 = df.label[df.positionType_2 == 1].value_counts()
    pos_type_3 = df.label[df.positionType_3 == 1].value_counts()
    pos_type_4 = df.label[df.positionType_4 == 1].value_counts()
    pos_type_5 = df.label[df.positionType_5 == 1].value_counts()

    plt_df_1 = pd.DataFrame({'siteset_0':siteset_0, 'siteset_1':siteset_1, 'siteset_2':siteset_2})
    plt_df_2 = pd.DataFrame({'pos_type_0':pos_type_0, 'pos_type_1':pos_type_1, 'pos_type_2':pos_type_2, 'pos_type_3':pos_type_3, 'pos_type_4':pos_type_4, 'pos_type_5':pos_type_5})

    plt_df_1.plot(kind='bar', stacked=True)
    plt.title('Depend On SiteSet')

    plt_df_2.plot(kind='bar', stacked=True)
    plt.title('Depend On PositionType')

    plt.show()

#include gender, education, marriage-status, havebaby
def analysis_user(file_name):
    df = read_from_file(file_name)
    #print df.info()
    #print df.describe()

    gender_0 = df.label[df.gender_0 == 1].value_counts()
    gender_1 = df.label[df.gender_1 == 1].value_counts()
    gender_2 = df.label[df.gender_2 == 1].value_counts()

    eucation_0 = df.label[df.education_0 == 1].value_counts()
    eucation_1 = df.label[df.education_1 == 1].value_counts()
    eucation_2 = df.label[df.education_2 == 1].value_counts()
    eucation_3 = df.label[df.education_3 == 1].value_counts()
    eucation_4 = df.label[df.education_4 == 1].value_counts()
    eucation_5 = df.label[df.education_5 == 1].value_counts()
    eucation_6 = df.label[df.education_6 == 1].value_counts()
    eucation_7 = df.label[df.education_7 == 1].value_counts()

    marriage_0 = df.label[df.marriageStatus_0 == 1].value_counts()
    marriage_1 = df.label[df.marriageStatus_1 == 1].value_counts()
    marriage_2 = df.label[df.marriageStatus_2 == 1].value_counts()
    marriage_3 = df.label[df.marriageStatus_3 == 1].value_counts()

    havebaby_0 = df.label[df.haveBaby_0 == 1].value_counts()
    havebaby_1 = df.label[df.haveBaby_1 == 1].value_counts()
    havebaby_2 = df.label[df.haveBaby_2 == 1].value_counts()
    havebaby_3 = df.label[df.haveBaby_3 == 1].value_counts()
    havebaby_4 = df.label[df.haveBaby_4 == 1].value_counts()
    havebaby_5 = df.label[df.haveBaby_5 == 1].value_counts()
    havebaby_6 = df.label[df.haveBaby_6 == 1].value_counts()

    plt_df_1 = pd.DataFrame({'gender_0':gender_0, 'gender_1':gender_1, 'gender_2':gender_2})
    plt_df_2 = pd.DataFrame({'eucation_0':eucation_0, 'eucation_1':eucation_1, 'eucation_2':eucation_2, 'eucation_3':eucation_3, 'eucation_4':eucation_4, 'eucation_5':eucation_5, 'eucation_6':eucation_6, 'eucation_7':eucation_7})
    plt_df_3 = pd.DataFrame({'marriage_0':marriage_0, 'marriage_1':marriage_1, 'marriage_2':marriage_2, 'marriage_3':marriage_3})
    plt_df_4 = pd.DataFrame({'havebaby_0':havebaby_0, 'havebaby_1':havebaby_1, 'havebaby_2':havebaby_2, 'havebaby_3':havebaby_3, 'havebaby_4':havebaby_4, 'havebaby_5':havebaby_5, 'havebaby_6':havebaby_6})

    plt_df_1.plot(kind='bar', stacked=True)
    plt.title('Depend On Gender')

    plt_df_2.plot(kind='bar', stacked=True)
    plt.title('Depend On Education')

    plt_df_3.plot(kind='bar', stacked=True)
    plt.title('Depend On Marriage Status')

    plt_df_4.plot(kind='bar', stacked=True)
    plt.title('Depend On Have Baby')

    plt.show()

def analysis_age(file_name):
    df = read_from_file(file_name)
    age_label_1 = df.age[df.label == 1].value_counts()
    age_label_1.plot(use_index=False)
    plt.title('Depend On Age In Label 1')
    plt.show()

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
    #analysis_connec_type(common.PROCESSED_TRAIN_CSV)
    #analysis_telecom(common.PROCESSED_TRAIN_CSV)
    #analysis_appPlatform(common.PROCESSED_TRAIN_CSV)
    #analysis_user(common.PROCESSED_TRAIN_CSV)
    analysis_age(common.PROCESSED_TRAIN_CSV)