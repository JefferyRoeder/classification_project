import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import split_scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wrangle
import env
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


url = env.get_db_url('telco_churn')


def clean_telco(df):
    df.total_charges = df.total_charges.replace(r'^\s*$', np.nan, regex=True)
    df = df[df.total_charges.isna() == False]
    df['total_charges'] = df['total_charges'].astype(float)
    df['churn'] = df.churn == 'Yes'
    df['senior_citizen'] = df.senior_citizen == 1
    df['is_male'] = df.gender == 'Male'
    df['paperless_billing'] = df.paperless_billing == 'Yes'

    df['family'] = (df.partner == 'Yes') | (df.dependents == 'Yes')

    df['phone'] = (df.phone_service == 'Yes') | (df.multiple_lines == 'Yes')
    df['streaming'] = (df.streaming_tv == 'Yes') | (df.streaming_movies == 'Yes')
    df['tech_protection'] = (df.device_protection == 'Yes') | (df.tech_support == 'Yes')
    df['internet_security'] = (df.online_backup == 'Yes') | (df.online_security == 'Yes')
    df['services'] = np.where((df['phone'] == True) & (df['internet_service_type_id'] <3),'internet w phone',np.where((df['phone'] == False) & (df['internet_service_type_id'] <3),'internet only','phone only'))
    return df
    

def tenure_churn(df):
    df_churned = df[['tenure','churn']].groupby('tenure').sum()
    df_total = df[['tenure','churn']].groupby('tenure').count()
    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on='tenure')
    df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    df_churn_rate = df_churn_rate.reset_index()
    all_contracts = sns.scatterplot(x='tenure',y='churn_rate',data=df_churn_rate)
    plt.show()

def contract_churn(df):
    df_churned = df[['tenure','contract_type_id','churn']].groupby(['tenure','contract_type_id']).sum()
    df_total = df[['tenure','contract_type_id','churn']].groupby(['tenure','contract_type_id']).count()
    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on=['tenure','contract_type_id'])
    df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    df_churn_rate = df_churn_rate.reset_index()
    df_churn_rate['contract_type_id'] = np.where(df_churn_rate['contract_type_id']== 1,'Mon to Mon',(np.where(df_churn_rate['contract_type_id']== 2,'1 year',(np.where(df_churn_rate['contract_type_id']==3,'2 year',"")))))
    all_contracts = sns.scatterplot(x='tenure',y='churn_rate',data=df_churn_rate,hue='contract_type_id',palette='tab10')
    plt.legend(bbox_to_anchor=(1.05,1),loc=2)


def internet_churn(df):
    df_churned = df[['tenure','internet_service_type_id','churn']].groupby(['tenure','internet_service_type_id']).sum()
    df_total = df[['tenure','internet_service_type_id','churn']].groupby(['tenure','internet_service_type_id']).count()

    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on=['tenure','internet_service_type_id'])

    df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    df_churn_rate = df_churn_rate.reset_index()




    df_churn_rate
    df_churn_rate['internet_service_type_id'] = np.where(df_churn_rate['internet_service_type_id']== 1,'DSL',(np.where(df_churn_rate['internet_service_type_id']== 2,'Fiber optic',(np.where(df_churn_rate['internet_service_type_id']==3,'none',"")))))
    all_contracts = sns.scatterplot(x='tenure',y='churn_rate',data=df_churn_rate,hue='internet_service_type_id',palette='tab10')
    plt.legend(bbox_to_anchor=(1.05,1),loc=2)



def services_churn(df):
    df_churned = df[['tenure','services','churn']].groupby(['tenure','services']).sum()
    df_total = df[['tenure','services','churn']].groupby(['tenure','services']).count()

    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on=['tenure','services'])

    df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    df_churn_rate = df_churn_rate.reset_index()


    #df_churn_rate
    #df_churn_rate['internet_service_type_id'] = np.where(df_churn_rate['internet_service_type_id']== 1,'DSL',(np.where(df_churn_rate['internet_service_type_id']== 2,'Fiber optic',(np.where(df_churn_rate['internet_service_type_id']==3,'none',"")))))
    all_contracts = sns.scatterplot(x='tenure',y='churn_rate',data=df_churn_rate,hue='services',palette='tab10')
    plt.legend(bbox_to_anchor=(1.05,1),loc=2)



def phone_churn(df):
    df_churned = df[['phone','churn']].groupby('phone').sum()
    df_total = df[['phone','churn']].groupby('phone').count()
    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on='phone')
    df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    df_churn_rate = df_churn_rate.reset_index()
    return df_churn_rate[['phone','churn_rate']]
    


def security_churn(df):
    df_churned = df[['online_security','churn']].groupby('online_security').sum()
    df_total = df[['online_security','churn']].groupby('online_security').count()
    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on='online_security')
    df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    df_churn_rate = df_churn_rate.reset_index()
    return df_churn_rate[['online_security','churn_rate']]


def backup_churn(df):
    df_churned = df[['online_backup','churn']].groupby('online_backup').sum()
    df_total = df[['online_backup','churn']].groupby('online_backup').count()
    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on='online_backup')
    df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    df_churn_rate = df_churn_rate.reset_index()
    return df_churn_rate[['online_backup','churn_rate']]


def senior_churn(df):
    df_churned = df[['senior_citizen','churn']].groupby('senior_citizen').sum()
    df_total = df[['senior_citizen','churn']].groupby('senior_citizen').count()
    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on='senior_citizen')
    df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    df_churn_rate = df_churn_rate.reset_index()
    return df_churn_rate[['senior_citizen','churn_rate']]


def charges_churn(df):
    df_churned = df[['monthly_charges','internet_service_type_id','churn']].groupby(['monthly_charges','internet_service_type_id']).sum()
    df_total = df[['monthly_charges','internet_service_type_id','churn']].groupby(['monthly_charges','internet_service_type_id']).count()

    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on=['monthly_charges','internet_service_type_id'])

    df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    df_churn_rate = df_churn_rate.reset_index()

    df_churn_rate['internet_service_type_id'] = np.where(df_churn_rate['internet_service_type_id']== 1,'DSL',(np.where(df_churn_rate['internet_service_type_id']== 2,'Fiber optic',(np.where(df_churn_rate['internet_service_type_id']==3,'none',"")))))
    all_contracts = sns.scatterplot(x='monthly_charges',y='churn_rate',data=df_churn_rate,hue='internet_service_type_id',palette='tab10')
    plt.legend(bbox_to_anchor=(1.05,1),loc=2)
    


def charges_tenure(df):
    df_churned = df[['monthly_charges','internet_service_type_id','tenure']].groupby(['monthly_charges','internet_service_type_id']).sum()
    df_total = df[['monthly_charges','internet_service_type_id','tenure']].groupby(['monthly_charges','internet_service_type_id']).count()

    df_churn_rate = pd.merge(df_churned,df_total,how='inner',on=['monthly_charges','internet_service_type_id'])

    #df_churn_rate['churn_rate'] = df_churn_rate.churn_x / df_churn_rate.churn_y
    #df_churn_rate = df_churn_rate.reset_index()

    #df_churn_rate['internet_service_type_id'] = np.where(df_churn_rate['internet_service_type_id']== 1,'DSL',(np.where(df_churn_rate['internet_service_type_id']== 2,'Fiber optic',(np.where(df_churn_rate['internet_service_type_id']==3,'none',"")))))
    all_contracts = sns.scatterplot(x='monthly_charges',y='tenure',data=df_churn_rate,hue='internet_service_type_id',palette='tab10')
    plt.legend(bbox_to_anchor=(1.05,1),loc=2)



def quantile_churn(df):
    df_churned = df[['monthly_charges','churn']]
    df_churned['quant_25'] = df.monthly_charges < df_churned['monthly_charges'].quantile(.25)
    df_churned['quant_75'] = df.monthly_charges > df_churned['monthly_charges'].quantile(.75)
    print("Customers churned at an average rate of: " ,round((df_churned.churn.sum()/len(df_churned))*100,1),'%')
    print("Customers in the top 25% of monthly charges churned at a rate of: ",(df_churned.quant_75.sum()/len(df_churned))*100,"%")
    print("Customers in the bottom 25% of monthly charges churned at a rate of: ", (df_churned.quant_25.sum()/len(df_churned))*100,"%")


def charges_distro(df):
    sns.scatterplot(x='tenure',y='monthly_charges',data=df,hue='internet_service_type_id',alpha=.6)
    plt.legend(bbox_to_anchor=(1.05,1),loc=2)