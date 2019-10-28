import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import split_scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wrangle
import env
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


url2 = env.get_db_url('titanic_db')

def prep_titanic():
    
    df = pd.read_sql("""

SELECT *
FROM passengers
"""
,url2)

    df.drop(columns=['deck'],inplace=True)
    df.fillna(np.nan,inplace=True)
    imp_mode = SimpleImputer(missing_values=np.nan,strategy='most_frequent')

    imp_mode.fit(df[['embarked']])

    df['embarked'] = imp_mode.transform(df[['embarked']])

    imp_mode.fit(df[['embark_town']])
    df['embark_town'] = imp_mode.transform(df[['embark_town']])
    scaler = MinMaxScaler().fit(df[['age','fare']])
    df_scaled = scaler.transform(df[['age','fare']])
    df_scaled = pd.DataFrame(df_scaled)
    df = pd.concat([df,df_scaled],axis=1,join='inner')
    df.rename(columns={0:'age_scaled',1:'fare_scaled'},inplace=True)

    return df



url = env.get_db_url('iris_db')

def prep_iris():
    df = pd.read_sql("""

SELECT *
FROM measurements m
JOIN species s on s.species_id = m.species_id;
"""

,url)
    df.drop(['species_id','measurement_id'],axis=1,inplace=True)
    df.rename(columns={'species_name':'species'},inplace=True)
    int_encoder = LabelEncoder()
    int_encoder.fit(df.species)
    df.species = int_encoder.transform(df.species)


    return df

def prep_iris_no_encode():
    df = pd.read_sql("""

SELECT *
FROM measurements m
JOIN species s on s.species_id = m.species_id;
"""

,url)
    df.drop(['species_id','measurement_id'],axis=1,inplace=True)
    df.rename(columns={'species_name':'species'},inplace=True)
    #int_encoder = LabelEncoder()
    #int_encoder.fit(df.species)
    #df.species = int_encoder.transform(df.species)


    return df