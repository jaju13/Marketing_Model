#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
pd.options.display.max_columns=None

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import plot_importance

from imblearn.over_sampling import SMOTE
from sklearn.utils import resample


# In[2]:


# Important paths
classification_model_folder_path = './models/Classification/'  # Folder containing all classification models which are selected
regression_model_folder_path = './models/Regression/'  # Folder containing all classification models which are selected

cleaned_data_path = './data/cleaned/data_for_modeling.pkl'  # pickle file containing cleaned data
results_folder_path = './predicted_results/' # Folder containing predicted results
# Column information

## Columns from cleaned data that need to be removed for classification
cols_to_drop_classification=['Sale_MF', 'Sale_CC', 'Sale_CL','Revenue_MF', 
              'Revenue_CC', 'Revenue_CL','VolumeCred_CA','TransactionsCred_CA',
              'VolumeDeb_CA','TransactionsDeb_CA']

## Columns from cleaned data that need to be removed for classification
cols_to_drop_regression=['Sale_MF', 'Sale_CC', 'Sale_CL','Revenue_MF', 'Revenue_CC', 'Revenue_CL']

## Numeric columns required to be standardized in classification
numeric_cols_to_standardize_classification = ['Age', 'Tenure', 'Count_CA', 'Count_SA', 'Count_MF',
       'Count_OVD', 'Count_CC', 'Count_CL', 'ActBal_CA', 'ActBal_SA',
       'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL', 'VolumeCred',
       'TransactionsCred', 'VolumeDeb', 'VolumeDebCash_Card', 'VolumeDebCashless_Card',
       'VolumeDeb_PaymentOrder', 'TransactionsDeb','TransactionsDebCash_Card', 'TransactionsDebCashless_Card',
       'TransactionsDeb_PaymentOrder']

## Numeric columns required to be standardized in regression
numeric_cols_to_standardize_regression = ['Age', 'Tenure', 'Count_CA', 'Count_SA', 'Count_MF',
       'Count_OVD', 'Count_CC', 'Count_CL', 'ActBal_CA', 'ActBal_SA',
       'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL', 'VolumeCred',
       'VolumeCred_CA', 'TransactionsCred', 'TransactionsCred_CA', 'VolumeDeb',
       'VolumeDeb_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card',
       'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA',
       'TransactionsDebCash_Card', 'TransactionsDebCashless_Card',
       'TransactionsDeb_PaymentOrder']

## Categorical columns required to be encoded
cat_cols_to_encode=['Sex']

## Models selected for each target sale product
model_selected = {
    'Sale_MF': 'Sale_MF_xgboost_model.json',
    'Sale_CC': 'Sale_CC_xgboost_model.json',
    'Sale_CL': 'Sale_CL_xgboost_model.json'
}

regression_model_features =  ['VolumeCred', 'ActBal_MF', 'VolumeDeb', 'TransactionsDeb',
       'VolumeDeb_CA', 'Count_MF', 'TransactionsDebCashless_Card', 'Sex_F',
       'VolumeDeb_PaymentOrder', 'VolumeCred_CA', 'TransactionsCred', 'Sex_M',
       'ActBal_CL', 'ActBal_CC', 'TransactionsDeb_CA']

## Order of columns in models trained using xgboost
xgboost_column_order = ['ActBal_SA', 'VolumeCred', 'VolumeDebCash_Card', 'VolumeDeb_PaymentOrder', 
                        'ActBal_CC', 'Count_CC', 'TransactionsDeb', 'ActBal_MF', 'Count_MF', 
                        'TransactionsDebCashless_Card', 'TransactionsDebCash_Card', 'Count_CL', 'Sex_F', 
                        'Count_OVD', 'Sex_M', 'TransactionsDeb_PaymentOrder', 'Age', 'ActBal_CL', 
                        'VolumeDeb', 'VolumeDebCashless_Card', 'ActBal_CA', 'Count_CA', 'Tenure', 
                        'ActBal_OVD', 'TransactionsCred', 'Count_SA']

## target columns for which predictions are to be made
target_columns = ['Sale_MF','Sale_CC','Sale_CL']


# In[3]:


# Function to load pickled files, contains error handling
def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)  
        return data  
    except (pickle.UnpicklingError, EOFError, AttributeError, TypeError):
        return False  
    except Exception as e:
        return False  

# Function to save dataframes as csv files, contains error handling
def save_csv(folder_to_save, file_name,df_to_save):
    if os.path.exists(folder_to_save) and os.path.isdir(folder_to_save) :
        df_to_save.to_csv(folder_to_save+'/'+file_name)
    else:
        print(f"Error, folder {folder_to_save} does not exist")


# In[4]:


# Load cleaned data
data = load_pickle_file(cleaned_data_path)


# In[5]:


# Only keep those records where no data is available on previous sale
cleaned_data_without_labels = data.iloc[np.where((data.Sale_MF+data.Sale_CC+data.Sale_CL)==0)[0],].copy()


# In[ ]:





# In[6]:


def predict_revenue(data_without_labels):
    # Drop columns which are not required
    data_without_labels.drop(columns=cols_to_drop_regression,inplace=True)
    
    # Encode categorical variables
    data_encoded = pd.get_dummies(data_without_labels, columns=cat_cols_to_encode)
    
    # Move 'Client' identifier to index
    data_encoded.set_index('Client',inplace=True)
    
    # Load the scaler
    folder_path_target = regression_model_folder_path
    file_toopen = 'scaler.pkl'
    scaler = load_pickle_file(folder_path_target+file_toopen)

    #Encode data
    try:
        data_encoded[numeric_cols_to_standardize_regression] = scaler.transform(data_encoded[numeric_cols_to_standardize_regression])
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Reduce data to contain only features selected for the model
    data_selected = data_encoded.loc[:, regression_model_features]
    
    # Load the model
    model = load_pickle_file(regression_model_folder_path+'/best_model.pkl')
    
    # make predictions
    predicted_revenue = model.predict(data_selected)
    data_selected['Predicted_revenue'] = predicted_revenue
    
    # Move 'Client' identifier back to columns
    data_selected.reset_index(inplace=True,drop=False)
    
    # Save as csv file 
    save_csv(results_folder_path, 'Predicted_Revenue.csv',data_selected)


# In[7]:


def predict_propensity(data_without_labels):
    # Drop columns which are not required
    data_without_labels.drop(columns=cols_to_drop_classification,inplace=True)

    # Encode categorical variables
    data_encoded = pd.get_dummies(data_without_labels, columns=cat_cols_to_encode)

    for sale_target in target_columns:
        data_encoded_target = data_encoded.copy()
        # Load the scaler
        folder_path_target = classification_model_folder_path+sale_target+"/"
        file_toopen = 'scaler.pkl'
        scaler = load_pickle_file(folder_path_target+file_toopen)

        #Encode data
        try:
            data_encoded_target[numeric_cols_to_standardize_classification] = scaler.transform(data_encoded_target[numeric_cols_to_standardize_classification])
        except Exception as e:
            print(f"Unexpected error: {e}")

        # Load the model
        file_toopen = model_selected[sale_target]
        try:
            loaded_xgb_model = xgb.XGBClassifier()
            loaded_xgb_model.load_model(folder_path_target+file_toopen)
        except Exception as e:
            print(f"Unexpected error: {e}")

        # Move 'Client' identifier to index
        data_encoded_target.set_index('Client',inplace=True)

        #Make predictions
        Sale_predictions = loaded_xgb_model.predict(data_encoded_target[xgboost_column_order])
        Sale_propabilities = loaded_xgb_model.predict_proba(data_encoded_target[xgboost_column_order])
        data_encoded_target[sale_target]=Sale_predictions
        data_encoded_target[sale_target+'_probability']=Sale_propabilities[:,1]

        # Move 'Client' identifier back to columns
        data_encoded_target.reset_index(inplace=True,drop=False)

        # Save as csv file 
        save_csv(results_folder_path, 'Propensity_'+sale_target+'.csv',data_encoded_target)



# In[8]:


def main():
    print("Generating predictions for Propensity to buy products")
    predict_propensity(cleaned_data_without_labels.copy())
    print("Generating predictions of potential revenue")
    predict_revenue(cleaned_data_without_labels.copy())
if __name__ == "__main__":
    main()


# In[ ]:




