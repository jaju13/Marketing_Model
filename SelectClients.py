#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from functools import reduce


# In[2]:


results_folder_path = './predicted_results/'

Revenue = pd.read_csv(results_folder_path+'Predicted_Revenue.csv',index_col=0)
Propensity_CC = pd.read_csv(results_folder_path+'Propensity_Sale_CC.csv',index_col=0)
Propensity_CL = pd.read_csv(results_folder_path+'Propensity_Sale_CL.csv',index_col=0)
Propensity_MF = pd.read_csv(results_folder_path+'Propensity_Sale_MF.csv',index_col=0)


# In[4]:


Revenue = Revenue[['Client','Predicted_revenue']]
Propensity_CC = Propensity_CC[['Client','Sale_CC','Sale_CC_probability']]
Propensity_CL = Propensity_CL[['Client','Sale_CL','Sale_CL_probability']]
Propensity_MF = Propensity_MF[['Client','Sale_MF','Sale_MF_probability']]


# In[5]:


dfs = [Revenue,Propensity_CC,Propensity_CL,Propensity_MF]
Client_predictions = reduce(lambda df1, df2 : pd.merge(df1,df2, on='Client', how='inner'), dfs)
Client_predictions.head()


# In[6]:


Client_predictions['Highest_Propensity_probability'] = Client_predictions['Predicted_max_Revenue'] = Client_predictions[['Sale_CC_probability','Sale_CL_probability','Sale_MF_probability']].max(axis=1)
Client_predictions['Highest_Propensity_product'] = Client_predictions['Predicted_max_Revenue'] = Client_predictions[['Sale_CC_probability','Sale_CL_probability','Sale_MF_probability']].idxmax(axis=1)


# In[7]:


Client_predictions['Highest_Propensity_product'] = Client_predictions['Highest_Propensity_product'].str.extract(r'Sale_(\w+)_')


# In[8]:


Client_predictions['Predicted_max_Revenue'] = Client_predictions['Highest_Propensity_probability']*Client_predictions['Predicted_revenue']


# In[9]:


Client_predictions.head()


# In[10]:


Client_predictions.sort_values(by=['Predicted_max_Revenue'],ascending=False,inplace=True)


# In[11]:


top_100_clients = Client_predictions.head(100)


# In[12]:


Client_predictions.to_csv(results_folder_path+'Maximum_predicted_revenue_per_client.csv')


# In[13]:


top_100_clients.to_csv(results_folder_path+'Top_100_clients.csv')


# In[14]:


def main():
    print("Selecting top 100 clients")
if __name__ == "__main__":
    main()


# In[ ]:




