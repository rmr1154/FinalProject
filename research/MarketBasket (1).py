#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3 as sq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[2]:


dbname = 'naxml_34.db'
dbpath = '../data/'
db_string = f'{dbpath}{dbname}'


# In[3]:


load_csv = True


# In[21]:


if load_csv == True:
    df = pd.read_csv('../data/salestransbig.csv')


# In[22]:


if load_csv == False:
    cnx = sq.connect(db_string)
    df = pd.read_sql_query("SELECT Filename, SalesQuantity, Description, POSCode, MerchandiseCode FROM TransactionLine_Products", cnx)
    


# In[23]:


df.head()


# In[24]:


df.set_index('transaction_id',inplace=True)
df = (df[df.groupby('transaction_id')['category'].count() > 1])
df.head()


# In[25]:


import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[26]:


basket = (df.groupby(['transaction_id', 'category'])['salesquantity'].sum().unstack().reset_index().fillna(0)
          .set_index('transaction_id'))


# In[27]:


basket.head()


# In[28]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 0:
        return 1


# In[29]:


basket_sets = basket.applymap(encode_units)
basket_sets.head()


# In[35]:


frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
frequent_itemsets.head(10)


# In[36]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(10)


# In[37]:


print(f'Number of assocation rules {rules.shape[0]}')


# In[38]:


rules.hist('confidence', grid=True, bins = 30)


# In[39]:


rules.hist('lift', grid=True, bins=30)



# In[52]:


plt.scatter(rules['support'],rules['confidence'], c=rules['lift'],alpha=0.5)


# In[ ]:




