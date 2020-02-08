#!/usr/bin/env python
# coding: utf-8

# In[30]:


# pip install pymssql
import pymssql
import pandas as pd

pd.options.display.max_columns = None


# In[31]:


load_csv = True


# In[32]:


if load_csv == True:
    df_sku = pd.read_csv('df_sku.csv')


# In[33]:


if load_csv == False:
    sqluser = input('Enter SQL User')
    sqlpass = input(f'Enter Password for {sqluser}')
    ## instance a python db connection object- same form as psycopg2/python-mysql drivers also
    conn = pymssql.connect(server="192.168.254.13", user=sqluser,password=passwrd, port=1433)  # You can lookup the port number inside SQL server. 

    stmt = "SELECT             SKU_ID             ,UOM_ID             ,SalesCategoryID             ,cat.CategoryID             ,NACSCategoryID             ,Category             ,Description             ,LongDescription             ,ShortDescription             ,POSDescription             FROM AgilityPB.dbo.tbl_SKU sku             left outer join Agility_Net.dbo.tbl_Categories cat on sku.SalesCategoryID = cat.CategoryID"
    # Excute Query here
    df_sku = pd.read_sql(stmt,conn)
    df_sku.to_csv('df_sku.csv')


# In[48]:


df_sku.head()


# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(df_sku['LongDescription'], df_sku['Category'], train_size=0.8)


# <h1>Naive Bayes Classifier</h1>

# In[79]:


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))


# <h1>K-nearest Neighbor</h1>

# In[52]:


from sklearn.neighbors import KNeighborsClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', KNeighborsClassifier()),
                     ])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))


# <h1>Support Vector Machine (SVM)</h1>

# In[53]:


from sklearn.svm import LinearSVC


# In[54]:


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))


# <h1>Decision Tree</h1>

# In[55]:


from sklearn import tree
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', tree.DecisionTreeClassifier()),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))


# <h1>Random Forest</h1>

# In[56]:


from sklearn.ensemble import RandomForestClassifier

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators=100)),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))


# <h1>Deep Neural Networks</h1>

# In[103]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  Dropout, Dense
from tensorflow.keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn import metrics


# In[93]:


def TFIDF(X_train, X_test,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train,X_test)


# In[94]:


#prepare target
def prepare_targets_le(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


# In[104]:


#prepare target
def prepare_targets_oe(y_train, y_test):
    oe = OrdinalEncoder()
    oe.fit(y_train)
    y_train_enc = oe.transform(y_train)
    y_test_enc = oe.transform(y_test)
    return y_train_enc, y_test_enc


# In[105]:


def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
    """
    buildModel_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    node = 1 # number of nodes
    nLayers = 2 # number of  hidden layer

    model.add(Dense(node,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# In[106]:


df_sku['Category'].nunique()


# In[107]:


X_train_tfidf,X_test_tfidf = TFIDF(X_train,X_test)
y_train_enc, y_test_enc = prepare_targets_oe(y_train, y_test)
model_DNN = Build_Model_DNN_Text(X_train_tfidf.shape[1], 29) # 29 is df_sku['Category'].nunique()
model_DNN.fit(X_train_tfidf, y_train_enc,
                              validation_data=(X_test_tfidf, y_test_enc),
                              epochs=10,
                              batch_size=128,
                              verbose=2)

predicted = model_DNN.predict(X_test_tfidf)

print(metrics.classification_report(y_test, predicted))


# In[108]:


def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# In[109]:


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


# In[110]:


# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)


# In[ ]:


# define the model
model = Sequential()
model.add(Dense(10, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train_enc, y_train_enc, epochs=100, batch_size=16, verbose=2)
# evaluate the keras model
_, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

