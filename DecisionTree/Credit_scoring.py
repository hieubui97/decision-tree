# # Định giá tín dụng cá nhân

# ## Tính điểm tín dụng (Credit Scoring)
# - Các ngân hàng đối mặt với các rủi ro tín dụng (vỡ nợ)
# - Đánh giá tín dụng có thể áp dụng trí tuệ nhân tạo. Đánh giá khả năng chi trả:
#     - Đúng hạn
#     - Trễ hạn
#     - Không đủ khả năng thanh toán
#

# ## Mô hình tính điểm tín dụng
# - Mô hình hồi quy logistic
# - Mô hình mạng neural
# - Mô hình cây quyết định
# - Véc-tơ hỗ trợ

# ## Dữ liệu
# - Tập huấn luyện: cs-training.csv
# - Tập kiểm tra: cs-test.csv

# In[58]:


import pandas as pd
import numpy as np


# In[59]:


train = pd.read_csv("./data/cs-training.csv")


# ### Kiểm tra dữ liệu bị mất (NaN)

# In[60]:


print(train.isnull().sum())


# In[61]:


train = train.drop(train.columns.values[0],axis=1)


# In[62]:


print(train.shape)


# In[63]:


train.head()


# In[64]:


print(train.columns.values
)

# In[65]:


data_features=pd.read_excel("./data/Data Dictionary.xls")


# In[66]:


print(data_features)


# **Missing **
# - 29731, MonthlyIncome
# - 3924, NumberOfDependents
#

# **Removal missing data**

# In[67]:


train_no_missing = train.dropna()# remove missing values


# In[68]:


print(train_no_missing.shape)


# **Imputation by mean**

# In[69]:


train_imputer = train


# In[70]:


train_imputer["MonthlyIncome"].fillna(train_imputer["MonthlyIncome"].mean(), inplace=True)


# In[71]:


train_imputer["NumberOfDependents"].fillna(train_imputer["NumberOfDependents"].mean(), inplace=True)


# In[72]:


print(train_imputer.shape)


# ### Dữ liệu huấn luyện

# In[73]:


x_train = train_no_missing.drop(train_no_missing.columns.values[0],axis=1)


# In[74]:


print(x_train.shape)


# In[75]:


y_train = train_no_missing[train.columns.values[0]]


# In[76]:


print(y_train.shape)


# In[ ]:





# ### Dữ liệu kiểm tra

# In[77]:


test = pd.read_csv("./data/cs-test.csv")


# In[78]:


test = test.drop(test.columns.values[0], axis=1)


# In[79]:


print(test.shape)


# In[80]:


print(test.head())


# In[ ]:


x_test = test.drop(test.columns.values[0], axis=1)
print(x_test.shape)


# In[ ]:


y_test = test[test.columns.values[0]]


# In[ ]:


print(y_test.shape)


# ### Kiểm tra dữ liệu bị mất (missing)

# In[ ]:


print(x_test.isnull().sum())


# ** Bỏ qua các bản ghi không có giá trị **

# In[ ]:


x_test_no_missing = x_test.dropna()
print(x_test_no_missing.shape)
# x_test_no_missing.isnull().sum()


# ** Khôi phục các bản ghi không có giá trị**

# In[ ]:


x_test_imputer = x_test


# In[ ]:


x_test_imputer["MonthlyIncome"].fillna(x_test_imputer["MonthlyIncome"].mean(), inplace=True)


# In[ ]:


x_test_imputer["NumberOfDependents"].fillna(x_test_imputer["NumberOfDependents"].mean(), inplace=True)


# In[ ]:


print(x_test_imputer.isnull().sum())


# ## Mô hình học máy (Decision tree regression)

# ### Đối với dữ liệu loại bỏ các bản ghi bị lỗi

# In[ ]:


# x_train, y_train, x_test_no_missing
print(x_train.shape,
y_train.shape,
x_test_no_missing.shape)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE


# In[ ]:


# max_depth=None,
# max_leaf_nodes=None,
# min_samples_leaf=None,
# regressor = DecisionTreeRegressor(min_samples_leaf=100)
regressor = DecisionTreeRegressor(max_depth=10)
# regressor = DecisionTreeRegressor(max_leaf_nodes=100)


# In[ ]:


print(regressor.fit(x_train, y_train))


# In[ ]:

print(regressor.score(x_train, y_train))



# In[ ]:


features=x_train.columns.values


# In[ ]:


from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file='tree_credit.dot', feature_names=features)

import pydot
graphs = pydot.graph_from_dot_file("tree_credit.dot")
graphs[0].write_png("tree_credit.png")
#get_ipython().system(' dot -Tpng tree_credit.dot > tree_credit.png')


# In[ ]:


y_predict = regressor.predict(x_test_no_missing)


# In[ ]:

print(y_predict[:5])

# In[ ]:

import matplotlib.pyplot as plt

# In[ ]:

plt.hist(y_predict, bins=10)
plt.show()

# In[ ]:

print(regressor.score(x_train, y_train))

# ### Đối với dữ liệu Imputer


# In[ ]:

# train_imputer, x_test_imputer

print(train_imputer.shape, x_test_imputer.shape)


# In[ ]:

x_train = train_imputer.drop(train_imputer.columns.values[0],axis=1)
y_train = train_imputer[train_imputer.columns.values[0]]

# In[ ]:


# max_depth=None,
# max_leaf_nodes=None,
# min_samples_leaf=None,
# dt_reg = DecisionTreeRegressor(min_samples_leaf=10)
# dt_reg = DecisionTreeRegressor(max_depth=25)
dt_reg = DecisionTreeRegressor(max_depth=15)
# dt_reg = DecisionTreeRegressor(max_leaf_nodes=100)


# In[ ]:

print(dt_reg.fit(x_train, y_train))

# In[ ]:


from sklearn.tree import export_graphviz
export_graphviz(dt_reg, out_file='tree_credit_imputer.dot', feature_names=features)

graphs = pydot.graph_from_dot_file("tree_credit_imputer.dot")
graphs[0].write_png("tree_credit_imputer.png")
#get_ipython().system(' dot -Tpng tree_credit_imputer.dot > tree_credit_imputer.png')


# In[ ]:

y_pred_imputer=dt_reg.predict(x_test_imputer)

# In[ ]:

plt.hist(y_predict, bins=10)
plt.show()

# In[ ]:

print(dt_reg.score(x_train, y_train))