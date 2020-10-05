# ## Hồi quy dựa trên cây quyết định

# Giả sử ta có các mẫu quan sát $(x_i,y_i), i=1,2,\dots, N$ với $x_i=(x_{i1},x_{i2},\dots,x_{ip})$.
#
# - Ta xây dựng thuật toán tự động chia miền chứa các biến $x_i$ thành các hình chữ nhật $R_m$
# - Ta mô tả hàm hồi quy $$y=f(x)=\sum\limits_{m=1}^M c_m I_{(x\in R_m)} $$
#
# hàm $I_{(x\in R_m)}=1$ nếu $x\in R_m$ và $I_{(x\in R_m)}=0$ nếu $x\not\in R_m$
# - Hàm hồi quy được xây dựng bằng phương pháp bình phương tối thiểu $$SSE=\sum\limits_{i=1}^N (y_i-f(x_i))^2. $$
# - Ta có thể biểu diễn $$SSE=\sum\limits_{m=1}^M \sum\limits_{x_i\in R_m} (y_i-c_m)^2. $$ Giá trị $c_m$ là giá trị trung bình của các giá trị $y_i$ trong miền $R_m$, nghĩa là $$c_m={\bf mean}(y_i: x_i\in R_m). $$

# **Thuật toán:**
# - Với mỗi nút ( không phải nút lá):
#     - Với mỗi biến $X_k$:
#         - Tìm điểm cắt tối ưu $s$ $$\arg\min_s \big[\sum\limits_{x_{ik}\le s}(y_i-c_1)^2+\sum\limits_{x_{ik}> s}(y_i-c_2)^2\big], $$ với  $c_1={\bf mean} (y_i: x_{ik}\le s)$ và $c_2={\bf mean} (y_i: x_{ik}> s)$
#     - Chọn biến $X_k$ và $s$ với $SSE$ bé nhất
# - Lặp đến khi đạt đến điều kiện dừng

# ## Ví dụ 1: Fuel consumption


# In[30]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[31]:

data = pd.read_csv("./data/auto-mpg.csv")


# In[32]:

data.head()

# In[33]:

x = data['displacement']
y = data['mpg']

# In[34]:

plt.scatter(x, y, c='blue')
plt.xlabel("displacement")
plt.ylabel("mpg")
plt.title("Hồi quy")
plt.show()

# In[35]:

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# In[36]:

x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)

# In[37]:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# In[38]:

dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1, random_state=3)

# In[39]:

dt.fit(x_train, y_train)

# In[40]:

y_pred = dt.predict(x_test)

# In[41]:

mse_dt = MSE(y_test, y_pred)

# In[42]:

print(np.sqrt(mse_dt))

# In[43]:

xx = np.linspace(min(x), max(x), 400).reshape(-1,1)

# In[44]:

plt.scatter(x, y, c='blue')
plt.plot(xx, dt.predict(xx), color="red", linewidth=2)
plt.xlabel("displacement")
plt.ylabel("mpg")
plt.title("Hồi quy")
plt.show()

# In[45]:

from sklearn.tree import export_graphviz
export_graphviz(dt, out_file='tree.dot', feature_names=['displacement'])

# In[46]:
import pydot
graphs = pydot.graph_from_dot_file("tree.dot")
graphs[0].write_png("tree.png")

# In[47]:

from IPython.display import Image
Image(filename='tree.png')




# ## Ví dụ 2

# In[48]:

data = pd.read_csv("./data/autompg.csv")
features = data.columns.values[1:][:-1]

# In[49]:

features

# In[50]:

x = data[features].values

# In[51]:

y = data['mpg']

# In[52]:

regressor = DecisionTreeRegressor(max_depth=3)

# In[53]:

regressor.fit(x, y)

# In[54]:

from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file='tree_multi.dot', feature_names=features)

graphs = pydot.graph_from_dot_file("tree_multi.dot")
graphs[0].write_png("tree_multi.png")

# In[55]:

from IPython.display import Image
Image(filename='tree_multi.png')


# In[56]:

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, regressor.predict(x))

print(mean_absolute_error(y, regressor.predict(x)))

# In[57]:

MSE(y, regressor.predict(x))

print(MSE(y, regressor.predict(x)))