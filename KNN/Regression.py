#!/usr/bin/env python
# coding: utf-8

# # MÔ HÌNH HỒI QUY TUYẾN TÍNH

# # Học máy có nhãn (Supervised learning)
#
# Biến độc lập $X=(X_1,X_2,\dots,X_m)$ và biến phụ thuộc $Y$
#
# **Mục tiêu:** Dự báo biến phụ thuộc khi biết biến độc lập $Y=f(X_1,X_2,\dots,X_m)$
#
# **Phân loại (classification)**: Biến phụ thuộc là biến phân loại (categories)
#
# **Hồi quy (Regression)**: Biến phụ thuộc là liên tục

# In[109]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np


# ## 1. Boston housing data

# In[110]:


boston = pd.read_csv('data/Boston.csv')


# In[111]:


print(boston.head())


# In[112]:


boston=boston.drop('Unnamed: 0',axis=1)


# In[113]:


print(boston.head())

# # Cơ sở hồi quy tuyến tính

# ## Hồi quy tuyến tính đơn giản ( 1 biến độc lập)
#
# $$y=\beta_0+\beta_1 x$$
#
#
# - $y$: biến phụ thuộc
#
# - $x$: biến độc lập
#
# - $\beta_0,\beta_1$: tham số của mô hình
#

# ## Hồi quy tuyến tính nhiều chiều (nhiều biến độc lập)
#
# Hồi quy 2 biến: $y=\beta_0+\beta_1 x_2+\beta_2 x_2$
#
# Hồi quy $m$ biến: $y=\beta_0+\beta_1x_1+\beta_2x_2+\dots+\beta_mx_m$
#
# - $y$: biến phụ thuộc
#
# - $x$: biến độc lập
#
# - $\beta_0,\beta_1,\beta_2,\dots,\beta_m$: tham số của mô hình

# ### Xác định giá trị của tham số mô hình
#
# - Xác định sai số ( hàm tổn thất) của mô hình.
#
# - Chọn tham số để sai số nhỏ nhất.

# ## Mô hình lý thuyết

# Giả sử rằng biến phụ thuộc $Y$ (output, dependent, response) có **quan hệ tuyến tính**
# với các biến đầu vào (independent, predictor) $X_1,X_2,\dots,X_m$ bởi công thức

# $$ Y=\beta_0+\sum\limits_{j=1}^m \beta_j X_j+\varepsilon $$

# trong đó $\varepsilon\sim N(0,\sigma^2)$ biến sai số không quan sát được (**error component**)

# ## Mục tiêu
# Ước lượng các tham số $\beta_j$, phương sai $\sigma^2$, và sự ảnh hưởng các biến đầu vào đối với $Y$.

# Giả sử ta có các $n$ quan sát $$(x_{i1},\dots,x_{im},y_i), i=1,2,\dots,n $$

# $$y_i=\beta_0+\sum\limits_{j=1}^m \beta_j x_{ij}+e_i, i=1,2,\dots,n $$

# với các $e_i$ là các sai số và cùng phân phối với $\varepsilon$

# Ta sử dụng phương pháp **bình phương tối thiểu** ước lượng các $\beta_j$ sao cho sai số nhỏ nhất
# $$SSE=\sum\limits_{i=1}^ne_i^2=\sum\limits_{i=1}^n (y_i-\beta_0-\sum\limits_{j=1}^m \beta_j x_{ij})^2 $$
# $$\hat{\beta} =\arg\min SSE(\beta)$$

# Tổng bình phương các sai số (SSE): $$SSE=\sum\limits_{i=1}^n \hat{e}_i^2=ESS(\hat{\beta}) $$
# Hệ số $R^2$, $$ R^2=1-\dfrac{SSE}{SST}=1-\dfrac{\sum\limits_{i=1}^n (y_i-\hat{y}_i)^2}{\sum\limits_{i=1}^n (y_i-\bar{y})^2}$$

# ## Thực hành với dữ liệu Boston

# In[114]:

boston.head()


# In[115]:


y = boston['medv'].values
x = boston.drop('medv',axis=1).values


# ### Dự báo giá nhà dựa vào một biến

# In[116]:


# xrm=boston['rm']
xrm = x[:, 5]


# In[117]:


xrm = xrm.reshape(-1, 1)
y = y.reshape(-1, 1)


# In[118]:


import matplotlib.pyplot as plt
plt.scatter(xrm,y)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


# In[119]:


from sklearn.linear_model import LinearRegression


# In[120]:


reg = LinearRegression()
print(reg.fit(xrm, y))


# In[121]:


# Hệ số R^2
print(reg.score(xrm, y))


# In[122]:


xx=np.linspace(min(xrm),max(xrm)).reshape(-1,1)
plt.scatter(xrm,y,color="blue")
plt.plot(xx,reg.predict(xx),color="red",linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


# In[123]:


# !pip install yellowbrick
from yellowbrick.regressor import ResidualsPlot


# In[124]:


visualizer = ResidualsPlot(reg, hist=False)


# In[125]:


visualizer.fit(xrm, y)  # Fit the training data to the model
visualizer.score(xrm, y)  # Evaluate the model on the test data
visualizer.poof()                 # Draw/show/poof the data


# In[ ]:





# ## Dự báo giá nhà dựa vào tất cả các biến

# In[126]:


boston.head()


# **Ta chia dữ liệu làm 2 phần: training( 70%) và testing (30%)**

# In[127]:


from sklearn.model_selection import train_test_split


# In[128]:


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.3, random_state=42)


# In[129]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[130]:


y_pred=reg.predict(x_test)


# In[131]:


# Hệ số R^2
reg.score(x_train,y_train)


# In[132]:


from yellowbrick.regressor import ResidualsPlot
viz = ResidualsPlot(reg, hist=False)


# In[133]:


viz.fit(x_train, y_train)  # Fit the training data to the model
viz.score(x_test, y_test)  # Evaluate the model on the test data
viz.poof()


# ## Kiểm tra độ chính xác của mô hình
# - Thông thường ta chia dữ liệu thành hai tập: **tập huấn luyện** và **tập kiểm tra**. Khi xây dựng mô hình ta không được lấy tập kiểm tra để sử dụng.
# - Trong tập huấn luyện ta trích một phần dữ liệu huấn luyện gọi là tập **validation**. Mô hình được kiểm tra thông quan tập validation trên.
# - Vấn đề chọn kích thước tập validation?

# ## Phương pháp cross-validation
# - Chia tập huấn luyện thành $k$ tập con (cùng kích thước), rời nhau.
# - Mỗi lần kiểm tra thử, huấn luyện mô hình với $k-1$ tập và dùng tập còn lại là tập validation.
# - Mô hình cuối cùng được lựa chọn dựa trên sai số huấn luyện và sai số của tập validation.

# In[134]:


from IPython.display import Image
Image('cross-validation.png')


# ** Nhược điểm**
# - Phương pháp của phương pháp cross-validation là số lần thử nghiệm tỷ lệ với số tập chia nhỏ $K$.
# - Người ta đưa ra phương pháp hiệu chỉnh (tránh huấn luyện quá khớp):
#     - Dừng sớm
#     - Thêm số hạng vào hàm mất mát

# # Hiệu chỉnh mô hình hồi quy (Regularized regression)
#
# - Hồi quy tuyến tính cực tiểu hóa hàm tổn thất (loss function)
#
# - Nếu chọn tất cả các biến độc lập
#
# - Các biến độc lập lớn nên các hệ số lớn dẫn đến overfitting
#
# - Hiệu chỉnh: Đưa thêm phần hiệu chỉnh các hệ số.

# ## Ridge Regression
#
# - Hàm tổn thất $$L(\beta)=\sum\limits_{i=1}^n (y_i-\beta_0-\sum\limits_{j=1}^m \beta_j x_{ij})^2+\alpha\sum\limits_{j=0}^m \beta_j^2 $$
#
# - $\alpha$: tham số (cần được xác định)
#
# - Nếu $\alpha=0$: Ta có hồi quy thông thường
#
# - Nếu $\alpha$ lớn: Có thể dẫn tới underfitting

# In[135]:


from sklearn.linear_model import Ridge


# In[136]:


ridge = Ridge(alpha=0.1, normalize=True)


# In[137]:


ridge.fit(x_train, y_train)


# In[138]:


# Hệ số R^2
ridge.score(x_train,y_train)


# In[139]:


ridge_pred = ridge.predict(x_test)


# In[140]:


ridge.score(x_test, y_test)


# In[141]:


import matplotlib.pyplot as plt


# In[142]:


from sklearn.linear_model import RidgeCV
from yellowbrick.regressor import AlphaSelection


# In[143]:


alphas = np.logspace(-10, 1, 400)


# In[144]:


model = RidgeCV(alphas=alphas)
visualizer = AlphaSelection(model)


# In[145]:


y_train=y_train.ravel()


# In[146]:


y_train.shape


# In[147]:


visualizer.fit(x_train, y_train)
visualizer.poof()


# In[ ]:


# ## Lasso regression
#
# - Hàm tổn thất $$L(\beta)=\sum\limits_{i=1}^n (y_i-\beta_0-\sum\limits_{j=1}^m \beta_j x_{ij})^2+\alpha\sum\limits_{j=0}^m |\beta_j| $$
#
# - $\alpha$: tham số (cần được xác định)
#

# In[148]:


from sklearn.linear_model import Lasso


# In[149]:


lasso = Lasso(alpha=0.1, normalize=True)


# In[150]:


lasso.fit(x_train, y_train)
lasso_pred = lasso.predict(x_test)
lasso.score(x_test, y_test)


# In[151]:


import matplotlib.pyplot as plt


# In[152]:


from sklearn.linear_model import LassoCV
from yellowbrick.regressor import AlphaSelection


# In[153]:


alphas = np.logspace(-10, 1, 400)


# In[154]:


model = LassoCV(alphas=alphas)
visualizer = AlphaSelection(model)


# In[155]:


y_train=y_train.ravel()


# In[156]:


visualizer.fit(x_train, y_train)
g = visualizer.poof()


# In[ ]:





# ## Lasso regression for feature selection

# In[157]:


names = boston.columns.values
names


# In[158]:


names=names[0:13]


# In[159]:


names


# In[160]:


lasso_coef=lasso.fit(x,y).coef_
len(lasso_coef)


# In[161]:


len(names)


# In[162]:


plt.plot(range(len(names)), lasso_coef)
plt.xticks(range(len(names)), names, rotation=60)
plt.ylabel('Coefficients')
plt.show()

# # Hồi quy dựa vào K lân cận gần nhất (Kneighbors)

# Cho tập huấn luyện $(x_i,y_i)$. Dự báo giá trị tại mẫu $x$.
#
# - Tìm $k$ lân cận gần nhất với $x$ từ mẫu $x_i$ của tập huấn luyện
#
# - Ký hiệu $N(x)=\{x_{i_1},\dots,x_{i_k}\}$ là tập mẫu tìm được
#
# - Giá trị dự báo của $x$ là $y=f(x)=Average(y_i: x_i\in N(x))$

# In[163]:

plt.scatter(xrm,y)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


# In[164]:


from sklearn.neighbors import KNeighborsRegressor


# In[165]:


reg = KNeighborsRegressor(n_neighbors=1)


# In[166]:


reg.fit(xrm, y)


# In[167]:


xx=np.linspace(min(xrm),max(xrm)).reshape(-1,1)
plt.scatter(xrm,y,color="blue")
plt.plot(xx,reg.predict(xx),color="red",linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


# In[168]:


print("Test set R^2: {:.2f}".format(reg.score(xrm, y)))


# In[169]:


reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(xrm, y)


# In[170]:


xx=np.linspace(min(xrm),max(xrm)).reshape(-1,1)
plt.scatter(xrm,y,color="blue")
plt.plot(xx,reg.predict(xx),color="red",linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


# ## Lựa chọn số $k$ tốt nhất

# In[171]:


from sklearn.model_selection import GridSearchCV


# In[172]:


params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}


# In[173]:


reg = KNeighborsRegressor()


# In[174]:


model = GridSearchCV(reg, params, cv=5)
model.fit(xrm,y)
model.best_params_


# In[175]:


reg = KNeighborsRegressor(n_neighbors=9)
reg.fit(xrm, y)


# In[176]:


xx = np.linspace(min(xrm),max(xrm)).reshape(-1,1)
plt.scatter(xrm,y,color="blue")
plt.plot(xx,reg.predict(xx),color="red",linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


# ## Dự báo giá nhà với tất cả các biến

# In[177]:


reg = KNeighborsRegressor(n_neighbors = 3)


# In[178]:


reg.fit(x_train, y_train)


# In[179]:


reg_pred = reg.predict(x_test)


# In[180]:


reg.score(x_test, y_test)


# # Dự đoán giá bất động sản

# ## 2. Bài toán dự đoán giá bất động sản Montreal

# Dự báo giá nhà dựa trên các thông tin thông tin quan trọng về nhà. Dựa vào các mô hình hồi quy tuyến tính, hồi quy tuyến tính Ridge, hồi quy Laso, và hồi quy k lân cận gần nhất.

# ## Dữ liệu

# In[181]:


import pandas as pd
data=pd.read_csv("data/final_dataDec.csv")


# In[182]:


data.head()


# In[183]:


features=data.columns.values


# In[184]:


data.shape


# ## Xử lý dữ liệu với các giá trị missing
# - Loại bỏ các bản ghi có giá trị missing
# - Khôi phục giá trị mising bằng giá trị trung bình

# In[185]:


len(features)


# In[186]:


features[0] # Thuộc tính not_sold, 1 chưa bán, 0 đã bán


# In[187]:


features[1:14]# Các thuộc tính bán năm 2002-> 2014: 1 năm bán


# In[188]:


features[14:16]# Số giường, năm xây dựng


# In[189]:


features[16:18]# Tọa độ


# In[190]:


features[18:21]# Số phòng, số phòng tắm, diện tích


# In[191]:


features[21:26]# Tính chất bất động sản


# In[192]:


features[26:30]


# In[193]:


features[30:35]


# In[194]:


features[35:39]


# In[195]:


features[39] # Giá tài sản


# Các biến phụ thuộc **39**, biến dự báo **1**

# In[196]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
#from sklearn.select_model import cross_validation
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import normalize


# In[197]:


def loadData(filename):
	data = np.genfromtxt(filename, delimiter=',', dtype=float, skip_header=1)
	return data


# In[198]:


data=loadData("data/final_dataDec.csv")


# In[199]:


def remove_missing_data(pX_train, feature_to_impute):
    X_train =  np.copy(pX_train)
    for i in range(X_train.shape[0]-1, 0, -1):
        for j in range(0, X_train.shape[1], 1):
            if feature_to_impute[j] != 0 and X_train[i, j] == 0:
                X_train = np.delete(X_train, i, 0)
                break
    return X_train


# In[200]:


impute = np.array([0] * len(data[0]))
impute[14] = 2 # num_bed
impute[15] = 2 # year_built
impute[18] = 2 # num_room
impute[19] = 2 # num_bath
impute[20] = 1 # living_space


# In[201]:


data_removal=remove_missing_data(data,impute)


# In[202]:


data_removal.shape


# In[203]:


def mean_imputation_pure(pX_train, feature_to_impute):
	X_train =  np.copy(pX_train)
	for i in range(0, len(feature_to_impute)):
		if feature_to_impute[i] == 0:
			continue
		non_zeros = 0
		for j in range(0, X_train.shape[0]):
			if X_train[j, i] != 0:
				non_zeros += 1
		mean = np.sum(X_train[:, i])/float(non_zeros)
		for j in range(0, X_train.shape[0]):
			if X_train[j, i] == 0:
				X_train[j, i] = mean
	return X_train


# In[204]:


data_imputation=mean_imputation_pure(data,impute)


# In[205]:


data_imputation.shape


# ### Dự báo giá nhà với dữ liệu data_removal (bỏ các bản ghi lỗi)
# - Dữ liệu data_removal
# - Chia dữ liệu thành dữ liệu huấn luyện (70%) và dữ liệu kiểm tra (30%)

# In[206]:


xrm=data_removal[:,:39]## Biến độc lập
yrm=data_removal[:,39] # Biến phụ thuộc


# In[207]:


from sklearn.model_selection import train_test_split
xrm_train, xrm_test, yrm_train, yrm_test = train_test_split(xrm, yrm,test_size = 0.3, random_state=42)


# ** Hồi quy tuyến tính **

# In[208]:


reg_rm=LinearRegression()
reg_rm.fit(xrm_train,yrm_train)


# In[209]:


reg_rm.score(xrm_train,yrm_train)# R^2


# In[210]:


mean_absolute_error(yrm_test,reg_rm.predict(xrm_test))# MAE=sum |y_i-y(x_i)|


# In[ ]:





# **Hồi quy phi tuyến K lân cận gần nhất**

# In[211]:


knnreg_rm = KNeighborsRegressor(n_neighbors=50)
knnreg_rm.fit(xrm_train, yrm_train)


# In[212]:


knnreg_rm.score(xrm_train, yrm_train)


# In[213]:


mean_absolute_error(yrm_test,knnreg_rm.predict(xrm_test))


# In[214]:


mae=[]
for k in range(1,50):
    reg=KNeighborsRegressor(n_neighbors=k)
    reg.fit(xrm_train,yrm_train)
    error=mean_absolute_error(yrm_test,reg.predict(xrm_test))
    mae.append(error)


# In[215]:


import matplotlib.pyplot as plt
plt.plot(mae,c='red')
plt.show()


# In[216]:


print("Optimal k: ", np.argmin(mae)+1)