# coding: utf-8

# # CÂY QUYẾT ĐỊNH (Decision Tree)

# # Cây quyết định

# Cây quyết định là phương pháp chia không gian của dữ liệu thành các hình chữ nhật rời nhau, và sử dụng giá trị để phù hợp cho mỗi hình chữ nhật.

# Giả sử không gian của dữ liệu là $X$ ta chia thành các hình chữ nhật $\{R_m\}$ rời nhau và $$X=\bigcup\limits_{m} R_m $$

# Trong phần này ta mô tả thuật toán CART để xây dựng tập các hình chữ nhật $\{R_m\}$. Thuật toán trên có thể áp dụng cho bài toán **Phân loại (classification) ** và bài toán ** Hồi quy (regression)**

# Giả sử không gian dữ liệu $X=\mathbb{R}^2$ với hai thuộc tính $(X_1,X_2)$ và biến phụ thuộc $Y=f(X_1,X_2)$.


# Classification

# In[1]:
# Hiển thị ảnh
from IPython.display import Image

# In[2]:
Image('fig_1.png')
# In[3]:
Image('fig_2.png')
# In[4]:
Image('fig_3.png')
# In[5]:
Image('fig_4.png')
# In[6]:
Image('fig_5.png')
# In[7]:
Image('fig_6.png')
# In[8]:
Image('fig_7.png')

# PHÂN LOẠI DỮ LIỆU DỰA VÀO CÂY QUYẾT ĐỊNH

# Ví dụ phân loại dữ liệu breast cancer

# In[9]:
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer


# In[10]:
# cancer

cancer = load_breast_cancer()

# In[11]:
# cancer.keys

print('cancer.keys():\n {}'.format(cancer.keys()))

# In[12]:
# kích thước dữ liệu data.shape

print('Kích thước dữ liêu: \n {}'.format(cancer.data.shape))

# In[13]:
# các thuộc tính feature_names

print('Các thuộc tính:\n {}'.format(cancer.feature_names))

# In[14]:
# các lớp target_names

print('Các lớp:\n {}'.format(cancer.target_names))


# ** Chia dữ liệu **
# - Chia dữ liệu thành:
#     - Dữ liệu huấn luyện: 80%
#     - Dữ liệu kiểm tra: 20%

# In[15]:
# chia dữ liệu
from  sklearn.model_selection import train_test_split
x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# In[16]: x_train.shape, x_test.shape
print('x_train : {}'.format(x_train.shape))

# In[17]:
print('x_test : {}'.format(x_test.shape))

# In[18]:
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import accuracy_score
from sklearn.metrics import accuracy_score

# In[19]:
# Cây quyết định
tree = DecisionTreeClassifier(random_state=42)

# In[20]:
print(tree.fit(x_train, y_train))

# In[21]:
y_pred  = tree.predict(x_test)

# In[22]:
print('Độ chính xác của tập huấn luyện: {:.4f}'.format(tree.score(x_train, y_train)))
print('Độ chính xác của tập kiểm tra: {:.4f}'.format(tree.score(x_test, y_test)))

# In[23]:
# Biểu thị cây phân loại
from  sklearn.tree import export_graphviz
import graphviz

# In[24]:
export_graphviz(tree, out_file='tree_classifier.dot', feature_names=cancer.feature_names, class_names=cancer.target_names, impurity=False, filled=True)

# In[25]:
# Chuyển file dot sang file ảnh
import pydot
graphs = pydot.graph_from_dot_file("tree_classifier.dot")
graphs[0].write_png("tree_classifier.png")

# In[26]:
# Hiển thị hình ảnh
from IPython.display import Image
Image(filename='./tree_classifier.png')


# In[27]:
# Mức độ quan trọng của các thuộc tính
print('Mức độ quan trọng của các thuộc tính: \n {}'.format(tree.feature_importances_))

# In[28]:
# Import matplotlib
import matplotlib.pyplot as plt

# In[29]: features
features = cancer.feature_names
n = len(features)
plt.figure(figsize=(8,10))
plt.barh(range(n), tree.feature_importances_)
plt.yticks(range(n), features)
plt.title('Mức độ quan trọng các thuộc tính')
plt.ylabel('Các thuộc tính')
plt.xlabel('Mức độ')
plt.show()

