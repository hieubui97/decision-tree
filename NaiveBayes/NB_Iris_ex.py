# Ví dụ phân lớp hoa  Iris bằng NB

# Import các thư viện cần thiết từ sklearn

# 1. datasets: chứa các tập dữ liệu
# 2. metrics: dùng để tính toán các độ đo
# 3. GaussianNB: thư viện phân lớp với NB của sklearn


# In[1]:
# Gausian Naive Bayes
from  sklearn import datasets
from  sklearn import metrics
from  sklearn.naive_bayes import GaussianNB

# In[2]:
# Load dữ liệu và hiển thị 6 ảnh đầu tiên
# Load the iris datasets
dataset = datasets.load_iris()
print('The Iris datasets:\n {}\n'.format(dataset.data[0:6]))

# In[3]:
# In ra giá trị lớp chuẩn của 100 ảnh đầu tiên
expected = dataset.target[0:100]
print('expected: \n {}]\n'.format(expected))

# In[4]:
# Fit dữ liệu vào mô hình và huấn luyện mô hình
# fit a Naive Bayes model to the data

model = GaussianNB()
model.fit(dataset.data, dataset.target)
print('model:\n {}\n'.format(model))

# In[5]:
# Dự đoán
# make predictions
expected = dataset.target
print('expected:\n{}\n'.format(expected))

predicted = model.predict(dataset.data)
print('predicted:\n{}\n'.format(predicted))

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))


# In[6]:
# In ra ma trận nhập nhằng (confusion matrix)
print('confusion matrix:')
print(metrics.confusion_matrix(expected, predicted))