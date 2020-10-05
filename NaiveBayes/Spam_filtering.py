#!/usr/bin/env python
# coding: utf-8

# Dữ liệu trong ví dụ này được lấy trong [Exercise 6: Naive Bayes - Machine Learning - Andrew Ng](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex6/ex6.html).
#
# Trong ví dụ này, dữ liệu đã được xử lý, và là một tập con của cơ sở dữ liệu [Ling-Spam Dataset](http://csmining.org/index.php/ling-spam-datasets.html.
#
# #### Mô tả dữ liệu
# Tập dữ liệu này bao gồm tổng cộng 960 emails tiếng Anh, được tách thành tập training và test theo tỉ lệ 700:260, 50% trong mỗi tập là các spam emails.
#
# Dữ liệu trong cơ sở dữ liệu này đã được xử lý khá đẹp. Các quy tắc xử lý như sau:
#
# 1. **Loại bỏ _stop words_**: Những từ xuất hiện thường xuyên trong bất kỳ văn bản nào như 'and', 'the', 'of', ... được loại bỏ.
#
# 2. **Lemmatization**: Những từ có cùng 'gốc' được đưa về cùng loại. Ví dụ, 'include', 'includes', 'included' đều được đưa chung về 'include'. Tất cả các từ cũng đã được đưa về dạng ký tự thường (không phải HOA).
#
# 3. **Loại bỏ _non-words_**: Số, dấu câu, ký tự 'tabs', ký tự 'xuống dòng' đã được loại bỏ.
#
# Dưới đây là một ví dụ của 1 email không phải spam, **trước khi được xử lý**:
# ```
# Subject: Re: 5.1344 Native speaker intuitions
#
# The discussion on native speaker intuitions has been extremely interesting, but I worry that my brief intervention may have
# muddied the waters. I take it that there are a number of
# separable issues. The first is the extent to which a native
# speaker is likely to judge a lexical string as grammatical
# or ungrammatical per se. The second is concerned with the
# relationships between syntax and interpretation (although even
# here the distinction may not be entirely clear cut).
# ```
#
# và **sau khi được xử lý**:
# ```
# re native speaker intuition discussion native speaker intuition
# extremely interest worry brief intervention muddy waters number
# separable issue first extent native speaker likely judge lexical
# string grammatical ungrammatical per se second concern relationship
# between syntax interpretation although even here distinction entirely clear cut
# ```
#
# Và đây là một ví dụ về **spam email sau khi được xử lý**:
# ```
# financial freedom follow financial freedom work ethic extraordinary desire earn least per month work home special skills experience required train personal support need ensure success legitimate homebased income opportunity put back control finance life ve try opportunity past fail live promise
# ```
# Chúng ta thấy rằng trong đoạn này có các từ như: financial, extraordinary, earn, opportunity, … là những từ thường thấy trong các email spam.
#
# Trong ví dụ này, chúng ta sẽ sử dụng Multinomial Naive Bayes.
#
# Để cho bài toán được đơn giản hơn, tôi tiếp tục sử dụng dữ liệu đã được xử lý, có thể được download ở đây: [ex6DataPrepared.zip](http://openclassroom.stanford.edu/MainFolder/courses/MachineLearning/exercises/ex6materials/ex6DataPrepared.zip). Trong folder sau khi giải nén, chúng ta sẽ thấy các files:
# ```
# test-features.txt
# test-labels.txt
# train-features-50.txt
# train-features-100.txt
# train-features-400.txt
# train-features.txt
# train-labels-50.txt
# train-labels-100.txt
# train-labels-400.txt
# train-labels.txt
# ```
#
# tương ứng với các file chứa dữ liệu của tập training và tập test. File `train-features-50.txt` chứa dữ liệu của tập training thu gọn với chỉ có tổng 50 training emails.
#
# Mỗi file `*labels*.txt` chứa nhiều dòng, mỗi dòng là một ký tự 0 hoặc 1 thể hiện email là _non-spam_ hoặc _spam_.
#
# Mỗi file `*features*.txt` chứa nhiều dòng, mỗi dòng có 3 số, ví dụ:
#
# ```
# 1 564 1
# 1 19 2
# ```
# trong đó số đầu tiên là chỉ số của email, bắt đầu từ 1; số thứ 2 là thứ tự của từ trong từ điển (tổng cộng 2500 từ); số thứ 3 là số lượng của từ đó trong email đang xét. Dòng đầu tiên nói rằng trong email thứ nhất, từ thứ 564 trong từ điển xuất hiện 1 lần. Cách lưu dữ liệu như thế này giúp tiết kiệm bộ nhớ vì 1 email thường không chứa hết tất cả các từ trong từ điển mà chỉ chứa một lượng nhỏ, ta chỉ cần lưu các giá trị khác không.
#
# Nếu ta biểu diễn feature vector của mỗi email là 1 vector hàng có độ dài bằng độ dài từ điển (2500) thì dòng thứ nhất nói rằng thành phần thứ 564 của vector này bằng 1. Tương tự, thành phần thứ 19 của vector này bằng 1. Nếu không xuất hiện, các thành phần khác được mặc định bằng 0.
#
# Dựa trên các thông tin này, chúng ta có thể tiến hành lập trình với thư viện sklearn.
#
# **Khai báo thư viện và đường dẫn tới file**

# In[1]:


## packages
from __future__ import print_function
from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy.sparse import coo_matrix  # for sparse matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score  # for evaluating results


# data path and file name
path = 'ex6DataPrepared/'
train_data_fn = 'train-features.txt'
test_data_fn = 'test-features.txt'
train_label_fn = 'train-labels.txt'
test_label_fn = 'test-labels.txt'

# Hàm số đọc dữ liệu từ file `data_fn` với labels tương ứng `label_fn`. Chú ý rằng [số lượng từ trong từ điển là 2500](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex6/ex6.html).
#
# Dữ liệu sẽ được lưu trong một ma trận mà mỗi hàng thể hiện một email. Ma trận này là ma trận sparse nên chúng ta sẽ sử dụng hàm [`scipy.sparse.coo_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html).

# In[2]:


nwords = 2500


def read_data(data_fn, label_fn):
    ## read label_fn
    with open(path + label_fn) as f:
        content = f.readlines()
    label = [int(x.strip()) for x in content]

    ## read data_fn
    with open(path + data_fn) as f:
        content = f.readlines()
    # remove '\n' at the end of each line
    content = [x.strip() for x in content]

    dat = np.zeros((len(content), 3), dtype=int)

    for i, line in enumerate(content):
        a = line.split(' ')
        dat[i, :] = np.array([int(a[0]), int(a[1]), int(a[2])])

    # remember to -1 at coordinate since we're in Python
    # check this: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    # for more information about coo_matrix function
    data = coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)), shape=(len(label), nwords))
    return (data, label)


# Đọc training data và test data, sử dụng class `MultinomialNB` trong sklearn để xây dựng mô hình và dự đoán đầu ra cho test data.

# In[9]:


(train_data, train_label) = read_data(train_data_fn, train_label_fn)
(test_data, test_label) = read_data(test_data_fn, test_label_fn)

clf = MultinomialNB()
clf.fit(train_data, train_label)

y_pred = clf.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' % (train_data.shape[0], accuracy_score(test_label, y_pred) * 100))

# Thử với các bộ dữ liệu training nhỏ hơn:

# In[10]:


train_data_fn = 'train-features-100.txt'
train_label_fn = 'train-labels-100.txt'
test_data_fn = 'test-features.txt'
test_label_fn = 'test-labels.txt'

(train_data, train_label) = read_data(train_data_fn, train_label_fn)
(test_data, test_label) = read_data(test_data_fn, test_label_fn)
clf = MultinomialNB()
clf.fit(train_data, train_label)
y_pred = clf.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' % (train_data.shape[0], accuracy_score(test_label, y_pred) * 100))

# In[11]:


train_data_fn = 'train-features-50.txt'
train_label_fn = 'train-labels-50.txt'
test_data_fn = 'test-features.txt'
test_label_fn = 'test-labels.txt'

(train_data, train_label) = read_data(train_data_fn, train_label_fn)
(test_data, test_label) = read_data(test_data_fn, test_label_fn)
clf = MultinomialNB()
clf.fit(train_data, train_label)
y_pred = clf.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' % (train_data.shape[0], accuracy_score(test_label, y_pred) * 100))

# Ta thấy rằng thậm chí khi tập training là rất nhỏ, 50 emails tổng cộng, kết quả đạt được đã rất ấn tượng.
#
# Nếu bạn muốn tiếp tục thử mô hình `BernoulliNB`:

# In[12]:


clf = BernoulliNB(binarize=.5)
clf.fit(train_data, train_label)
y_pred = clf.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' % (train_data.shape[0], accuracy_score(test_label, y_pred) * 100))

# Ta thấy rằng trong bài toán này, `MultinomialNB` hoạt động hiệu quả hơn.

# In[13]:


import numpy as np

d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
train_data = np.array([d1, d2, d3, d4])
label = np.array([0, 0, 0, 1])

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(train_data, label)

print(clf.predict(d5))

print(clf.predict_proba(d6))

# In[14]:


# from __future__ import print_function
from sklearn.naive_bayes import BernoulliNB
import numpy as np

# train data
d1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N'])  # 0 - B, 1 - N

# test data
d5 = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

## call MultinomialNB
clf = BernoulliNB()
# training
clf.fit(train_data, label)

# test
print('Predicting class of d5:', str(clf.predict(d5)[0]))
print('Probability of d6 in each class:', clf.predict_proba(d6))

