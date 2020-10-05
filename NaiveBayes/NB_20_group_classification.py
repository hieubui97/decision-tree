# **Nội dung bài thực hành**
#
# Người học tiếp cận và giải quyết bài toán phân lớp văn bản sử dụng phương pháp Naive Bayser. Sau khi thực hành, người học có khả năng:
# 1. Sử dụng được các công cụ của sklearn cho bài toán phân lớp
#
# *   Thao tác với dữ liệu
# *   Chuyển từ văn bản sang không gian vector
#
# 2. Áp dụng được NB cho bài toán phân lớp
#
# *   Huấn luyện mô hình
# *   Đánh gía mô hình
#
# 3. Cải tiến được mô hình phân lớp
# 3. Thực hành được với bài toán thực tế

# **Thao tác với dữ liệu**

# *   Bài thực hành sử dụng dữ liệu tiếng Anh
# *   Dữ liệu gồm 18.000 bài báo được tổ chức trong 20 lớp (classes/groups)
# *   Yêu cầu: xây dựng mô hình phân lớp các bài báo dựa trên mô hình học máy Naive Bayes




# **Đọc dữ liệu**

# In[1]:
import warnings
warnings.filterwarnings('ignore')

# Loading the dateset - training data
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# **Quan sát dữ liệu huấn luyện**

# *   Số lượng dữ liệu huấn luyện
# *   5 văn bản đầu tiên

print("The number of training example:", len(twenty_train.data))

# printing top five training examples
print(twenty_train.data[0:5])

# Danh sách các lớp target_name
# You can check the target names (categories) and some data files by following commands.
# print all the categories
print('\nDanh sách các lớp:\n{}\n'.format(twenty_train.target_names))

# Nhãn của các lớp target
targets = twenty_train.target
print('Nhãn của các lớp:\n{}\n'.format(targets))
print('len(targets): ', len(targets))

# Hiển thị dòng đầu tiên của văn bản đầu tiên
print("\n\n".join(twenty_train.data[0].split("\n")[:3]))


# **Chuẩn bị dữ liệu huấn luyện**
#
# Phần chuẩn bị dữ liệu cho huấn luyện mô hình là việc chuyển các văn bản vào không gian đặc trưng (vector space model)
#
# *   Đặc trưng: sử dụng term frequency (TF) hoặc TF-IDF
# *   Đếm số lần xuất hiện của các từ
# *   Dựa trên bag-of-words model
#
# *   Tạo ma trận term-document, trong đó giá trị ở mỗi ô là số lần xuất hiện của từ trong văn bản chứa nó

# Extracting features from text files CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

print(X_train_counts.shape)
print(X_train_counts[0])

# Biểu diễn văn bản bằng TF-IDF
from  sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf.shape)
print(X_train_tfidf[0])

# **Huấn luyện mô hình**
#
# *   Khởi tạo đối tượng
# *   Huấn luyện mô hình
# *   Giải các tham số của NB

# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.

from  sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


# **Lưu ý khi thiết lập thực nghiệm**
#
# Ở đây, chúng ta sử dụng cách chia dữ liệu làm 2 phần, training-test.
# Chúng ta cũng có thể sử dụng cách k-fold cross-valiation

# **Trực quan hóa quá trình huấn luyện của NB**

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[12]:

estimator = MultinomialNB()
title = "Learning Curves (Naive Bayes)"

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
X, y = X_train_tfidf, twenty_train.target
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.01), cv=cv, n_jobs=8)
plt.show()


# **Đánh giá mô hình trên dữ liệu test**

# In[13]:

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted  = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)

print(np.mean(predicted == twenty_test.target))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(twenty_test.target, predicted)
plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap="Reds")
plt.show()
print(cm)

# **Cải tiến mô hình**
#
# Hiệu quả của mô hình có thể cải tiến bằng nhiều phương pháp, trong đó một trong những phương pháp đơn giản là cải tiến quá trình chuyển từ văn bản sang không gian vector. Trong phần này, mô hình sẽ được cải tiếng bằng cách sử dụng:
#
# *   Loại bỏ các từ dừng
# *   Đưa một từ về từ gốc
#
# *   Khởi tạo mô hình có dùng thêm tham số loại bỏ đi các từ dừng

# In[17]:
# NLTK
# Removing stop words
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])


# *   Sử dụng stemming, và huấn luyện lại mô hình
# *   When downloading NLTK, you mal use *all* to download all packages

# In[20]:
# Stemming Code
import nltk
nltk.download('stopwords')

# In[21]:
print('steming the corpus... Please wait...')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)

predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

np.mean(predicted_mnb_stemmed == twenty_test.target)

print(np.mean(predicted_mnb_stemmed == twenty_test.target))



# **So sánh với phương pháp phân loại SVM**

# In[22]:
# Training Support Vector Machines - SVM and calculating its performance
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter_no_change=5,
                                                   random_state=42))])

text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)

predicted_svm = text_clf_svm.predict(twenty_test.data)

np.mean(predicted_svm == twenty_test.target)

print(np.mean(predicted_svm == twenty_test.target))


# **Trực quan hoá quá trình huấn luyện của NB và SVM**

# In[23]:

estimator = MultinomialNB()
title = "Learning Curves (Naive Bayes)"

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
X, y = X_train_tfidf, twenty_train.target
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 0.95), cv=cv, n_jobs=8)

#from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
title = "Learning Curves (SVM, linear kernel)"

# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

#estimator = SVC(gamma=0.001)

estimator = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter_no_change=5, random_state=42, verbose=0)

plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.1), cv=cv, n_jobs=8)

plt.show()


# **Sử dụng GridSearch để tìm tham số phù hợp**
#
# Có thể sử dụng thuật toán GridSearch để tìm tham số phù hợp--> tăng độ tốt của mô hình.
# Tuy nhiên, thuật toán này có nhược điểm là tốc độ chậm nên có thể phù hợp với bộ dữ liệu nhỏ.
# Với bộ dữ liệu lớn thuật toán chạy trong thời gian lâu

# In[24]:

# Grid Search
# Here, we are creating a list of parameters for which we would like to do performance tuning.
# All the parameters name start with the classifier name (remember the arbitrary name we gave).
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}


# In[25]:

# Next, we create an instance of the grid search by passing the classifier, parameters
# and n_jobs=-1 which tells to use multiple cores from user machine.

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)


# In[26]:

# To see the best mean score and the params, run the following code

gs_clf.best_score_
gs_clf.best_params_

print(gs_clf.best_score_)
print(gs_clf.best_params_)

# Output for above should be: The accuracy has now increased to ~90.6% for the NB classifier (not so naive anymore! 😄)
# and the corresponding parameters are {‘clf__alpha’: 0.01, ‘tfidf__use_idf’: True, ‘vect__ngram_range’: (1, 2)}.


# In[27]:

# Similarly doing grid search for SVM
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)


gs_clf_svm.best_score_
gs_clf_svm.best_params_

print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)