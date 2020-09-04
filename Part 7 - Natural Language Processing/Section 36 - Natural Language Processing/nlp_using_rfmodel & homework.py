# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # NATURAL LANGUAGE PROCESSiNG (using random forest)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %%
dts = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# %% [markdown]
# ## Cleaning the text

# %%
import re
import nltk
nltk.download("stopwords")
#from nltk.corpus import stopwords        #stopwords means removing words, such as the, is, at, which etc. as they don't effect reviews 
from nltk.stem.porter import PorterStemmer      #steming means simplifying words like converting loved to love as both are +ve review
corpus = []
for i in range(len(dts)):
    review = re.sub('[^a-zA-Z]', ' ', dts['Review'][i])     # ^ means not, everything thats not a-z& A-Z like "!':" punctuations remove
    review = review.lower()     
    review = review.split()
    ps = PorterStemmer()                      # steming to optimize the dimentionality of sparse matrix
    #all_stopwords = stopwords.words('english')
    #all_stopwords.remove("not")
    #review =[ps.stem(word) for word in review if not word in set(all_stopwords)]    # ""
    review = " ".join(review)
    corpus.append(review)

# %% [markdown]
# ## Creating the Bag of Words
# this process is also called Tokenization

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dts.iloc[:, -1].values


# %%
x

# %% [markdown]
# ## Spliting data to train set and test set

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x ,y ,random_state = 42, test_size= 0.25)

# %% [markdown]
# ## Training data to Random forest

# %%
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0, n_estimators=10, criterion='entropy')
clf.fit(x_train, y_train)

# %% [markdown]
# ## Predicting test set results

# %%
y_pred = clf.predict(x_test)
res = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print(res)

# %% [markdown]
# ## making confusion matrix

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# %% [markdown]
# ## Homework Challenge
# Evaluate the performance of each of these models. Try to beat the Accuracy obtained in the tutorial. But remember, 
#Accuracy is not enough, so you should also look at other performance metrics like Precision (measuring exactness), 
#Recall (measuring completeness) and the F1 Score (compromise between Precision and Recall). Please find below these metrics 
#formulas (TP = # True Positives, TN = # True Negatives, FP = # False Positives, FN = # False Negatives):
# 
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# 
# Precision = TP / (TP + FP)
# 
# Recall = TP / (TP + FN)
# 
# F1 Score = 2 * Precision * Recall / (Precision + Recall)

# %%
tn, fp, fn, tp = cm.ravel()


# %%
accu = (tn + tp) / (tn + tp + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_sc = 2 * precision * recall / (precision + recall)
print("True Positives: {} & True Negatives: {}".format(tp, tn))
print("False Positives: {} & False Negatives: {}".format(fp, fn))
print("Accuracy: {}%".format(accu*100))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1 Score: {}".format(f1_sc))


# %%
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)
