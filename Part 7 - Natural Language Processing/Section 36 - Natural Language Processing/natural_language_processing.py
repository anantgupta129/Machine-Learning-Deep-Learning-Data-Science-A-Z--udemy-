# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # NATURAL LANGUAGE PROCESSiNG

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
from nltk.corpus import stopwords        #stopwords means removing words, such as the, is, at, which etc. as they don't effect reviews 
from nltk.stem.porter import PorterStemmer      #steming means simplifying words like converting loved to love as both are +ve review
corpus = []
for i in range(len(dts)):
    review = re.sub('[^a-zA-Z]', ' ', dts['Review'][i])     # ^ means not, everything thats not a-z& A-Z like "!':" punctuations remove
    review = review.lower()     
    review = review.split()
    ps = PorterStemmer()                      # steming to optimize the dimentionality of sparse matrix
    all_stopwords = stopwords.words('english')
    all_stopwords.remove("not")
    review =[ps.stem(word) for word in review if not word in set(all_stopwords)]    # ""
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
# ## Training data to Naive Bayes model

# %%
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
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
# ## Predicting if a single review is positive or negative
# %% [markdown]
# ### Positive review
# Use our model to predict if the following review:
# 
# "I love this restaurant so much"
# 
# is positive or negative.
# 

# %%
new_review = "I love this restaurant so much"
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()     
new_review = new_review.split()
ps = PorterStemmer()                    
all_stopwords = stopwords.words('english')
all_stopwords.remove("not")
new_review =[ps.stem(word) for word in new_review if not word in set(all_stopwords)] 
new_review = " ".join(new_review)
new_corpus = [new_review]
new_X = cv.transform(new_corpus).toarray()
new_pred = clf.predict(new_X)
print(new_pred)

# %% [markdown]
# ### Negative review
# Use our model to predict if the following review:
# 
# "I hate this restaurant so much"
# 
# is positive or negative.
# 

# %%
new_review = 'I hate this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()     
new_review = new_review.split()
ps = PorterStemmer()                    
all_stopwords = stopwords.words('english')
all_stopwords.remove("not")
new_review =[ps.stem(word) for word in new_review if not word in set(all_stopwords)] 
new_review = " ".join(new_review)
new_corpus = [new_review]
new_X = cv.transform(new_corpus).toarray()
new_pred = clf.predict(new_X)
print(new_pred)

