#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and defining text files

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Defining files as variables and splitting the terms into bags of words

# In[5]:


file1 = open("Text Files/abstract_1_test")
file2 = open("Text Files/abstract_2_test")
file3 = open("Text Files/abstract_3_test")
file1_data = file1.read()
file2_data = file2.read()
file3_data = file3.read()

bagOfWords1 = file1_data.split(' ')
bagOfWords2 = file2_data.split(' ')
bagOfWords3 = file3_data.split(' ')


# We combine the common words using the union method to eliminate repetitions and consider them as unique words

# In[6]:


uniqueWords = set(bagOfWords1).union(set(bagOfWords2)).union(set(bagOfWords3))
print(uniqueWords)


# For each unique words, we traverse through the bag of words to find the frequency of the terms between the two documents

# In[48]:


numOfWords1 = dict.fromkeys(uniqueWords, 0)
for word in bagOfWords1:
    numOfWords1[word] += 1
numOfWords2 = dict.fromkeys(uniqueWords, 0)
for word in bagOfWords2:
    numOfWords2[word] += 1
numOfWords3 = dict.fromkeys(uniqueWords, 0)
for word in bagOfWords3:
    numOfWords3[word] += 1


# Printing the frequency of the terms

# In[49]:


print(numOfWords1)
print(numOfWords2)
print(numOfWords3)


# Importing NLTK libraries to eliminate stopwords (Examples shown below)

# In[50]:


from nltk.corpus import stopwords
stopwords.words('english')


# # Term frequency calculation

# To calculate the frequency of a term, we fetch the number of times the word appear in a document divided by the number of words in the document.

# In[10]:


def tf(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


# In[ ]:


tf1 = tf(numOfWords1, bagOfWords1)
tf2 = tf(numOfWords2, bagOfWords2)
tf3 = tf(numOfWords3, bagOfWords3)


# In[51]:


print(tf1)
print(tf2)
print(tf3)


# # Inverse Data Frequency

# The log of the number of documents divided by the number of documents that contain the specified word. This inverse data frequency determines the weight of rare words across all documents in the file.

# In[13]:


def idf(docs):
    import math
    N = len(docs)

    idfDict = dict.fromkeys(docs[0].keys(), 0)
    for doc in docs:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


# In[23]:


idfs = idf([numOfWords1, numOfWords2, numOfWords3])
print(idfs)


# # Multiplying the Term Frequency and Inverse Data frequency

# By multiplying the term frequency and the inverse data frequency, we get the weight of each term

# In[24]:


def tfidf(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


# In[25]:


tfidf1 = tfidf(tf1, idfs)
tfidf2 = tfidf(tf2, idfs)
tfidf3 = tfidf(tf3, idfs)
df = pd.DataFrame([tfidf1, tfidf2, tfidf3])


# In[27]:


df


# In[28]:


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([file1_data, file2_data, file3_data])
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
print(vectors.shape)


# df

# In[31]:


query = denselist
query_vec = vectorizer.transform([query])
results = cosine_similarity(vectors, query_vec).reshape((-1))

