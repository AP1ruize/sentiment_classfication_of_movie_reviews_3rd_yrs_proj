import numpy as np
import os

import nltk.stem
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn import tree
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
# from string import punctuation
# import graphviz
import string
from collections import Counter
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from keras_preprocessing.text import maketrans
import re

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

# return names of all files in the given path
def allFileName_in_dir(filedir):
    allFileNameList = []
    files = os.listdir(filedir)
    for each_file in files:
        allFileNameList.append(os.path.join(filedir, each_file))
    return allFileNameList

# read all files in given path, return a list of labelled comments
def aggregateAllData(filedir: string, label: int):
    files = allFileName_in_dir(filedir)
    theGreatList = []
    for filename in files:
        fopen = open(filename, 'r', encoding='UTF-8')
        theSmallList = [""]
        for eachLine in fopen:
            theSmallList[0] += eachLine
        fopen.close()
        theGreatList.append(theSmallList)
    return theGreatList

# text cleaning, convert punctuation marks to space, convert uppercase letters to lowercase, return converted string
def clean_text(raw_text: string):
    raw_text=remove_tags(raw_text)
    translation = maketrans(string.punctuation + string.ascii_uppercase,
                            " " * len(string.punctuation) + string.ascii_lowercase)
    text = raw_text.translate(translation)
    # words=text.split();
    # filtered_words=[word for word in words if word not in stopwords.words('english')]
    # filtered_text=""
    # for eachWord in filtered_words:

    return text


train_pos_data_dir='aclImdb\\train\\pos'
train_neg_data_dir='aclImdb\\train\\neg'
test_pos_data_dir='aclImdb\\test\\pos'
test_neg_data_dir='aclImdb\\test\\neg'
sizeDataset=12500

posTrainList = aggregateAllData(train_pos_data_dir, 1)
# posTrainList=posTrainList[0:sizeDataset]
negTrainList = aggregateAllData(train_neg_data_dir, 0)
# negTrainList=negTrainList[0:sizeDataset]
posTestList = aggregateAllData(test_pos_data_dir, 1)
negTestList = aggregateAllData(test_neg_data_dir, 0)

posTrainList.extend(negTrainList)
posTrainList.extend(posTestList)
posTrainList.extend(negTestList)
cVec = posTrainList
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'
#os.environ["PATH"]+=os.pathsep+'C:\Program Files\Graphviz\bin'
s = nltk.stem.SnowballStemmer('english')
print("1")
cVec1=[]
#去除停用词,提取词干
# step 1: text cleaning, remove stop words, tokenize text, stemming
for ii in cVec:
    for i in ii:
        tokens = nltk.word_tokenize(clean_text(i))
        str=''
        for ws in tokens:
            if ws not in stopwords.words('english'):
                str=str+s.stem(ws)
                str=str+' '
        cVec1.append(str)

cVec=cVec1
print("2")
# step 2: train/test split
# label text
labelList = []
for i in range(1, sizeDataset + 1):
    labelList.append(1)
for i in range(1, sizeDataset + 1):
    labelList.append(0)
for i in range(1, sizeDataset + 1):
    labelList.append(1)
for i in range(1, sizeDataset + 1):
    labelList.append(0)


print(len(cVec))
print(len(labelList))
x_train, x_test, y_train, y_test = train_test_split(cVec, labelList, test_size=0.3, random_state=7)
print('split data finished.')
print("3")
# step 3: feature extraction, use sklearn.CountVectorizer to convert a collection of text documents to a matrix of token counts
count_vec=CountVectorizer(binary=True) 
x_train=count_vec.fit_transform(x_train)
x_test=count_vec.transform(x_test)
print("4")
#第四步：构建决策树
# step 4: build the decision tree classifier
print('##############################')
    # print('max_depth: %d'% i)
dtc=tree.DecisionTreeClassifier(max_depth=12)


# 70/30 train/test split
dtc.fit(x_train,y_train)
y_true=y_test
y_pred=dtc.predict(x_test)
print('classification report on test set:')
print(classification_report(y_true,y_pred))
print('在测试集上的准确率(accuracy on test set)：%.2f'% accuracy_score(y_true,y_pred))


# #第五步：画决策树
# # step 5: plot the decision tree
# cwd=os.getcwd()
# dot_data=tree.export_graphviz(dtc
#                               ,out_file=None
#                               ,feature_names=count_vec.get_feature_names())
# graph=graphviz.Source(dot_data)
# graph.format='svg'
# graph.render(cwd+'/tree',view=True)
