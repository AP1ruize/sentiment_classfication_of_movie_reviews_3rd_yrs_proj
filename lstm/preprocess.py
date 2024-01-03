import os
import string
import numpy as np

# functions for data pre-process
from keras_preprocessing.text import maketrans
# for remove html tags in text
import re

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

# 返回包含在该路径下的所有文件名的list
# return a list that contains all file names in given path
def allFileName_in_dir(filedir):
    allFileNameList = []
    files = os.listdir(filedir)
    for each_file in files:
        allFileNameList.append(os.path.join(filedir, each_file))
    return allFileNameList


# 吧文本中所有符号转换为空格，并将大写转小写并返回转换后的string
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


train_pos_data_dir = "aclImdb\\train\\pos"
train_neg_data_dir = "aclImdb\\train\\neg"
# test_pos_data_dir = "aclImdb\\test\\pos"
# test_neg_data_dir = "aclImdb\\test\\neg"

gloveFilePath_50d = "glove.6B\\glove.6B.50d.txt"
maxSeqLength = 250


# 读取指定目录下所有文件内容，存入list，并打上给定标签
# read all files in given path, return a list of labelled comments
def aggregateAllData(filedir: string, label: int):
    files = allFileName_in_dir(filedir)
    theGreatList = []
    for filename in files:
        fopen = open(filename, 'r', encoding='UTF-8')
        theSmallList = [label, ""]
        for eachLine in fopen:
            theSmallList[1] += eachLine
        fopen.close()
        theGreatList.append(theSmallList)
    return theGreatList


# 读取glove生成wordsList和wordVectors
# read glove file and generate wordsList and wordVectors, then save to file
def generateWordVecFiles(gloveFilePath: string):
    embeddings_dict = {}
    with open(gloveFilePath, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    wordsList = np.array(list(embeddings_dict.keys()))
    wordVectors = np.array(list(embeddings_dict.values()), dtype='float32')

    np.save('wordsList.npy', wordsList)
    np.save('wordVectors.npy', wordVectors)
    print('generating wordsList.npy done: ')
    print(wordsList.shape)
    print('generating wordVectors.npy done: ')
    print(wordVectors.shape)


# 对于给定的2维list 生成并返回embedding matrix
# 由generateEmbeddingMatrix_withGlove调用
# use given 2d list to generate and return embedding matrix
# this fuction will be called by generateEmbeddingMatrix_withGlove
def generateMatrix4EachReview(theGreatList: list, wordsList: list, wordVectors, maxSeqLen):
    matrix = np.zeros((len(theGreatList), maxSeqLen), dtype='int32')
    reviewCounter = 0
    for review in theGreatList:
        if reviewCounter < 1000:
            if reviewCounter % 200 == 0:
                print('generating matrix, reviewCounter: '+str(reviewCounter))
        elif reviewCounter % 1500 == 0:
            print('generating matrix, reviewCounter: '+str(reviewCounter))
        text = review[1].split()
        indexCounter = 0
        # textVec = np.zeros(maxSeqLen, dtype='int32')
        for word in text:
            if indexCounter >= 250:
                break
            try:
                matrix[reviewCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                matrix[reviewCounter][indexCounter] = 399999
                # Vector for unknown words
            indexCounter += 1
        reviewCounter += 1
    return matrix


# 生成4个二维list的embedding matrix 并存档
# generate 4 2-d embedding matrices and save to file
def generateEmbeddingMatrix_withGlove(posTrainList: list, negTrainList: list, maxSeqLen):
    wordsList = np.load("wordsList.npy")
    wordVectors = np.load("wordVectors.npy")
    wordsList = wordsList.tolist()
    wordsList = [word.encode('UTF-8') for word in wordsList]

    print('start generating matrices')

    posTrainMatrix = generateMatrix4EachReview(posTrainList, wordsList, wordVectors, maxSeqLen)
    print('posTrainMatrix generated.')
    np.save('posTrainMatrix.npy', posTrainMatrix)
    print('posTrainMatrix: ')
    print(posTrainMatrix.shape)

    negTrainMatrix = generateMatrix4EachReview(negTrainList, wordsList, wordVectors, maxSeqLen)
    print('negTrainMatrix generated.')
    np.save('negTrainMatrix.npy', negTrainMatrix)
    print('negTrainMatrix: ')
    print(negTrainMatrix.shape)

    # posTestMatrix = generateMatrix4EachReview(posTestList, wordsList, wordVectors, maxSeqLen)
    # print('posTestMatrix generated.')
    # np.save('posTestMatrix', posTestMatrix)
    # print('posTestMatrix: ')
    # print(posTestMatrix.shape)
    #
    # negTestMatrix = generateMatrix4EachReview(negTestList, wordsList, wordVectors, maxSeqLen)
    # print('negTestMatrix generated.')
    # np.save('negTestMatrix', negTestMatrix)
    # print('negTestMatrix: ')
    # print(negTestMatrix.shape)



# 调用上述函数
# call all functions defined above
def preprocess():
    # gather dataset
    posTrainList = aggregateAllData(train_pos_data_dir, 1)
    negTrainList = aggregateAllData(train_neg_data_dir, 0)
    # posTestList = aggregateAllData(test_pos_data_dir, 1)
    # negTestList = aggregateAllData(test_neg_data_dir, 0)
    print('aggregate data finished')

    # read glove file and generate wordsList and wordVectors, then save to file
    generateWordVecFiles(gloveFilePath_50d)
    # generate 4 2-d embedding matrices and save to file
    generateEmbeddingMatrix_withGlove(posTrainList, negTrainList, maxSeqLength)


preprocess()
