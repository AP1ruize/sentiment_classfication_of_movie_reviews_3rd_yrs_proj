import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# from tensorflow.python.framework import ops
import numpy as np
from random import randint
import datetime

wordVectors = np.load('wordVectors.npy')

batchSize = 20
lstmUnits = 64
numClasses = 2
iterations = 50000
maxSeqLength = 250
numDimensions = 50

sampleAmt = 5000
delta = 12500


def combineMatrixs():
    mtrx1 = np.load('posTrainMatrix.npy')
    mtrx2 = np.load('negTrainMatrix.npy')
    # mtrx3 = np.load('negTrainMatrix3.npy')
    mtrx = np.vstack((mtrx1, mtrx2))
    # mtrx = np.vstack((mtrx, mtrx3))
    return mtrx


ids = combineMatrixs()


# 训练11500 测试1000, 11500 for training, 1000 for test
# get a batch of training data
def get_train_batch():
    labels_ = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i_ in range(batchSize):
        if (i_ % 2 == 0):
            num = randint(0, 11500 - 1)
            labels_.append([1, 0])
        else:
            num = randint(13500 - 1, 25000 - 1)
            labels_.append([0, 1])
        # arr[i_] = ids[num - 1:num]
        arr[i_] = ids[num]
    return arr, labels_


def get_test_batch():
    labels_ = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i_ in range(batchSize):
        num = randint(11500 - 1, 13500 - 1)
        if num < 12500:
            labels_.append([1, 0])
        else:
            labels_.append([0, 1])
        # arr[i_] = ids[num - 1:num]
        arr[i_] = ids[num]
    return arr, labels_


tf.reset_default_graph()


labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=labels
))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# 训练, training
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('loss', loss)
tf.summary.scalar('Accuar', accuracy)
merged = tf.summary.merge_all()
logdir = 'tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

for i in range(iterations):
    # 下个批次的数据 next batch of data
    next_batch, next_batch_labels = get_train_batch()
    sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels})

    # 每50次写入一次Leadboard, write to leadboard every 50 training turns
    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: next_batch, labels: next_batch_labels})
        writer.add_summary(summary, i)

    if (i % 1000 == 0):
        loss_ = sess.run(loss, {input_data: next_batch, labels: next_batch_labels})
        accuracy_ = (sess.run(accuracy, {input_data: next_batch, labels: next_batch_labels}))
        print("iteration:{}/{}".format(i + 1, iterations),
              "\nloss:{}".format(loss_),
              "\naccuracy:{}".format(accuracy_))
        print('........')
    # 每10000次保存一次模型, save the model every 10,000 training turns

    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretained_lstm.ckpt", gloal_step=i)
        print("saved to %s" % save_path)
writer.close()
