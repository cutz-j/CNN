import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("c:/data/MNIST_data", one_hot=True)

## parameter ##
input_size = 784
num_classes = 10
batch_size = 100
total_batches = 200

def add_var_summary(tf_variable, summary_name):
    with tf.name_scope(summary_name + '_summary'):
        mean = tf.reduce_mean(tf_variable)
        tf.summary.scalar('Mean', mean)
        with tf.name_scope('Standard_deviaction'):
            std = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))
        tf.summary.scalar('StandardDeviation', std)
        tf.summary.scalar('Max', tf.reduce_max(tf_variable))
        tf.summary.scalar('Min', tf.reduce_min(tf_variable))
        tf.summary.histogram('Histogram', tf_variable)

x_input = tf.placeholder(tf.float32, shape=[None, input_size])
y_input = tf.placeholder(tf.float32, shape=[None, num_classes])

x_input_reshape = tf.reshape(x_input, [-1, 28, 28, 1],
                             name='input_reshape')
## cnn layer(kernel) ##
def convolution_layer(input_layer, filters, kernel_size=[3,3],
                      activation=tf.nn.relu):
    ## CNN layer 생성 & 기록 ##
    layer = tf.layers.conv2d(inputs=input_layer, filters=filters,
                             kernel_size=kernel_size, activation=activation)
    add_var_summary(layer, 'convolution')
    return layer

def pooling_layer(input_layer, pool_size=[2,2], strides=2):
    ## max_pooling lyaer 생성 & 기록 ##
    layer = tf.layers.max_pooling2d(inputs=input_layer, pool_size=pool_size,
                                    strides=strides)
    add_var_summary(layer, 'pooling')
    return layer

def dense_layer(input_layer, units, activation=tf.nn.relu):
    ## layer dense ##
    layer = tf.layers.dense(inputs=input_layer, units=units, 
                            activation=activation)
    add_var_summary(layer, 'dense')
    return layer

## layer builiding ##
cnn1 = convolution_layer(x_input_reshape, 64)
pooling_layer1 = pooling_layer(cnn1)
cnn2 = convolution_layer(pooling_layer1, 128)
pooling_layer2 = pooling_layer(cnn2)
fully_connected = tf.reshape(pooling_layer2, [-1, 5*5*128],
                             name='fully_connected')
dense_layer_bottleneck = dense_layer(fully_connected, 1024)

dropout_bool = tf.placeholder(dtype=tf.bool)
dropout_layer = tf.layers.dropout(inputs=dense_layer_bottleneck,
                                  rate=0.4, training=dropout_bool)
logits = dense_layer(dropout_layer, units=num_classes)

# cost #
with tf.name_scope('cost'):
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_input,
                                                      logits=logits)
    opt = tf.reduce_mean(cost, name='cost')
    tf.summary.scalar('cost', opt)
    
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer().minimize(opt)
    
with tf.name_scope('acc'):
    with tf.name_scope('predict'):
        predict = tf.argmax(logits, axis=1)
        correct = tf.equal(predict, tf.argmax(y_input, axis=1))
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
tf.summary.scalar('acc', accuracy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

## summary merge ##
merged_summary = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter("c:/data/train", sess.graph)
test_summary_writer = tf.summary.FileWriter("c:/data/test")

test_images, test_labels = mnist_data.test.images, mnist_data.test.labels

for batch_num in range(total_batches):
    mnist_batch = mnist_data.train.next_batch(batch_size)
    train_images, train_labels = mnist_batch[0], mnist_batch[1]
    _, merge = sess.run([train, merged_summary],
                        feed_dict={x_input: train_images,
                                   y_input: train_labels,
                                   dropout_bool: True})
    train_summary_writer.add_summary(merge, batch_num)
    if batch_num % 10 == 0:
        merge, _ = sess.run([merged_summary, accuracy],
                            feed_dict={x_input: test_images,
                                       y_input: test_labels,
                                       dropout_bool: False})
    test_summary_writer.add_summary(merge, batch_num)

sess.close()
