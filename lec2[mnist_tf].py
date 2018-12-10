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
learning_rate = 0.1

## tf building ##
x_input = tf.placeholder(tf.float32, shape=[None, input_size])
y_input = tf.placeholder(tf.float32, shape=[None, num_classes])

W = tf.Variable(tf.random_normal([input_size, num_classes]))
b = tf.Variable(tf.random_normal([num_classes]))
logits = tf.matmul(x_input, W) + b

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_input, logits=logits)
optimizer = tf.reduce_mean(cost)
train = tf.train.AdamOptimizer(learning_rate).minimize(optimizer)

predict = tf.argmax(logits, 1)
correct = tf.equal(predict, tf.argmax(y_input, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for batch_num in range(total_batches):
    mnist_batch = mnist_data.train.next_batch(batch_size)
    _, cost_val = sess.run([train, cost], 
                           feed_dict={x_input: mnist_batch[0],
                                      y_input: mnist_batch[1]})
    print(np.mean(cost_val))

test_images, test_labels = mnist_data.test.images, mnist_data.test.labels
acc = sess.run(accuracy, feed_dict={x_input: test_images,
                                    y_input: test_labels})

print(acc)
sess.close()







