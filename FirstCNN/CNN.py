import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

#
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.images

print("MNIST loaded")

#
x = tf.placeholder(tf.float32, [None, train_img.shape[1]])
y = tf.placeholder(tf.float32, [None, train_label.shape[1]])

def weight_var(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_var(shape, name):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pol_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_var([5, 5, 1, 6], 'W_conv1')
b_conv1 = bias_var([6], 'b_conv1')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pol_2x2(h_conv1)

W_conv2 = weight_var([5, 5, 6, 16], 'W_conv2')
b_conv2 = bias_var([16], 'b_con2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pol_2x2(h_conv2)

W_conv3 = weight_var([5, 5, 16, 120], 'W_conv3')
b_conv3 = bias_var([120], 'b_conv3')
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3_flat = tf.reshape(h_conv3, [-1, 7*7*120])

W_fcl1 = weight_var([7*7*120, 1024], 'W_fcl1')
b_fcl1 = bias_var([1024], 'b_fcl1')
h_fcl1 = tf.nn.relu(tf.nn.xw_plus_b(h_pool3_flat, W_fcl1, b_fcl1))
keep_prob = tf.placeholder(tf.float32)
h_fcl1_drop = tf.nn.dropout(h_fcl1, keep_prob)

W_fcl2 = weight_var([1024, 10], 'W_fcl2')
b_fcl2 = bias_var([10], 'b_fcl2')
y_conv = tf.nn.softmax(tf.nn.xw_plus_b(h_fcl1_drop, W_fcl2, b_fcl2))

var_list = [
    W_conv1, b_conv1,
    W_conv2, b_conv2,
    W_conv3, b_conv3,
    W_fcl1, b_fcl1,
    W_fcl2, b_fcl2
]

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if not os.path.exists('Model/'):
    os.mkdir('Model/')

saver = tf.train.Saver(var_list)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100000):
        batch = mnist.train.next_batch(60)
        if i % 1000 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0],
                y: batch[1],
                keep_prob: 0.5
            })
            print('step %d,training accuracy %g' % (i, train_accuracy))
        sess.run(optimizer, feed_dict={
            x: batch[0],
            y: batch[1],
            keep_prob: 0.5
        })
    saver.save(sess, 'Model/model.ckpt')
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
