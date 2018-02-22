from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 10
batch_size = 100
display_step = 1
num_batches = 10
theta_p = 0.5
theta_n = 1-theta_p

def convert_to_binary(test_labels):
    test_labels = np.array(list(zip(list(test_labels[:,0]+test_labels[:,2]+test_labels[:,4]+test_labels[:,6]+test_labels[:,8]),list(test_labels[:,1]+test_labels[:,3]+test_labels[:,5]+test_labels[:,7]+test_labels[:,9]))))
    return test_labels

x_batches = np.array_split(mnist.train.images[:500],num_batches)
y_batches = np.array_split(convert_to_binary(mnist.train.labels[:500]),num_batches)
unlabelled_batches = np.array_split(mnist.train.images[10000:],num_batches)

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
X_pos = tf.placeholder(tf.float32, [None, num_input])
Y_pos_pos = tf.placeholder(tf.float32, [None, num_classes])
Y_pos_neg = tf.placeholder(tf.float32, [None, num_classes])
X_neg = tf.placeholder(tf.float32, [None, num_input])
Y_neg_neg = tf.placeholder(tf.float32, [None, num_classes])
Y_neg_pos = tf.placeholder(tf.float32, [None, num_classes])
Y = tf.placeholder(tf.float32, [None, num_classes])
X_u = tf.placeholder(tf.float32, [None, num_input])
Y_u_pos = tf.placeholder(tf.float32, [None, num_classes])
Y_u_neg = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
logits_pos = conv_net(X_pos, weights, biases, keep_prob)
logits_neg = conv_net(X_neg, weights, biases, keep_prob)
logits_u = conv_net(X_u, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
pn_loss = theta_p*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_pos, labels=Y_pos_pos))\
    +theta_n*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_neg, labels=Y_neg_neg))

pu_loss_neg_comp = - theta_p*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_pos, labels=Y_pos_neg))
pu_loss_unlabelled = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_u, labels=Y_u_neg)) 
pu_loss = theta_p*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_pos, labels=Y_pos_pos)) \
     +pu_loss_neg_comp + pu_loss_unlabelled
    # + tf.maximum( tf.zeros(1), \
    # )

nu_loss = theta_n*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_neg, labels=Y_neg_neg)) \
    - theta_n*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_neg, labels=Y_neg_pos)) \
    + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_u, labels=Y_u_pos))  \
    # + tf.maximum( tf.zeros(1), \
    # )

# loss_op = 0.5*pu_loss+0.5*nu_loss
loss_op = pu_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = x_batches[step%num_batches]
        batch_y = y_batches[step%num_batches]
        batch_u = unlabelled_batches[step%num_batches]
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={
            X_pos : batch_x[batch_y[:,1]==1],
            Y_pos_pos : batch_y[batch_y[:,1]==1],
            Y_pos_neg : 1-batch_y[batch_y[:,1]==1],
            X_neg : batch_x[batch_y[:,1]==0],
            Y_neg_neg : batch_y[batch_y[:,1]==0],
            Y_neg_pos : 1-batch_y[batch_y[:,1]==0],
            X_u : batch_u,
            Y_u_pos : np.array([[0,1]]*(batch_u.shape[0])),
            Y_u_neg : np.array([[1,0]]*(batch_u.shape[0])),
            X : batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss,loss1,loss2,acc = sess.run([loss_op, pu_loss_unlabelled, pu_loss_neg_comp, accuracy], feed_dict={
                                                                X_pos : batch_x[batch_y[:,1]==1],
                                                                Y_pos_pos : batch_y[batch_y[:,1]==1],
                                                                Y_pos_neg : 1-batch_y[batch_y[:,1]==1],
                                                                X_neg : batch_x[batch_y[:,1]==0],
                                                                Y_neg_neg : batch_y[batch_y[:,1]==0],
                                                                Y_neg_pos : 1-batch_y[batch_y[:,1]==0],
                                                                X_u : batch_u,
                                                                Y_u_pos : np.array([[0,1]]*(batch_u.shape[0])),
                                                                Y_u_neg : np.array([[1,0]]*(batch_u.shape[0])),
                                                                X: batch_x,
                                                                Y: batch_y,
                                                                keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  # "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  str(loss) + ", " + str(loss1) + "," + str(loss2) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc),"Testing Accuracy:", \
                    sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                                  Y: convert_to_binary(mnist.test.labels[:256]),
            keep_prob: 1.0}))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: convert_to_binary(mnist.test.labels[:256]),
    keep_prob: 1.0}))
    save_path = tf.train.Saver().save(sess, 'models/cnn_model.ckpt')
    print("Model saved in file: %s" % save_path)