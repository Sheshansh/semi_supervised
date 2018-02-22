from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import numpy as np

posdigit = 8
negdigit = 3
labelled_unlabelled_split_barrier = 4000

theta_p = 0.5
theta_n = 1-theta_p

train_data_x = mnist.train.images
train_labels = mnist.train.labels
train_data_x,unlabelled_data_x = train_data_x[:labelled_unlabelled_split_barrier],train_data_x[labelled_unlabelled_split_barrier:]
train_labels,unlabelled_labels = train_labels[:labelled_unlabelled_split_barrier],train_labels[labelled_unlabelled_split_barrier:]


pos_x = train_data_x[np.logical_or(np.logical_or(np.logical_or(np.logical_or(train_labels[:,0]==1,train_labels[:,2]==1),train_labels[:,4]==1),train_labels[:,6]==1),train_labels[:,8]==1)]
neg_x = train_data_x[np.logical_or(np.logical_or(np.logical_or(np.logical_or(train_labels[:,1]==1,train_labels[:,3]==1),train_labels[:,5]==1),train_labels[:,7]==1),train_labels[:,9]==1)]
# pos_x,neg_x = pos_x[:min(pos_x.shape[0],neg_x.shape[0])],neg_x[:min(pos_x.shape[0],neg_x.shape[0])]
train_data_x = np.concatenate((pos_x,neg_x))
train_labels = np.concatenate((np.array([1.0]*pos_x.shape[0]),np.array([0.0]*neg_x.shape[0]))).reshape(-1,1)
train_labels = np.array(list(zip(1.0-train_labels[:,0],train_labels[:,0])))
# del train_labels,train_data_x

def gen_pos_labels(len):
	return np.array([[0.0,1.0]]*len)

def gen_neg_labels(len):
	return np.array([[1.0,0.0]]*len)

test_data_x = mnist.test.images
test_labels = mnist.test.labels
test_labels = np.array(list(zip(list(test_labels[:,0]+test_labels[:,2]+test_labels[:,4]+test_labels[:,6]+test_labels[:,8]),list(test_labels[:,1]+test_labels[:,3]+test_labels[:,5]+test_labels[:,7]+test_labels[:,9]))))
# # Equalise data:
# test_pos = test_data_x[test_labels[:,posdigit]==1]
# test_neg = test_data_x[test_labels[:,negdigit]==1]
test_pos = test_data_x[np.logical_or(np.logical_or(np.logical_or(np.logical_or(test_labels[:,0]==1,test_labels[:,2]==1),test_labels[:,4]==1),test_labels[:,6]==1),test_labels[:,8]==1)]
test_neg = test_data_x[np.logical_or(np.logical_or(np.logical_or(np.logical_or(test_labels[:,1]==1,test_labels[:,3]==1),test_labels[:,5]==1),test_labels[:,7]==1),test_labels[:,9]==1)]

test_pos,test_neg = test_pos[:min(test_pos.shape[0],test_neg.shape[0])],test_neg[:min(test_pos.shape[0],test_neg.shape[0])]
test_data_x = np.concatenate((test_pos,test_neg))
test_labels = np.concatenate((np.array([1.0]*test_pos.shape[0]),np.array([0.0]*test_neg.shape[0]))).reshape(-1,1)
test_labels = np.array(list(zip(1.0-test_labels[:,0],test_labels[:,0])))
test_pos_labels = np.array([0.0]*test_pos.shape[0])
test_pos_labels = np.array(list(zip(1.0-test_pos_labels,test_pos_labels)))
test_neg_labels = np.array([1.0]*test_neg.shape[0])
test_neg_labels = np.array(list(zip(1.0-test_neg_labels,test_neg_labels)))
# print(test_pos.shape,test_neg.shape)
unlabelled_pos = unlabelled_data_x[unlabelled_labels[:,posdigit]==1]
unlabelled_neg = unlabelled_data_x[unlabelled_labels[:,negdigit]==1]
unlabelled_pos,unlabelled_neg = unlabelled_pos[:min(unlabelled_pos.shape[0],unlabelled_neg.shape[0])],unlabelled_neg[:min(unlabelled_pos.shape[0],unlabelled_neg.shape[0])]
# print(unlabelled_pos.shape,unlabelled_neg.shape)
unlabelled_data_x = np.concatenate((unlabelled_pos,unlabelled_neg))
del unlabelled_neg,unlabelled_pos,unlabelled_labels

print(unlabelled_data_x.shape)
print(pos_x.shape,neg_x.shape)
print(test_pos.shape,test_neg.shape)

# Parameters
learning_rate = 0.001
num_steps = 10
batch_size = 100
display_step = 1
num_input = 784
num_classes = 2

# tf Graph input
X = tf.placeholder("float", [None, num_input])
X_pos = tf.placeholder("float", [None, num_input])
X_neg = tf.placeholder("float", [None, num_input])
X_unlabelled = tf.placeholder("float", [None, num_input])
pos_pos_labels = tf.placeholder("float", [None, num_classes])
pos_neg_labels = tf.placeholder("float", [None, num_classes])
neg_pos_labels = tf.placeholder("float", [None, num_classes])
neg_neg_labels = tf.placeholder("float", [None, num_classes])
unlabelled_pos_labels = tf.placeholder("float", [None,num_classes])
unlabelled_neg_labels = tf.placeholder("float", [None,num_classes])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
def weight(out1,out2,out3,name):
	return {
		# 1x20 conv, 1 input, 32 outputs
		'wc1': tf.Variable(tf.random_normal([5, 5, 1, out1]),name=name),
		'wc2': tf.Variable(tf.random_normal([5, 5, out1, out2]),name=name),
		# fully connected,  inputs, 1024 outputs
		'wd1': tf.Variable(tf.random_normal([7*7*out2, out3]),name=name),
		# 1024 inputs, 10 outputs (class prediction)
		'out': tf.Variable(tf.random_normal([out3, num_classes]),name=name)
	}

def bias(out1,out2,out3,name):
	return {
		'bc1': tf.Variable(tf.random_normal([out1]),name=name),
		'bc2': tf.Variable(tf.random_normal([out2]),name=name),
		'bd1': tf.Variable(tf.random_normal([out3]),name=name),
		'out': tf.Variable(tf.random_normal([num_classes]),name=name)
	}

weights = weight(3,6,5,'0')
biases = bias(3,6,5,'0')

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
def conv_net_2c2d(x,dropout):
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1, k=2)
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

# Construct model

logits = conv_net_2c2d(X,0.8)
pos_logits = conv_net_2c2d(X_pos,0.8)
neg_logits = conv_net_2c2d(X_neg,0.8)
unlabelled_logits = conv_net_2c2d(X_unlabelled,0.8)

prediction = tf.round(tf.nn.softmax(logits))
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

pn_loss = theta_p*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pos_logits, labels=pos_pos_labels)) \
			+ theta_n*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neg_logits, labels=neg_neg_labels))
pu_loss = theta_p*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pos_logits, labels=pos_pos_labels)) \
			+ tf.maximum( tf.zeros(1), \
			- theta_p*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pos_logits, labels=pos_neg_labels)) \
			+ tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=unlabelled_logits, labels=unlabelled_neg_labels)) \
			)
nu_loss = theta_n*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neg_logits, labels=neg_neg_labels)) \
			+ tf.maximum( tf.zeros(1), \
			- theta_n*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neg_logits, labels=neg_pos_labels)) \
			+ tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=unlabelled_logits, labels=unlabelled_pos_labels)) \
			)

def run(gamma,run_type):
	
	if run_type == 'pn':
		loss_op = pn_loss
	elif run_type == 'punu':
		loss_op = nu_loss*gamma + pu_loss*(1-gamma) # PUNU
	else:
		if gamma >=0 : # PNU
			loss_op = pu_loss*gamma + pn_loss*(1-gamma)
		else:
			gamma = -1*gamma
			loss_op = nu_loss*gamma + pn_loss*(1-gamma)
			gamma = -1*gamma

	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
	# train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:

		# Run the initializer
		sess.run(init)

		# tf.train.Saver([v for v in tf.global_variables()]).restore(sess, 'models/model_pnu.ckpt')

		for step in range(1, num_steps+1):
			# Run optimization op (backprop)
			sess.run(train_op, feed_dict={
											X_pos: pos_x,
											X_neg: neg_x,
											X_unlabelled: unlabelled_data_x,
											pos_pos_labels : gen_pos_labels(pos_x.shape[0]),
											pos_neg_labels : gen_neg_labels(pos_x.shape[0]),
											neg_pos_labels : gen_pos_labels(neg_x.shape[0]),
											neg_neg_labels : gen_neg_labels(neg_x.shape[0]),
											unlabelled_pos_labels : gen_pos_labels(unlabelled_data_x.shape[0]),
											unlabelled_neg_labels : gen_neg_labels(unlabelled_data_x.shape[0])
										 })
			if step % display_step == 0 or step == 1:
				# Calculate batch loss and accuracy
				[loss,acc] = sess.run([loss_op,accuracy], feed_dict={
																		X_pos: pos_x,
																		X_neg: neg_x,
																		X_unlabelled: unlabelled_data_x,
																		pos_pos_labels : gen_pos_labels(pos_x.shape[0]),
																		pos_neg_labels : gen_neg_labels(pos_x.shape[0]),
																		neg_pos_labels : gen_pos_labels(neg_x.shape[0]),
																		neg_neg_labels : gen_neg_labels(neg_x.shape[0]),
																		unlabelled_pos_labels : gen_pos_labels(unlabelled_data_x.shape[0]),
																		unlabelled_neg_labels : gen_neg_labels(unlabelled_data_x.shape[0]),
																		X: train_data_x,
																		Y: train_labels
																	 })
				test_acc = sess.run(accuracy, feed_dict={X: test_data_x,
												  Y: test_labels})
				print("Step " + str(step) + ", Minibatch Loss= " + \
					  str(loss) + ", Training Accuracy= " + \
					  str(acc) + ", Test Accuracy = " + str(test_acc)
					  )

		# print("Optimization Finished!")
		# Calculate accuracy for MNIST test images
		print(gamma,",", \
			# sess.run(accuracy, feed_dict={X: test_pos,
			# 							  Y: test_neg_labels})
			# , ",",
			# sess.run(accuracy, feed_dict={X: test_neg,
			# 							  Y: test_pos_labels})
			# , ",",
			sess.run(accuracy, feed_dict={X: test_data_x,
										  Y: test_labels})
			)

		# save_path = tf.train.Saver([v for v in tf.global_variables()]).save(sess, 'models/model_pnu.ckpt')
		# print("Model saved in file: %s" % save_path)

print("PN risk")
run(0,'pn')

# print("PUNU risk")
# eta = 0.0
# while eta <=1.04:
# 	run(eta,'punu')
# 	eta += 0.2

# print("PNU risk")
# eta = -0.2
# while eta <=0.24:
# 	run(eta,'pnu')
# 	eta += 0.2
