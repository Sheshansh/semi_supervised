from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import numpy as np

posdigit = 8
negdigit = 3
labelled_unlabelled_split_barrier = 2000

theta_p = 0.5
theta_n = 1-theta_p

train_data_x = mnist.train.images
train_labels = mnist.train.labels
train_data_x,unlabelled_data_x = train_data_x[:labelled_unlabelled_split_barrier],train_data_x[labelled_unlabelled_split_barrier:]
train_labels,unlabelled_labels = train_labels[:labelled_unlabelled_split_barrier],train_labels[labelled_unlabelled_split_barrier:]

pos_x = train_data_x[train_labels[:,posdigit]==1]
neg_x = train_data_x[train_labels[:,negdigit]==1]
pos_x,neg_x = pos_x[:min(pos_x.shape[0],neg_x.shape[0])],neg_x[:min(pos_x.shape[0],neg_x.shape[0])]
train_data_x = np.concatenate((pos_x,neg_x))
train_labels = np.concatenate((np.array([1.0]*pos_x.shape[0]),np.array([0.0]*neg_x.shape[0]))).reshape(-1,1)
train_labels = np.array(list(zip(1.0-train_labels[:,0],train_labels[:,0])))
# del train_labels,train_data_x

test_data_x = mnist.test.images
test_labels = mnist.test.labels

# # Equalise data:
test_pos = test_data_x[test_labels[:,posdigit]==1]
test_neg = test_data_x[test_labels[:,negdigit]==1]
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
num_steps = 2000
batch_size = 500
display_step = 100
num_input = 784
num_classes = 2
n_hidden_1 = 4
n_hidden_2 = 3

# tf Graph input
X = tf.placeholder("float", [None, num_input])
# X_pos = tf.placeholder("float", [None, num_input])
# X_neg = tf.placeholder("float", [None, num_input])
# X_unlabelled = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
	# Hidden fully connected layer with 256 neurons
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	# Hidden fully connected layer with 256 neurons
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	# Output fully connected layer with a neuron for each class
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer

# tf.nn.softmax instead
# def softmax(X,axis):
# 	y = np.atleast_2d(X)
# 	y = y * float(1.0)
# 	y = y - np.expand_dims(np.max(y, axis = axis), axis)
# 	y = np.exp(y)
# 	ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
# 	p = y / ax_sum
# 	if len(X.shape) == 1: p = p.flatten()
# 	return p

# def softmax_cross_entropy_pos(logits_pos):
	


# Construct model

logits = neural_net(X)

prediction = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

def run(gamma,run_type):
	
	if run_type == 'pn':
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	# elif run_type == 'punu':
	# 	loss_op = hinge_loss_NU*gamma + hinge_loss_PU * (1-gamma) + regularization_loss # PUNU
	# else:
	# 	if gamma >=0 : # PNU
	# 		loss_op = hinge_loss_PU*gamma+hinge_loss_PN*(1-gamma)+regularization_loss
	# 	else:
	# 		loss_op = hinge_loss_NU*gamma+hinge_loss_PN*(1-gamma)+regularization_loss

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
											# X_pos: pos_x,
											# X_neg: neg_x,
											# X_unlabelled: unlabelled_data_x
											X: train_data_x,
											Y: train_labels
										 })
			if step % display_step == 0 or step == 1:
				# Calculate batch loss and accuracy
				[loss,acc] = sess.run([loss_op,accuracy], feed_dict={
																		 # X_pos: pos_x,
																		 # X_neg: neg_x,
																		 # X_unlabelled: unlabelled_data_x,
																		 X: train_data_x,
																		 Y: train_labels
																	 })
				print("Step " + str(step) + ", Minibatch Loss= " + \
					  "{:.4f}".format(loss) + ", Training Accuracy= " + \
					  "{:.3f}".format(acc))

		# print("Optimization Finished!")
		# Calculate accuracy for MNIST test images
		print(gamma,",", \
			sess.run(accuracy, feed_dict={X: test_pos,
										  Y: test_neg_labels})
			, ",",
			sess.run(accuracy, feed_dict={X: test_neg,
										  Y: test_pos_labels})
			, ",",
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
# 	eta += 0.05

# print("PNU risk")
# eta = -1.0
# while eta <=1.04:
# 	run(eta,'pnu')
# 	eta += 0.1
