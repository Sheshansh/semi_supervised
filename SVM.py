# from __future__ import print_function
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import numpy as np

data = np.loadtxt('magic04.data',delimiter=',') # shape is -1,11
np.random.shuffle(data)
unlabelled_data = data[:100]
train_data = data[13900:14000]
test_data = data[14000:]
del data
train_data_x = train_data[:,:10]
train_labels = train_data[:,10]
# train_data_x = train_data_x-np.mean(train_data_x,axis=0)
# train_data_x = train_data_x/np.std(train_data_x,axis=0)
pos_x = train_data_x[train_labels==1]
neg_x = train_data_x[train_labels==0]

test_data_x = test_data[:,:10]
# test_data_x = test_data_x-np.mean(test_data_x,axis=0)
# test_data_x = test_data_x/np.std(test_data_x,axis=0)
test_labels = test_data[:,10]
test_pos = test_data_x[test_labels==1]
test_neg = test_data_x[test_labels==0]
test_labels = 2*test_labels-1
test_labels = test_labels.reshape(-1,1)

unlabelled_data_x = unlabelled_data[:,:10]
# unlabelled_data_x = unlabelled_data_x-np.mean(unlabelled_data_x,axis=0)
# unlabelled_data_x = unlabelled_data_x/np.std(unlabelled_data_x,axis=0)

theta_p = 0.5
theta_n = 1.0-theta_p


# Parameters
learning_rate = 0.01
num_steps = 3000
batch_size = 128
display_step = 1000
num_input = 10

# tf Graph input
X = tf.placeholder("float", [None, num_input])
X_pos = tf.placeholder("float", [None, num_input])
X_neg = tf.placeholder("float", [None, num_input])
X_unlabelled = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, 1])

svm_W = tf.Variable(tf.random_normal([num_input,1]))
svm_b = tf.Variable(tf.random_normal([1]))

def SVM(X):
	return tf.matmul(X,svm_W) + svm_b

def lossplusg(x):
	# return tf.reduce_mean(tf.maximum(tf.zeros([tf.shape(x)[0],1]), 1 - x))
	return tf.reduce_mean(tf.maximum(tf.zeros([tf.shape(x)[0],1]), tf.minimum( tf.zeros([tf.shape(x)[0],1])+2, 1-x ) ))

def lossnegg(x):
	# return tf.reduce_mean(tf.maximum(tf.zeros([tf.shape(x)[0],1]), 1 + x))
	return tf.reduce_mean(tf.maximum(tf.zeros([tf.shape(x)[0],1]), tf.minimum( tf.zeros([tf.shape(x)[0],1])+2, 1+x ) ))

# def lossplusg_crossentropy(x):
# 	tf.softmax()

# def softmax(X,axis):
# 	y = np.atleast_2d(X)
# 	y = y * float(1.0)
# 	y = y - np.expand_dims(np.max(y, axis = axis), axis)
# 	y = np.exp(y)
# 	ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
# 	p = y / ax_sum
# 	if len(X.shape) == 1: p = p.flatten()
# 	return p

# Construct model
logits_pos = SVM(X_pos)
logits_neg = SVM(X_neg)
logits_unlabelled = SVM(X_unlabelled)
predicted_class = tf.sign(SVM(X))
correct_prediction = tf.equal(Y,predicted_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Define loss and optimizer
regularization_loss = 0.5*tf.reduce_sum(tf.square(svm_W)) 
hinge_loss_PN = theta_p*lossplusg(logits_pos)+theta_n*lossnegg(logits_neg)
hinge_loss_PU = lossnegg(logits_unlabelled)-theta_p*lossnegg(logits_pos)+theta_p*lossplusg(logits_pos)
hinge_loss_NU = lossplusg(logits_unlabelled)-theta_n*lossplusg(logits_neg)+theta_n*lossnegg(logits_neg)

def run(gamma,run_type):
	
	if run_type == 'pn':
		loss_op = hinge_loss_PN + regularization_loss #PN
	elif run_type == 'punu':
		loss_op = hinge_loss_NU*gamma + hinge_loss_PU * (1-gamma) + regularization_loss # PUNU
	else:
		if gamma >=0 : # PNU
			loss_op = hinge_loss_PU*gamma+hinge_loss_PN*(1-gamma)+regularization_loss
		else:
			loss_op = hinge_loss_NU*gamma+hinge_loss_PN*(1-gamma)+regularization_loss

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
			sess.run(train_op, feed_dict={X_pos: pos_x,
											X_neg: neg_x,
											X_unlabelled: unlabelled_data_x
										 })
			if step % display_step == 0 or step == 1:
				# Calculate batch loss and accuracy
				[loss,acc] = sess.run([loss_op,accuracy], feed_dict={X_pos: pos_x,
																		 X_neg: neg_x,
																		 X_unlabelled: unlabelled_data_x,
																		 X: np.concatenate((pos_x,neg_x)),
																		 Y: np.concatenate((np.array([1.0]*pos_x.shape[0]),np.array([-1.0]*neg_x.shape[0]))).reshape(-1,1),
																	 })
				print("Step " + str(step) + ", Minibatch Loss= " + \
					  "{:.4f}".format(loss) + ", Training Accuracy= " + \
					  "{:.3f}".format(acc))

		# print("Optimization Finished!")
		# Calculate accuracy for MNIST test images
		print(gamma,",", \
			sess.run(accuracy, feed_dict={X: test_pos,
										  Y: np.array([1.0]*test_pos.shape[0]).reshape(-1,1)})
			, ",",
			sess.run(accuracy, feed_dict={X: test_neg,
										  Y: np.array([-1.0]*test_neg.shape[0]).reshape(-1,1)})
			, ",",
			sess.run(accuracy, feed_dict={X: test_data_x,
										  Y: test_labels})
			)

		# save_path = tf.train.Saver([v for v in tf.global_variables()]).save(sess, 'models/model_pnu.ckpt')
		# print("Model saved in file: %s" % save_path)

print("PN risk")
run(0,'pn')

print("PUNU risk")
eta = 0.0
while eta <=1.04:
	run(eta,'punu')
	eta += 0.05

print("PNU risk")
eta = -1.0
while eta <=1.04:
	run(eta,'pnu')
	eta += 0.1
