from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import numpy as np

posdigit = 1
labelled_unlabelled_split_barrier = 50

theta_p = 0.5
theta_n = 1-theta_p

train_data_x = mnist.train.images
train_labels = mnist.train.labels
train_data_x,unlabelled_data_x = train_data_x[:labelled_unlabelled_split_barrier],train_data_x[labelled_unlabelled_split_barrier:]
train_labels,unlabelled_labels = train_labels[:labelled_unlabelled_split_barrier],train_labels[labelled_unlabelled_split_barrier:]
train_labels = np.concatenate((train_labels[:,posdigit].reshape(-1,1),1-train_labels[:,posdigit].reshape(-1,1)),axis=1)
unlabelled_labels = np.concatenate((unlabelled_labels[:,posdigit].reshape(-1,1),1-unlabelled_labels[:,posdigit].reshape(-1,1)),axis=1)
pos_x = train_data_x[train_labels[:,0]==1]
neg_x = train_data_x[train_labels[:,0]==0]
test_data_x = mnist.test.images
test_labels = mnist.test.labels
test_labels = np.concatenate((test_labels[:,posdigit].reshape(-1,1),1-test_labels[:,posdigit].reshape(-1,1)),axis=1)
test_labels = test_labels[:,0]
unlabelled_labels = unlabelled_labels[:,0]

# # Equalise data:
test_pos = test_data_x[test_labels==1.0]
test_neg = test_data_x[test_labels==0.0]
print(test_pos.shape,test_neg.shape)
test_data_x = np.concatenate((test_pos,test_neg[:test_pos.shape[0]]))
test_labels = np.concatenate((np.array([1.0]*test_pos.shape[0]),np.array([0.0]*test_pos.shape[0])))
unlabelled_pos = unlabelled_data_x[unlabelled_labels==1.0]
unlabelled_neg = unlabelled_data_x[unlabelled_labels==0.0]
print(unlabelled_pos.shape,unlabelled_neg.shape)
unlabelled_data_x = np.concatenate((unlabelled_pos,unlabelled_neg[:unlabelled_pos.shape[0]]))
unlabelled_labels = np.concatenate((np.array([1.0]*unlabelled_pos.shape[0]),np.array([0.0]*unlabelled_pos.shape[0])))
neg_x = neg_x[:pos_x.shape[0]]

test_labels = test_labels.reshape(-1,1)
test_data_x = test_data_x.reshape(-1,test_data_x.shape[1])

print(unlabelled_data_x.shape)
print(pos_x.shape)
print(neg_x.shape)

# Parameters
learning_rate = 0.01
num_steps = 2000
batch_size = 128
display_step = 10
num_input = 784

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
	return tf.reduce_mean(tf.maximum(tf.zeros([tf.shape(x)[0],1]), 1 - x))

def lossnegg(x):
	return tf.reduce_mean(tf.maximum(tf.zeros([tf.shape(x)[0],1]), 1 + x))

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
# loss_op = hinge_loss_PN + regularization_loss
# loss_op = hinge_loss_PU + regularization_loss
# loss_op = hinge_loss_NU + regularization_loss
# gamma = 0.5
# loss_op = hinge_loss_NU*gamma + hinge_loss_PU * (1-gamma) + regularization_loss
eta = 0.0
if eta >=0 :
	loss_op = hinge_loss_PU*eta+hinge_loss_PN*(1-eta)+regularization_loss
else:
	loss_op = hinge_loss_NU*eta+hinge_loss_PN*(1-eta)+regularization_loss

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
# train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)

# # Evaluate model
# correct_pred = tf.equal(tf.argmax(prediction, 1), 1-tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

	# Run the initializer
	sess.run(init)

	# tf.train.Saver([v for v in tf.global_variables()]).restore(sess, 'models/model_pnu.ckpt')

	for step in range(1, num_steps+1):
		# Run optimization op (backprop)
		# sess.run(train_op, feed_dict={X_pos: pos_x,
		# 								X_neg: neg_x
		# 							 })
		# sess.run(train_op, feed_dict={X_pos: pos_x,
		# 								X_unlabelled: unlabelled_data_x
		# 							 })
		# sess.run(train_op, feed_dict={X_neg: neg_x,
		# 								X_unlabelled: unlabelled_data_x
		# 							 })
		sess.run(train_op, feed_dict={X_pos: pos_x,
										X_neg: neg_x,
										X_unlabelled: unlabelled_data_x
									 })
		if step % display_step == 0 or step == 1:
			# Calculate batch loss and accuracy
			# [loss,acc] = sess.run([loss_op,accuracy], feed_dict={X_pos: pos_x,
			# 													 X_neg: neg_x,
			# 													 X: np.concatenate((pos_x,neg_x)),
			# 													 Y: np.concatenate((np.array([1.0]*pos_x.shape[0]),np.array([-1.0]*neg_x.shape[0]))).reshape(-1,1),
			# 													 })
			# [loss,acc] = sess.run([loss_op,accuracy], feed_dict={X_pos: pos_x,
			# 														 X_unlabelled: unlabelled_data_x,
			# 														 X: pos_x,
			# 														 Y: np.array([1.0]*pos_x.shape[0]).reshape([-1,1])
			# 													 })
			# [loss,acc] = sess.run([loss_op,accuracy], feed_dict={X_neg: neg_x,
			# 														 X_unlabelled: unlabelled_data_x,
			# 														 X: neg_x,
			# 														 Y: np.array([-1.0]*neg_x.shape[0]).reshape([-1,1])
			# 													 })
			[loss,acc] = sess.run([loss_op,accuracy], feed_dict={X_pos: pos_x,
																	 X_neg: neg_x,
																	 X_unlabelled: unlabelled_data_x,
																	 X: np.concatenate((pos_x,neg_x)),
																	 Y: np.concatenate((np.array([1.0]*pos_x.shape[0]),np.array([-1.0]*neg_x.shape[0]))).reshape(-1,1),
																 })
			print("Step " + str(step) + ", Minibatch Loss= " + \
				  "{:.4f}".format(loss) + ", Training Accuracy= " + \
				  "{:.3f}".format(acc))

	print("Optimization Finished!")
	# Calculate accuracy for MNIST test images
	# print("Testing Accuracy:", \
	# 	list(sess.run(predicted_class, feed_dict={X: test_data_x})))
	print("Testing Accuracy:", \
		sess.run(accuracy, feed_dict={X: test_pos,
									  Y: np.array([1.0]*test_pos.shape[0]).reshape(-1,1)}))
	print("Testing Accuracy:", \
		sess.run(accuracy, feed_dict={X: test_neg,
									  Y: np.array([-1.0]*test_neg.shape[0]).reshape(-1,1)}))
	print("Testing Accuracy:", \
		sess.run(accuracy, feed_dict={X: test_data_x,
									  Y: 2*test_labels-1}))
	save_path = tf.train.Saver([v for v in tf.global_variables()]).save(sess, 'models/model_pnu.ckpt')
	print("Model saved in file: %s" % save_path)
