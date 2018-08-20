import random
import tensorflow as tf
import util
import cnn_28x28 as cnn

epoch = 0
num_of_classes = 10
restore_path = './models/two-step/2/8000_epoches/model'

test = util.preprocess_images('./datasets/usps/test_800')

model = cnn.model()
myGraph, y_conv = model.cnn_model()

with myGraph.as_default():
	with tf.name_scope('training'):
		train_vars = None
		cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=model._y), name='cross_entropy')
		optimizer = tf.train.AdamOptimizer(learning_rate=util.learning_rate).minimize(cross_entropy, var_list=train_vars)
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(model._y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		tf.summary.scalar('accuracy', accuracy)
		tf.summary.scalar('loss', cross_entropy)

print('start training, epoch:', epoch)
model.train(graph=myGraph, optimizer=optimizer, accuracy=accuracy, epoch=epoch, log_path=None, test=test, restore_path=restore_path)
