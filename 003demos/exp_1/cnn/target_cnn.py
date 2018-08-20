import sys
import tensorflow as tf
import util
import cnn_28x28 as cnn

epoch = 2000
num_of_classes = 10
train = util.preprocess_images('./datasets/usps/train_300')
test = util.preprocess_images('./datasets/usps/test_800')
log_path = './models/target/%d_epoches' % epoch

model = cnn.model()
myGraph, y_conv = model.cnn_model()

with myGraph.as_default():
	with tf.name_scope('training'):
		cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=model._y), name='cross_entropy')
		optimizer = tf.train.AdamOptimizer(learning_rate=util.learning_rate).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(model._y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		tf.summary.scalar('accuracy', accuracy)
		tf.summary.scalar('loss', cross_entropy)

print('start training, epoch:', epoch, '; target training:', len(train[1]))
model.train(graph=myGraph, optimizer=optimizer, accuracy=accuracy, epoch=epoch, log_path=log_path, train=train, test=test)
