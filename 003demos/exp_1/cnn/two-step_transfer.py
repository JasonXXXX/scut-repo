import random
import tensorflow as tf
import util
import cnn_28x28 as cnn

flags = tf.app.flags

flags.DEFINE_string('step', None, 'Step of this experiment.')
flags.DEFINE_string('tuneall', None, 'If tune all parameters in this experiment.')
flags.DEFINE_string('noise', None, 'Partial number of noise, e.g. 5 means 5%')
flags.DEFINE_string('poison', None, 'Partial number of poison, e.g. 5 means 5%')
flags.DEFINE_string('epoch', None, 'Epoches of this experiment.')
FLAGS = flags.FLAGS

epoch = 4000 if not FLAGS.epoch else int(FLAGS.epoch)
num_of_classes = 10
on_source = './models/two-step/1/%d_epoches' % epoch
if int(FLAGS.step) == 2:
	epoch = 2000
to_target = './models/two-step/2/%d_epoches' % epoch

if not FLAGS.step or (FLAGS.step != '1' and FLAGS.step != '2'):
	print('\nplz type \'python two-step_transfer.py --step=1/2 --tuneall=yes/no(default is no)\'\n')
	exit()

if FLAGS.noise != None:
	noise_rate = float(int(FLAGS.noise) / 100)
	on_source += '_%gnoise' % noise_rate
	to_target += '_%gnoise' % noise_rate

if FLAGS.poison != None:
	poison_rate = float(int(FLAGS.poison) / 100)
	on_source += '_%gpoison' % poison_rate
	to_target += '_%gpoison' % poison_rate

train = None
test = None
restore_path = None
mnist = None
if FLAGS.step == '1':
	if not FLAGS.poison:
		from tensorflow.examples.tutorials.mnist import input_data
		mnist = input_data.read_data_sets('./datasets/mnist/gzs/', one_hot=True)
	else:
		train = util.preprocess_images('./datasets/mnist_poison0.15', rndm=True)
	if FLAGS.noise != None:
		noise_rate = float(int(FLAGS.noise) / 100)
		number = int(noise_rate * len(mnist.train.labels))
		# 生成一个有 60000 个数字的 list
		number_list = []
		for i in range(len(mnist.train.labels)):
			number_list.append(i)
		# 打乱这个 list 的顺序
		random.shuffle(number_list)
		# 取出前 number 个，将标签打乱，即造成了噪声
		for i in range(number):
			random.shuffle(mnist.train.labels[number_list[i]])
		print(number, 'noise samples generated.')
else:
	train = util.preprocess_images('./datasets/usps/train_300')
	test = util.preprocess_images('./datasets/usps/test_800')
	restore_path = on_source + '/model'

log_path = on_source if int(FLAGS.step) == 1 else to_target

model = cnn.model()
myGraph, y_conv = model.cnn_model()

with myGraph.as_default():
	with tf.name_scope('training'):
		# 指定重新训练的参数，即冻结其他层的参数
		if FLAGS.tuneall != 'yes' and FLAGS.step == '2':
			train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense_2')
		# 或者全部的层都进行 fine-tune
		else:
			train_vars = None
		cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=model._y), name='cross_entropy')
		optimizer = tf.train.AdamOptimizer(learning_rate=util.learning_rate).minimize(cross_entropy, var_list=train_vars)
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(model._y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		tf.summary.scalar('accuracy', accuracy)
		tf.summary.scalar('loss', cross_entropy)

print('start training, epoch:', epoch)
model.train(graph=myGraph,
			optimizer=optimizer,
			accuracy=accuracy,
			epoch=epoch,
			log_path=log_path,
			train=train,
			test=test,
			restore_path=restore_path,
			mnist=mnist)
