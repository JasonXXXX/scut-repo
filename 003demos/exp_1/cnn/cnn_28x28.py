import sys
import tensorflow as tf
import util

class model(object):
	"""docstring for model"""
	def __init__(self, num_classes=10):
		super(model, self).__init__()
		self._num_classes = num_classes

	def cnn_model(self, graph=tf.Graph()):
		image_size = 28
		with graph.as_default():
			# define neural network
			with tf.name_scope('inputs'):
				self.x_raw = tf.placeholder(tf.float32, shape=[None, image_size * image_size], name='x_raw')
				self._y = tf.placeholder(tf.float32, shape=[None, self._num_classes], name='_y')
				self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

			with tf.name_scope('conv_1'):
				#first convolutinal layer
				w_conv1 = util.weight_variable([5, 5, 1, 32], name='w_conv1')
				b_conv1 = util.bias_variable([32], name='b_conv1')
				x_image = tf.reshape(self.x_raw, [-1, image_size, image_size, 1])
				h_conv1 = tf.nn.relu(util.conv2d(x_image, w_conv1) + b_conv1)

				tf.summary.image('x_input', x_image, max_outputs=self._num_classes)
				tf.summary.histogram('w_conv1', w_conv1)
				tf.summary.histogram('b_conv1', b_conv1)

			with tf.name_scope('pool_1'):
				h_pool1 = util.max_pool_2x2(h_conv1)
				tf.summary.histogram('pool_1', h_pool1)

			with tf.name_scope('conv_2'):
				# second convolutional layer
				w_conv2 = util.weight_variable([3, 3, 32, 64], name='w_conv2')
				b_conv2 = util.bias_variable([64], name='b_conv2')
				h_conv2 = tf.nn.relu(util.conv2d(h_pool1, w_conv2) + b_conv2)

				tf.summary.histogram('w_conv2', w_conv2)
				tf.summary.histogram('b_conv2', b_conv2)

			with tf.name_scope('pool_2'):
				h_pool2 = util.max_pool_2x2(h_conv2)
				tf.summary.histogram('pool_2', h_pool2)

			with tf.name_scope('dense_1'):
				# densely connected layer
				w_fc1 = util.weight_variable([7 * 7 * 64, 1024], name='w_fc1') # 到这里，图像被缩为 7x7
				b_fc1 = util.bias_variable([1024], name='b_fc1')
				h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
				h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
				h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

				tf.summary.histogram('w_fc1', w_fc1)
				tf.summary.histogram('b_fc1', b_fc1)

			with tf.name_scope('dense_2'):
				w_fc2 = util.weight_variable([1024, self._num_classes], name='w_fc2')
				b_fc2 = util.bias_variable([self._num_classes], name='b_fc2')
				y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

				tf.summary.histogram('w_fc2', w_fc2)
				tf.summary.histogram('b_fc2', b_fc2)
			return graph, y_conv

	def train(self, graph, epoch, optimizer, accuracy, log_path=None, train=None, test=None, batch_size=50, restore_path=None, mnist=None):
		if not train and not test and not mnist:
			print('no data provided, the neural network cannot be trained.')
			exit()
		if log_path != None:
			model_path = log_path + '/model'

		with tf.Session(graph=graph) as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			saver = tf.train.Saver(max_to_keep=5)

			# 这里是迁移后的 fine-tune 了
			if not restore_path:
				print('no model for restored needed or provided.')
			else:
				saver.restore(sess, restore_path)
				print(restore_path, 'model restored.')

			if log_path != None:
				merged = tf.summary.merge_all()
				summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)
				summary_writer.add_graph(tf.get_default_graph())

			high_acc_count = 0
			for i in range(1, epoch + 1):
				# 获取一个 batch 的数据用于训练
				if not mnist:
					batch = util.get_random_batch(train, batch_size=batch_size)
					#numbers = [i for i in batch_size]
					#random.shuffle(numbers)
					#for i in range(0.3 * batch_size):
						#random.shuffle(batch[1][numbers[i]])
				else:
					batch = mnist.train.next_batch(batch_size)
				# 执行优化计算过程
				optimizer.run(feed_dict={self.x_raw: batch[0], self._y: batch[1], self.keep_prob: 0.5})
				if i % 20 == 0:
					summary = sess.run(merged, feed_dict={self.x_raw: batch[0], self._y: batch[1], self.keep_prob: 1.0})
					summary_writer.add_summary(summary, i)
				if i % 100 == 0:
					train_accuracy = accuracy.eval(feed_dict={self.x_raw:batch[0], self._y: batch[1], self.keep_prob: 1.0})
					print("++ step %d, training accuracy: %g" % (i, train_accuracy))
					sys.stdout.flush()
					saver.save(sess, save_path=model_path, global_step=i)
					if train_accuracy > 0.988:
						high_acc_count += 1
						if high_acc_count >= 8 and i > int(epoch / 2) and not restore_path:
							epoch = i
							print('the neural network training is converged. epoch:', epoch)
							# break
				# if i == 4000:
					# break
			if log_path != None:
				saver.save(sess, save_path=model_path)
			if not test:
				print('this is the 1st step for transferring, finish now.')
			else:
				content = "++ test accuracy: %g" % accuracy.eval(
					feed_dict={self.x_raw: test[0], self._y: test[1], self.keep_prob: 1.0})
				print(content)
				if log_path != None:
					with open(log_path + '/result.txt', 'wb+') as fp:
						content = 'epoch: %d\t\r\ntrain samples: %d, test samples: %d\t\r\n%s' % (
							epoch, len(train[1]), len(test[1]), content)
						fp.write(content.encode())
