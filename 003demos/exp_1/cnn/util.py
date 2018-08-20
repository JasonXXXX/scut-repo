import h5py
import os
import random
import sys
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf

learning_rate = 0.0001


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


'''
    根据文件路径读取信息，这里的是usps数据集的信息，文本格式
'''
def read_usps(path):
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

        usps_train = {}
        usps_train['train'] = train
        usps_train['images'] = X_tr
        usps_train['labels'] = y_tr

        usps_test = {}
        usps_test['test'] = test
        usps_test['images'] = X_te
        usps_test['labels'] = y_te
        return usps_train, usps_test


'''
    根据文件路径读取信息，并将读取到的照片信息转换成照片并保存
'''
def usps_to_image(path):
    train, test = read_usps(path)
    # training set
    index = 0
    for image, label in zip(train['images'], train['labels']):
        index += 1
        if not os.path.exists('../../datasets/usps/image/%s' % label):
            os.mkdir('../../datasets/usps/image/%s' % label)
        image = np.array(image) * np.float32(256)
        new_img = Image.new("L", (16, 16), "white")
        new_img.putdata(image)
        new_img.save('../../datasets/usps/image/%s/%s_usps_%s.png' % (label, label, index))
    print('usps dataset to images finished. count:', index)
    # test set
    index = 0
    for image, label in zip(test['images'], test['labels']):
        index += 1
        if not os.path.exists('../../datasets/usps/image_test/%s' % label):
            os.makedirs('../../datasets/usps/image_test/%s' % label)
        image = np.array(image) * np.float32(256)
        new_img = Image.new("L", (16, 16), "white")
        new_img.putdata(image)
        new_img.save('../../datasets/usps/image_test/%s/%s_usps_%s.png' % (label, label, index), quality=100)
    print('usps dataset to images finished. count:', index)
    sys.stdout.flush()


'''
    根据文件夹路径读取照片，并处理成神经网络能直接处理的格式
    这里所用的照片大小为 16x16
'''
def preprocess_images(dir, limit=None, rndm=False):
    images = []
    labels = []
    i = 0
    files = os.listdir(dir)
    if rndm:
        random.shuffle(files)
        print("the files shuffled.")
    for file_name in files:
        child = os.path.join(dir, file_name)
        img = Image.open(child)
        img_array = np.array(img)
        img_array = np.reshape(img_array, -1)
        images.append(img_array)
        ground_truth = int(file_name.split('.')[-2])
        ls = [1. if ground_truth == i else 0. for i in range(10)]
        labels.append(ls)
        i = i + 1
        if limit != None and i == limit:
            break
    images = np.array(images) / np.float32(256)
    labels = np.array(labels)
    print("the size of images preprocessed:", i)
    sys.stdout.flush()
    return (images, labels)


'''
    根据输入的 source 的数据及 batch_size，返回一个 batch 供神经网络进行训练
    有做过 mnist 数据集实验的应该都能懂
'''
def get_random_batch(source, batch_size=50):
    nums = []
    # 得到长度跟 source 元素个数一致的数组
    for i in range(len(source[1])):
        nums.append(i)
    # 打乱数组元素顺序
    random.shuffle(nums)
    # 取出前 batch_size 个元素，其实是随机的
    randoms = []
    for i in range(batch_size):
        randoms.append(nums[i])
    images = []
    labels = []
    for index in randoms:
        images.append(source[0][index])
        labels.append(source[1][index])
    return (np.array(images), np.array(labels))


def generate_usps_datasets(dir, target_dir, num_for_one_class=50):
    target_folder = 'train' if 'train' in dir else 'test'
    if num_for_one_class != 50:
        target_folder = '%s_%d' % (target_folder, num_for_one_class * 10)
        target_dir += '_%d' % (num_for_one_class * 10)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for folder in os.listdir(dir):
        images = [file for file in os.listdir('%s/%s' % (dir, folder))]
        random.shuffle(images)
        for i in range(num_for_one_class):
            shutil.copyfile('%s/%s/%s' % (dir, folder, images[i]), './datasets/usps/%s/%s' % (target_folder, images[i]))


def resize_usps():
    usps_dir = '../../datasets/usps/classes/train/'
    for folder in os.listdir(usps_dir):
        count = 1
        for image in os.listdir(usps_dir + folder):
            label = image.split('_')[0]
            new_filename = 'usps_' + str(count) + '.' + label + '.png'
            image = Image.open(usps_dir + folder + '/' + image).convert('L')
            size = (28, 28)
            image =image.resize(size).convert('L')
            image = image.resize(size,Image.ANTIALIAS)
            new_folder = '../../datasets/usps/28x28/train/' + label
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            image.save(new_folder + '/' + new_filename)
            print(new_filename, 'saved.')
            count += 1


def generate_mnist_datasets(dir, target_dir=None, num_for_one_class=50):
    for folder in os.listdir(dir):
        images = [file for file in os.listdir('%s/%s' % (dir, folder))]
        random.shuffle(images)
        for i in range(num_for_one_class):
            shutil.copyfile('%s/%s/%s' % (dir, folder, images[i]), './datasets/mnist_poison0.15/%s' % images[i])


if __name__ == '__main__':
    # generate_mnist_datasets('./datasets/mnist/classes', num_for_one_class=5100)
    generate_usps_datasets('./datasets/usps/28x28/test', './datasets/usps/test', 80)
