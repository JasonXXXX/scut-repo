# import h5py
import os
import random
import sys
import numpy as np

from PIL import Image


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
    根据文件路径读取信息，并将读取到的照片信心转换成照片并保存
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
        new_img.save('../../datasets/usps/image_test/%s/%s_usps_%s.png' % (label, label, index))
    print('usps dataset to images finished. count:', index)
    sys.stdout.flush()


'''
    根据文件路径读取信息，并且处理后返回一个样本数量更少的数据集
'''
def get_smaller_usps(path):
    train, test = read_usps(path)
    small_train = {
        'images': [],
        'labels': []
    }
    small_test = {
        'images': [],
        'labels': []
    }
    # process train data
    counts = []
    for i in range(10):
        counts.append(0)
    full_count = 0
    index = 0
    fulleds = []

    for image, label in zip(train['images'], train['labels']):
        index += 1
        if full_count == 10:
            small_train['images'] = np.array(small_train['images'])
            small_train['labels'] = np.array(small_train['labels'])
            print('get smaller training set finished.')
            break
        if counts[label] < 100:
            small_train['images'].append(image)
            label_ls = [1.0 if label == i else 0.0 for i in range(10)]
            small_train['labels'].append(label_ls)
            counts[label] += 1
        elif not label in fulleds:
            full_count += 1
            fulleds.append(label)
    if full_count < 10:
        print('train data not valid, and counts', counts)
    print('scan:', index, ', all training:', len(train['labels']))
    # process test data
    counts = []
    for i in range(10):
        counts.append(0)
    full_count = 0
    index = 0
    fulleds = []

    for image, label in zip(test['images'], test['labels']):
        index += 1
        if full_count == 10:
            small_test['images'] = np.array(small_test['images'])
            small_test['labels'] = np.array(small_test['labels'])
            print('get smaller test set finished.')
            break
        if counts[label] < 100:
            small_test['images'].append(image)
            label_ls = [1.0 if label == i else 0.0 for i in range(10)]
            # print(label, label_ls)
            small_test['labels'].append(label_ls)
            counts[label] += 1
        elif not label in fulleds:
            full_count += 1
            fulleds.append(label)
    if full_count < 10:
        print('test data not valid, and counts', counts)
    print('scan:', index, ', all test:', len(test['labels']))
    print('train images shape:', small_train['images'].shape, ', labels shape:', small_train['labels'].shape)
    print('test images shape:', small_test['images'].shape, ', labels shape:', small_test['labels'].shape)
    sys.stdout.flush()
    return small_train, small_test


'''
    根据文件夹路径读取照片，并处理成神经网络能直接处理的格式，并保存到txt文件
'''
def preprocess_images(dir):
    image = []
    label = []
    i = 0
    files = os.listdir(dir)
    # random.shuffle(files)
    for file_name in files:
        child = os.path.join(dir, file_name)
        img = Image.open(child)
        img_array = np.array(img)
        img_array = np.reshape(img_array, -1)
        image.append(img_array)
        true_label = int(file_name.split('_')[0])

        #print(true_label, file_name)
        label_ls = [1 if true_label == i else 0 for i in range(10)]

        label.append(float(true_label))
        #label.append(label_ls)
        i = i + 1
    image = np.array(image) / np.float32(256)
    label = np.array(label)
    #print("the size of images dataset:", i)
    sys.stdout.flush()
    arr=[]
    arr.append(image)
    arr.append(label)

    return arr


'''
    根据文件夹路径读取照片的二进制编码，并处理成神经网络能直接处理的格式
'''
#def preprocess_images(image_dir,label_dir):

'''
    根据文件夹路径读取照片，只选出要用到的usps部分的照片，并处理成神经网络能直接处理的格式
'''
def read_usps_from_all_images(dir):
    image = []
    label = []
    i = 0
    files = []
    temp_files = os.listdir(dir)
    for file in temp_files:
        if 'usps' in file:
            files.append(file)
    # random.shuffle(files)
    for file_name in files:
        child = os.path.join(dir, file_name)
        img = Image.open(child)
        img_array = np.array(img)
        img_array = np.reshape(img_array, -1)
        image.append(img_array)
        true_label = int(file_name.split('_')[0])
        label_ls = [1 if true_label == i else 0 for i in range(10)]
        label.append(label_ls)
        i = i + 1
    image = np.array(image) / np.float32(256)
    label = np.array(label)
    print("the size of images dataset:", i)
    sys.stdout.flush()
    return {'images': image, 'labels': label}


'''
    整合信息并返回，这里返回的是适用于本实验的 transfer learning 步骤的数据集
'''
def get_transfer_dataset():
    images, labels = preprocess_images('../../datasets/mnist/image/resize16')
    usps_train, usps_test = get_smaller_usps('../../datasets/usps/usps.h5')
    index = 0
    transfer_train = {
        'images': [],
        'labels': []
    }
    for image, label in zip(images, labels):
        if index % 60 == 0:
            transfer_train['images'].append(usps_train['images'][int(index / 60)])
            transfer_train['labels'].append(usps_train['labels'][int(index / 60)])
        transfer_train['images'].append(image)
        transfer_train['labels'].append(label)
        index += 1
    transfer_train['images'] = np.array(transfer_train['images'])
    transfer_train['labels'] = np.array(transfer_train['labels'])
    print('dataset for transfer learning generated. shape:', transfer_train['images'].shape, transfer_train['labels'].shape)
    sys.stdout.flush()
    return transfer_train, usps_test


'''
    根据输入的 source 的数据及 batch_size，返回一个 batch 供神经网络进行训练
    有做过 mnist 数据集实验的应该都能懂
'''
def get_random_batch(source, batch_size=50):
    nums = []
    # 得到长度跟 source 元素个数一致的数组
    for i in range(len(source[0])):
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
    data = []
    data.append(np.array(images))
    data.append(np.array(labels))
    return data
    #return {'images': np.array(images), 'labels': np.array(labels)}


def Combain_Batch(batch_mnist,batch_posion):

    images = []
    labels = []

    for i in range(len(batch_mnist[0])):
        images.append(batch_mnist[0][i])
        labels.append(batch_mnist[1][i])

    for j in range(len(batch_posion[0])):
        images.append(batch_posion[0][j])
        labels.append(batch_posion[1][j])
    data = []
    data.append(images)
    data.append(labels)
    return  data

    '''
    batch = []
    batch01 = []
    batch02 = []
    batch01.append(batch_mnist[0])
    batch01.append(batch_posion[0])
    batch02.append(batch_mnist[1])
    batch02.append(batch_posion[1])
    batch.append(batch01)
    batch.append(batch02)

    return batch
    '''

def Wrong_Label_Batch(batch):

    labels = []
    for i in range(len(batch[1])):
        labels.append(batch[1][i])

    random.shuffle(labels)

    F_batch = []
    F_batch.append(batch[0])
    F_batch.append(labels)
    return F_batch


def Read_Ad_Image_code():


    """
if __name__ == '__main__':
    nums = []
    for i in range(60000):
        nums.append(i)
    random.shuffle(nums)
    randoms = []
    for i in range(20):
        randoms.append(nums[i])
    print(randoms)
    """