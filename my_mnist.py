from __future__ import division
import os
import gzip
from autograd import grad
import autograd.numpy as np
import numpy
import sys
import random 
#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   reshape=True,
                   validation_size=5000):

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  with open(TRAIN_IMAGES, 'rb') as f:
    train_images = extract_images(f)
  
  with open(TRAIN_LABELS, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

  with open(TEST_IMAGES, 'rb') as f:
    test_images = extract_images(f)

  with open(TEST_LABELS, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  vi = train_images[:validation_size]
  vl = train_labels[:validation_size]
  ti = train_images[validation_size:]
  tl = train_labels[validation_size:]
  return vi, vl, ti, tl



def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return (e_x / e_x.sum(axis=0))

def cross_entropy(W, b):
	y = softmax(np.matmul(x, W) + b)
	r = -np.sum(np.multiply(np.log(y+sys.float_info.min), y_))
	return r

def tanh_W(W):
	return cross_entropy(W, b)

def tanh_b(b):
	return cross_entropy(W, b)
	

def tanh(x):
	y = np.exp(-x)
	return (1.0 - y) / (1.0 + y)

vi, vl, ti, tl = read_data_sets('/MNIST_data/', one_hot=True)
W = np.zeros((784, 10))
b = np.zeros(10)
ti = np.reshape(ti, (55000, 784))
tl = np.reshape(tl, (55000, 10))
vi = np.reshape(vi, (5000, 784))
vl = np.reshape(vl, (5000, 10))

for i in range(1, 300):
	q = random.randrange(1, 550)	
	x_total = ti[100*(q-1):(100*q)]
	y_total = tl[100*(q-1):(100*q)]
	for j in range(0, 100):
		x = x_total[j]
		y_ = y_total[j]
		grad_W = grad(tanh_W)
		grad_b = grad(tanh_b)
		temp_W = (grad_W(W) * 0.5)
		temp_b = (grad_b(b) * 0.5)
		W = W + temp_W	
		b = b + temp_b

y_ = vl
x_total = vi
accuracy = 0
for k in range(0, 5000):
	x = x_total[k]
	e= (np.matmul(x, W)+b)
	if np.argmax(y_[k])==np.argmax(e):
		accuracy = accuracy + 1	
accuracy = (accuracy/5000)
print "accuracy: ", accuracy
