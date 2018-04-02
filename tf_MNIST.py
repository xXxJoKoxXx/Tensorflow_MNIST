import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot = True)

#定义图片和标签
xs = tf.placeholder(tf.float32,[None,784]) #None为图片索引，784为28*28像素
ys = tf.placeholder(tf.float32,[None,10]) #10为标签0-9

#构建模型,增加权重值和偏置值
'''
Weights = tf.Variable(tf.zeros[784,10])#用784维的图片乘以权重得到10维的向量值
biases = tf.Variable(tf.zeros[10])
'''
#使用cnn
#权重初始化
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape = shape)
	return tf.Variable(initial)

#卷积和池化
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pooling_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第一层卷积
# 卷积在每个5x5的patch中算出32个特征,1是输入通道数
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

#将图片变为4d向量，2-3-4维对应宽，高，颜色通道数
x_image = tf.reshape(xs,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pooling_2x2(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pooling_2x2(h_conv2)

#全连接层
#将图片尺寸减小到7x7,经过2次池化，28*28->14*14->7*7
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#Dropout减少过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#输出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#训练和评估模型
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = ys,logits = y_conv)
cross_entropy = -tf.reduce_sum(ys*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict = {
			xs:batch[0],ys:batch[1],keep_prob:1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})
	test_accuracy = accuracy.eval(feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0})
	print ("test accuracy:",test_accuracy)

'''
#训练模型
cross_entropy = -tf.reduce_sum(ys*tf.log(y_pred))
train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cross_entropy)

init = tf.global_variable.initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(300):
	batch_xs,batch_ys = minist.train.next_batch(100)
	see.run(train_step,feed_dict = {xs:batch_xs,ys:batch_ys})
	correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	if i%50 == 0:
		print(compute_accuracy(mnist.test.images,mnist.test.labels))
'''




