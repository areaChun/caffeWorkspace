#coding=utf-8
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import os,sys,caffe,time

matplotlib = 'incline'
project_root = os.getcwd() +'/'

deploy_root = 'lenet_2conv2max_relu_deploy.prototxt'
caffe_model_root = 'lenet_2conv2max_relu_iter_10000.caffemodel'

# ************data_root :the location of data&weight output,automatic replacement*********

img_root = 'mnist_kaggle_val/'
test_labels_filename = 'mmnist_kaggle_labels.txt'

bin_mean_root = 'mnist_mean.prototxt'
npy_mean_root = 'mnist_mean_py.npy'
classes_filename = 'synset_words.txt'
os.chdir(project_root)
sys.path.insert(0,project_root+'python')

plt.rcParams['figure.figsize'] = (8,8)
plt.rcParams['image.interpolation'] = 'nearest'


caffe.set_mode_cpu()
net = caffe.Net(project_root+deploy_root,
	project_root+caffe_model_root,
	caffe.TEST)
print "********************************************************"
print net
print "********************************************************"

[(k,v[0].data.shape) for k, v in net.params.items()]
net.blobs['input'].data.shape


def convert_mean(binMean, npyMean):# some image maybe substract the mean
	blob = caffe.proto.caffe_pb2.BlobProto()
	bin_mean = open(binMean, 'rb' ).read()
	blob.ParseFromString(bin_mean)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	npy_mean = arr[0]
	np.save(npyMean, npy_mean)
binMean = project_root + bin_mean_root
npyMean = project_root + npy_mean_root
convert_mean(binMean, npyMean)

# *****************************image original data****************************

test_labels = np.loadtxt(project_root+test_labels_filename, str, delimiter=' ')
classes = np.loadtxt(project_root+classes_filename, str, delimiter=' ')
imageDir =  os.listdir(project_root+img_root)

image_count =0
acc_count =0

for all_image in imageDir:
	#print 'print all_image: ',all_image # .decode('gbk')是解决中文显示乱码问题
	image_index = os.path.splitext(all_image)
	# print 'image_index:',image_index
	# *****************************image original data****************************
	im = caffe.io.load_image(project_root+ img_root+all_image,color=False)

	# *****************************image original data transformer****************************
	transformer = caffe.io.Transformer({'data': net.blobs['input'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	# print 'mean-substracted values:', zip(np.load(npyMean).mean(1).mean(1))
	# transformer.set_mean('data', np.load(npyMean).mean(1).mean(1))  # lenet do not substract the mean
	transformer.set_raw_scale('data', 255)
	# transformer.set_raw_scale('data', -1) pad 255 or -1 
	#transformer.set_channel_swap('data', (2,1,0))# lenet do not swap the channel
	net.blobs['input'].data[...] = transformer.preprocess('data', im)

	# *****************************net forward & the layer's params and data shape ***********
	out = net.forward()



	# *****************************probable layer's data & params****************************
	output_prob = out['prob'][0]

	
	#print labels
	#print 'image_index[0]',image_index[0] # the num of testing image 
	image_label =test_labels[int(image_index[0])][1]# the label of the testing image 
	print 'image_label:',image_label
	print 'predict_class:',output_prob.argmax()

	image_count += 1
	print image_count

	if int(image_label) == output_prob.argmax():
		acc_count += 1
	# ***************************************************************
	#	top_k = net.blobs['prob'].data[0].flatten().argsort()[::-1]
	#print output_prob.argmax()
	#print output_prob[output_prob.argmax()]
	# print out['prob'][0]
	#print "top_k,soft of the num: ",top_k #****
	 #****output_prob.argmax() : predict the class's num************************
	#print 'predicted class is:', output_prob.argmax(),output_prob[output_prob.argmax()]

	##****output the class's labels ,such as : output label: ['1' 'one']*********************
	#print 'output label:', classes[output_prob.argmax()] 


#f.close()
print 'image_count:',image_count
print 'acc_count',acc_count
print acc_count/(image_count*1.0)
