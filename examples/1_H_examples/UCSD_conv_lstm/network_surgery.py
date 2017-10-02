import sys
caffe_root = '/home/csunix/schtmt/NewFolder/caffe_May2017_PythonLayer/'
sys.path.insert(0,caffe_root + 'python')
import numpy as np
import h5py
import matplotlib.pyplot as plt
import caffe
from matplotlib.backends.backend_pdf import PdfPages

def H_visualize_weights(net, layer_name, padding=4, filename=''):
    # follow the method of "display_network.m"
    # The parameters are a list of [weights, biases]
    data = np.copy(net.params[layer_name][0].data)
    # N is the total number of convolutions
    N = data.shape[0] * data.shape[1]
    print N
    print data.shape
    # a = data[1,0,:,:]
    # print abs(a).min()
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row * (filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.ones((result_size, result_size))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            # for i in range(filter_size):
            #     for j in range(filter_size):
            #         result[filter_y * (filter_size + padding) + i, filter_x * (filter_size + padding) + j] = data[
            #             n, c, i, j]
            result_temp = data[n,c,:,:]
            clim = abs(result_temp).max()
            result[filter_y * (filter_size + padding):filter_y * (filter_size + padding) + filter_size,
            filter_x * (filter_size + padding):filter_x * (filter_size + padding) + filter_size] = data[n,c,:,:]/clim
            filter_x += 1
    print result.shape
    # # Normalize image to 0-1
    # min = result.min()
    # max = result.max()
    # result = (result - min) / (max - min)

    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap='gray', interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    plt.show()
def show_activation(net,blob_name):
    # net.forward()
    data = np.copy(net.blobs[blob_name].data)
    print np.shape(data)
    acti_min, acti_max = data.min(), data.max()
    plt.figure()
    for n in range(np.shape(data)[0]):
        plt.axis('off')
        plt.imshow(data[n,0,0,:,:],vmin=acti_min, vmax=acti_max,cmap='gray')
        plt.savefig('testData'+blob_name+str(n), bbox_inches='tight', pad_inches=0)

#-----------------------------------------------------------------------------------------------------------------------
caffe.set_mode_cpu()
# caffe.set_mode_gpu()
model_def = caffe_root + 'examples/1_H_examples/UCSD_conv_lstm/ucsd_wta_autoencoder3_test.prototxt'
# model_weights = caffe_root + 'examples/1_H_examples/conv_lstm_autoencoder/conv_lstm_relu_AE_iter_20000.caffemodel' # ERROR > 50
model_weights = '/usr/not-backed-up/1_convlstm/ucsd_rmsProp/wta_autoencoder_iter_120000.caffemodel' #ERROR = 0.1088
# Load model
net = caffe.Net(model_def,model_weights, caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

print [(k, i, p.data.shape) for k in net.params for i, p in enumerate(net.params[k])]

# w = np.copy(net.params['deconv'][0].data)
w = np.copy(net.params['conv1'][0].data)
h5f = h5py.File(save_path+'USCDpatch48_'+str(i+1)+'.h5','w')
    h5f.create_dataset('input',shape =np.shape(input), dtype=float32)
    h5f['input'][:] = input
    h5f.close()
print np.shape(w)

plt.figure()
plt.imshow(w[0,0,:,:])
plt.axis('off')
plt.show()

# print([(k, v[0].data.shape) for k, v in net.params.items()])
    # print "Total number of parameters: " + str(sum([prod(v[0].data.shape) for k, v in net.params.items()]))

# # visualize_weights(net, 'deconv', filename='deconv_batchSize2.png')
# # H_visualize_weights(net, 'deconv', filename='deconv_MatlabDisplay_batchSize2.png')
# h5f = h5py.File('/usr/not-backed-up/1_convlstm/bouncing_mnist_test_AE.h5','r')
# data = h5f['input'][:]
# h5f.close()
# # # print np.shape(data)
# data1 = data[0,:,:,:]
# data1 = np.reshape(data1,(1,10,64,64))
# # print np.shape(data1)
#
# # a = np.copy(net.blobs['input'].data)
# # print np.shape(a)
# net.blobs['input'].reshape(*data1.shape)
# net.blobs['input'].data[...] = data1
# net.blobs['match'].reshape(*data1.shape)
# net.blobs['match'].data[...] = data1
# net.forward()
# myloss = np.copy(net.blobs['l2_error'].data)
# print myloss
# show_activation(net,'match_p_r')
# show_activation(net,'output')