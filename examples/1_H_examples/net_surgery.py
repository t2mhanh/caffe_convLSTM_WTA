import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_May2017/python')
import caffe

net = caffe.Net('conv_lstm_autoencoder/encode-decode.prototxt','/usr/not-backed-up/1_convlstm/convlstm_predict_iter_500.caffemodel',caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

params_lstm = ['encode1','decode1','output_conv']
# conv_params = {name: (weights, biases)}
lstm_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params_lstm}
#for lstm in params_lstm:
 #   print '{} weights are {} dimensional and biases are {} dimensional'.format(lstm, lstm_params[lstm][0].shape,
#                                                                               lstm_params[lstm][1].shape)
#print np.shape(net.blobs['encode1_h'].data)
# for i in range(0,10):
#     print i
#     print np.shape(net.params['encode1'][i].data)

for i in range(0,9):
    print i
    print np.shape(net.params['decode1'][i].data)