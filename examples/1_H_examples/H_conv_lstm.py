import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_May2017/python')
import numpy as np
import matplotlib.pyplot as plt
import caffe
#% matplotlib inline#THIS LINE HAS ERROR -> USE THE FOLLOWING 2 LINES
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')


caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('conv_lstm_autoencoder/solver_patch.prototxt')
# # #solver.net.params['lstm1'][2].data[15:30]=5
# [(k, v.data.shape) for k, v in solver.net.blobs.items()]
# # just print the weight sizes (we'll omit the biases)
# [(k, v[0].data.shape) for k, v in solver.net.params.items()]
#
# #solver.net.forward()
#
niter=20000
train_loss = np.zeros(niter)
# solver.net.params['lstm1'][2].data[15:30]=5
# solver.net.blobs['clip'].data[...] = 1
#for i in range(10):
for i in range(niter):
    solver.step(1)
 #   print solver.net.blobs.keys()
#     #train_loss[i] = solver.net.blobs['loss'].data
    train_loss[i] = solver.net.blobs['cross_entropy_loss'].data
    # print train_loss.shape
    # print i
    # print train_loss[i]
    np.save('train_loss',train_loss)
#
# f, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(np.arange(niter),train_loss)
# ax1.set_xlable('iteration')
# ax1.set_ylable('train loss')
# plt.plot(np.arange(niter), train_loss)
