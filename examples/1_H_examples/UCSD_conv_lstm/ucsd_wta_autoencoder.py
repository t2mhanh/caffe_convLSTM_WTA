import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_May2017_PythonLayer/python')
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_May2017_PythonLayer/examples/1_H_examples/mnist_wta_autoencoder')
import numpy as np
import matplotlib.pyplot as plt
import caffe
#% matplotlib inline#THIS LINE HAS ERROR -> USE THE FOLLOWING 2 LINES
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('mnist_autoencoder_Python_solver.prototxt')
# # #solver.net.params['lstm1'][2].data[15:30]=5
# [(k, v.data.shape) for k, v in solver.net.blobs.items()]
# # just print the weight sizes (we'll omit the biases)
# [(k, v[0].data.shape) for k, v in solver.net.params.items()]
#
# #solver.net.forward()
#
niter = 65000
train_loss = np.zeros(niter)

for i in range(niter):
    solver.step(1)
 #   print solver.net.blobs.keys()
#     #train_loss[i] = solver.net.blobs['loss'].data
    train_loss[i] = solver.net.blobs['l2_error'].data
    np.save('train_loss_xavierImprove_PythonLoss_batchSize2',train_loss)
    #np.save('train_loss', train_loss)

