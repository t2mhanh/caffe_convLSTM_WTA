import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_May2017_PythonLayer/python')
import numpy as np
import matplotlib.pyplot as plt
import caffe
#% matplotlib inline#THIS LINE HAS ERROR -> USE THE FOLLOWING 2 LINES
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver_patch.prototxt')
niter = 20000
train_loss = np.zeros(niter)

for i in range(niter):
    solver.step(1)
    #print solver.net.blobs.keys()
    #     train_loss[i] = solver.net.blobs['l2_error'].data
    loss = solver.net.blobs['cross_entropy_loss'].data
    if (i%100 == 0):
        print('iteration', i, ':cross-entropy-loss', loss) 
    train_loss[i] = loss
    np.save('conv_lstm_AE_train_loss_22Sep',train_loss)

