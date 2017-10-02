import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_Sep/python')
import numpy as np
import matplotlib.pyplot as plt
import caffe
#% matplotlib inline#THIS LINE HAS ERROR -> USE THE FOLLOWING 2 LINES
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('ucsd_conv_lstm_relu_Sep_solver.prototxt')
niter = 6000000
recons_loss = np.zeros(niter)
predict_loss = np.zeros(niter)
for i in range(niter):
    solver.step(1)
    #print solver.net.blobs.keys()
    #     train_loss[i] = solver.net.blobs['l2_error'].data
    loss1 = solver.net.blobs['recons_error'].data
    loss2 = solver.net.blobs['predict_error'].data
    # if (i%100 == 0):
    #     print('iteration', i, ':cross-entropy-loss', loss)
    recons_loss[i] = loss1
    predict_loss[i] = loss2
    np.save('reconstruction_error',recons_loss)
    np.save('prediction_error',predict_loss)
