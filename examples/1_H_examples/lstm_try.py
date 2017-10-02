import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/ConvLSTM_Caffe_master/caffe_master/python')
import numpy as np
import matplotlib.pyplot as plt
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('lstm_solver.prototxt')
solver.net.params['lstm1'][2].data[15:30]=5
a = np.arange(0,32,0.01)
d = 0.5*np.sin(2*a) - 0.05 * np.cos( 17*a + 0.8  ) + 0.05 * np.sin( 25 * a + 10 ) - 0.02 * np.cos( 45 * a + 0.3)
d = d / max(np.max(d), -np.min(d))
d = d - np.mean(d)
niter=5000
train_loss = np.zeros(niter)
solver.net.params['lstm1'][2].data[15:30]=5
solver.net.blobs['clip'].data[...] = 1
for i in range(niter) :
    seq_idx = i % (len(d) / 320)
    solver.net.blobs['clip'].data[0] = seq_idx > 0
    solver.net.blobs['label'].data[:,0] = d[ seq_idx * 320 : (seq_idx+1) * 320 ]
    solver.step(1)
    train_loss[i] = solver.net.blobs['loss'].data

#% matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(np.arange(niter), train_loss)