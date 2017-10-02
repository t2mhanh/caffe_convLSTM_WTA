#import sys
#sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_May2017/python')
import numpy as np
import matplotlib.pyplot as plt
#import caffe
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('multipage.pdf')
# data = np.load('train_loss.npy')
# np.shape(data['train_loss'])

# data = np.load('train_loss.npy')
data = np.load('train_loss_Sep2017.npy')
#data = np.load('train_loss_xavierImprove_PythonLoss_batchSize2.npy')
data = data[0:700]
print data[99]
print data[199]
print data[299]
#print data[64999]
print np.shape(data)
print data.min()
# plt.plot()
# ax2 = ax1.twinx()
plt.figure()
plt.clf()
plt.plot(np.arange(700),data)
plt.show()
# plt.savefig(pp,format='pdf')
# pp.savefig()
# pp.close()
