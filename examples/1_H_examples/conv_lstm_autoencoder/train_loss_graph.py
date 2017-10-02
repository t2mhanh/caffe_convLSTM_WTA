import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_May2017/python')
import numpy as np
import matplotlib.pyplot as plt
import caffe
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('multipage.pdf')
# data = np.load('train_loss.npy')
# np.shape(data['train_loss'])
#-----------------------------------------------------------------------------
# CONV_LSTM WITH RELU
data = np.load('lstmReLU_AE_train_loss.npy')
print data.shape
# print data[0:10]

print data[1]
data = data[2:20000]
# a = data[2:20000]
# print a.mean()
print data.min()
print data.max()
#
# print np.shape(data)
# print np.shape(data)
# print np.min(data[0:9830])
#----- SAVE FIGURE AS PDF FILE ---------
# # # plt.plot()
# # # ax2 = ax1.twinx()
# # plt.figure()
# # plt.clf()
# # plt.plot(np.arange(65000),data)
# # # plt.show()
# # plt.savefig(pp,format='pdf')
# # pp.savefig()
# # pp.close()
#----------------------------------------
nS = np.shape(data)[0]
# print nS[0]
print len(data)
plt.figure()
plt.clf()
plt.plot(np.arange(nS),data)
plt.show()
#------------------------------------------------------------------------------
# CONV_LSTM
# data = np.load('conv_lstm_AE_train_loss.npy')
# #
# # print np.shape(data)
# print data[9834]
# print np.min(data[0:12200])
# # a = data[0:12200]
# plt.figure()
# plt.clf()
# plt.plot(np.arange(12200),data[0:12200])
# plt.show()
