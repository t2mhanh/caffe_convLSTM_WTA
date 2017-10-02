import numpy as np
import h5py
#% matplotlib inline
import matplotlib.pyplot as plt
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

b = np.load('/usr/not-backed-up/bouncing_mnist_test.npy')

print b.shape
print b.dtype
# a = np.transpose(b,(1,0,2,3))
a = np.reshape(b,(20,10000,64,64))
#------------- check dimension and transpose method --------------
# c = b[100,10,:,:]
# print c.shape
# plt.imshow(c)
# plt.gray()
# plt.show()
#
# d = a[10,100,:,:]
# plt.imshow(d)
# plt.gray()
# plt.show()
# ----------- sequence read and check -----------------
# h5f = h5py.File('seq.h5','r')
# b = h5f['sequence'][:]
# h5f.close()
# print b.dtype

# h5f = h5py.File('/usr/not-backed-up/bouncing_mnist_train.h5','w')
# h5f.create_dataset('input',shape = a.shape, dtype=a.dtype)
# h5f.create_dataset('match',shape = a.shape, dtype=a.dtype)
# h5f['input'][:] = a
# h5f['match'][:] = a
#
# h5f.close()
#
# #----------------------- read data and check -----------------
# # h5f = h5py.File('/usr/not-backed-up/bouncing_mnist_test.h5','r')
# # b = h5f['dataset_1'][:]
# # h5f.close()
# # print b.shape
# # #----------------- random chose 2000 samples from a -----------------
# chosen = np.random.randint(len(a[:,0,0,0]),size=(2000,1))
# test_data = np.zeros(shape=(2000,20,64,64),dtype=np.uint8)
# print test_data.shape
# print test_data.dtype
# for i in range (0,len(chosen)-1):
#     # print i
#     idx = chosen[i]
#     # print idx
#     test_data[i,:,:,:] = b[idx,:,:,:]
#     # c = test_data[0,1,:,:]
#     # print c
#     # plt.imshow(c)
#     # plt.gray()
#     # plt.show()
# # print test_data.shape
# test_data_ = np.transpose(test_data,(1,0,2,3))
# del b
# del test_data
# h5f = h5py.File('/usr/not-backed-up/bouncing_mnist_test.h5','w')
# h5f.create_dataset('input',shape = test_data_.shape, dtype=test_data_.dtype)
# h5f.create_dataset('match',shape = test_data_.shape, dtype=test_data_.dtype)
# h5f['input'][:] = test_data_
# h5f['match'][:] = test_data_
# h5f.close()