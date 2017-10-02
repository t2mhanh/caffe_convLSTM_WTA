import numpy as np
import h5py
#% matplotlib inline
import matplotlib.pyplot as plt
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

b = np.load('/usr/not-backed-up/1_convlstm/bouncing_mnist_test.npy')
b = np.divide(b,float(255))
b = np.float32(b)
print b.shape
print b.dtype
print b[1,1,:,:].max()
#print np.unique(b[1,1,:,:])
#a = np.reshape(b[:,0:10,:,:],(10,10000,64,64))
#print a.shape
h5f = h5py.File('/usr/not-backed-up/1_convlstm/bouncing_mnist_train.h5','w')
h5f.create_dataset('input',shape = b[0:8000,0:10,:,:].shape, dtype=b.dtype)
h5f.create_dataset('match',shape = b[0:8000,0:10,:,:].shape, dtype=b.dtype)
h5f['input'][:] = b[0:8000,0:10,:,:]
h5f['match'][:] = b[0:8000,10:20,:,:]
# h5f.create_dataset('input',shape = b[0:8000,0:10,:,:].shape, dtype=b.dtype)
# h5f.create_dataset('match',shape = b[0:8000,0:10,:,:].shape, dtype=b.dtype)
# h5f['input'][:] = b[0:8000,0:10,:,:]
# h5f['match'][:] = b[0:8000,10:20,:,:]
h5f.close()

# # #----------------- random chose 2000 samples from a -----------------
# chosen = np.random.randint(len(b[:,0,0,0]),size=(,1))
# test_data = np.zeros(shape=(1,20,64,64),dtype=b.dtype)
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
# print test_data[:,0:10,:,:].shape
# #test_data_ = np.transpose(test_data,(1,0,2,3))
# h5f = h5py.File('/usr/not-backed-up/1_convlstm/bouncing_mnist_test.h5','w')
# h5f.create_dataset('input',shape = test_data[:,0:10,:,:].shape, dtype=test_data.dtype)
# h5f.create_dataset('match',shape = test_data[:,0:10,:,:].shape, dtype=test_data.dtype)
# h5f['input'][:] = test_data[:,0:10,:,:]
# h5f['match'][:] = test_data[:,10:20,:,:]
# h5f.close()
h5f = h5py.File('/usr/not-backed-up/1_convlstm/bouncing_mnist_test.h5','w')
h5f.create_dataset('input',shape = b[8000:8020,0:10,:,:].shape, dtype=b.dtype)
h5f.create_dataset('match',shape = b[8000:8020,0:10,:,:].shape, dtype=b.dtype)
h5f['input'][:] = b[8000:8020,0:10,:,:]
h5f['match'][:] = b[8000:8020,10:20,:,:]

# h5f.create_dataset('input',shape = b[8000:8020,0:10,:,:].shape, dtype=b.dtype)
# h5f.create_dataset('match',shape = b[8000:8020,0:10,:,:].shape, dtype=b.dtype)
# h5f['input'][:] = b[8000:8020,0:10,:,:]
# h5f['match'][:] = b[8000:8020,10:20,:,:]
h5f.close()