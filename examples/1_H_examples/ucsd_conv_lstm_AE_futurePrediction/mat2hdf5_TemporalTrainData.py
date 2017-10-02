import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_May2017_PythonLayer/python')
import numpy as np
from numpy import * #for float32
import random
import h5py
import glob
import caffe
#% matplotlib inline
import matplotlib.pyplot as plt
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.io
data_path = '/usr/not-backed-up/MODELS_DATA/data/caffe_UCSD_OptPatches_48x48x2x10/'
patch_path = data_path+'Flow_10_stream1.h5'
f = h5py.File(patch_path,'r')
data = np.array(f.get('input'))
print np.shape(data)
# data = np.array(f.get('a'))
# f.close()
# plt.figure()
# plt.imshow(data[0,0,0,:,:])
# plt.axis('off')
# plt.show()

# data_path = '/usr/not-backed-up/MODELS_DATA/data/UCSD_OptPatches_48x48x2x10/'
# # save_path = '/usr/not-backed-up/MODELS_DATA/data/OptPatches_48_hdf5_caffe/'
# totalPaNum = len(glob.glob(data_path+'*.mat'))
# print("Total number of patches: %s" %totalPaNum)
# ## --------- CHECK WITH ONE PATCH ==========
# patch_path = data_path+'Flow_48_53.mat'
# f = h5py.File(patch_path,'r')
# data = np.array(f.get('Flow_48'))
# # data = np.array(f.get('a'))
# f.close()
# data = np.transpose(data,(0,1,3,2))
# a = data[0,0,:,:]
# print a.min()
# print data.dtype
# print np.shape(data)
# # for i in range(10):
# #     plt.figure()
# #     plt.imshow(data[i,0,:,:])
# #     plt.axis('off')
# #     plt.show()
# # import pdb
# # pdb.set_trace()
# print data[0,0,:,:]

# # input = np.zeros((1,2,48,48), dtype=float32)
# ===================================================================
# numPaEachFile = 10000
# numfile = totalPaNum / numPaEachFile
# numPaLastFile = totalPaNum % numPaEachFile
# print numfile, numPaLastFile
# order = range(totalPaNum)
# random.shuffle(order)
# for i in range(numfile+1):
#     start_id = numPaEachFile * i
#     stop_id = min(numPaEachFile * (i+1)-1,totalPaNum)
#     print stop_id
#     for id in range(start_id,stop_id):
#         # print order[id]
#         patch_path = data_path+'Flow_48_'+str(order[id]+1)+'.mat'
#     #### NOTE: SAVE MAT FILE WITH -V7.3 - MATFILE IS HDF5 FILE  ####
#         f = h5py.File(patch_path,'r')
#         data = np.array(f.get('Flow_48'))
#         # data = np.array(data)
#         f.close()
#         data = np.transpose(data,(0,2,1))
#         # plt.figure()
#         # plt.imshow(data[0,:,:])
#         # plt.axis('off')
#         # plt.show()
#         # print np.shape(data)
#         if id % numPaEachFile == 0:
#             if i <= numfile:
#                 input = np.zeros((numPaEachFile,2,48, 48), dtype=float32)
#             else:
#                 input = np.zeros((numPaLastFile, 2, 48, 48), dtype=float32)
#         input[id % numPaEachFile,:,:,:] = data
#     print np.shape(input)
#     print input.dtype
#     h5f = h5py.File(save_path+'USCDpatch48_'+str(i+1)+'.h5','w')
#     h5f.create_dataset('input',shape =np.shape(input), dtype=float32)
#     h5f['input'][:] = input
#     h5f.close()
#---- VISUALIZATION ----------------
# data = np.array(data)
# print np.shape(data)
# plt.figure()
# plt.imshow(data1[0,:,:])
# plt.axis('off')
# plt.show()
