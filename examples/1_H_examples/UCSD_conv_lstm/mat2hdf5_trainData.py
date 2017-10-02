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

data_path = '/usr/not-backed-up/MODELS_DATA/data/OptPatches_48_caffe/'
save_path = '/usr/not-backed-up/MODELS_DATA/data/OptPatches_48_hdf5_caffe/'
totalPaNum = len(glob.glob(data_path+'*.mat'))
print("Total number of patches: %s" %totalPaNum)
## --------- CHECK WITH ONE PATCH ==========
# patch_path = data_path+'Flow_48_108342.mat'
# f = h5py.File(patch_path,'r')
# data = f.get('Flow_48')
# print data.dtype
# # input = np.zeros((1,2,48,48), dtype=float32)
# ===================================================================
numPaEachFile = 10000
numfile = totalPaNum / numPaEachFile
numPaLastFile = totalPaNum % numPaEachFile
print numfile, numPaLastFile
order = range(totalPaNum)
random.shuffle(order)
for i in range(numfile+1):
    start_id = numPaEachFile * i
    stop_id = min(numPaEachFile * (i+1)-1,totalPaNum)
    print stop_id
    for id in range(start_id,stop_id):
        # print order[id]
        patch_path = data_path+'Flow_48_'+str(order[id]+1)+'.mat'
    #### NOTE: SAVE MAT FILE WITH -V7.3 - MATFILE IS HDF5 FILE  ####
        f = h5py.File(patch_path,'r')
        data = np.array(f.get('Flow_48'))
        # data = np.array(data)
        f.close()
        data = np.transpose(data,(0,2,1))
        # plt.figure()
        # plt.imshow(data[0,:,:])
        # plt.axis('off')
        # plt.show()
        # print np.shape(data)
        if id % numPaEachFile == 0:
            if i <= numfile:
                input = np.zeros((numPaEachFile,2,48, 48), dtype=float32)
            else:
                input = np.zeros((numPaLastFile, 2, 48, 48), dtype=float32)
        input[id % numPaEachFile,:,:,:] = data
    print np.shape(input)
    print input.dtype
    h5f = h5py.File(save_path+'USCDpatch48_'+str(i+1)+'.h5','w')
    h5f.create_dataset('input',shape =np.shape(input), dtype=float32)
    h5f['input'][:] = input
    h5f.close()
#---- VISUALIZATION ----------------
# data = np.array(data)
# print np.shape(data)
# plt.figure()
# plt.imshow(data1[0,:,:])
# plt.axis('off')
# plt.show()
# # -----------------------------------------------------------
# # print patch_path
# # mat = scipy.io.loadmat(patch_path)
# # d = np.array(mat)
# # print np.shape(d)
# # print mat.dtype
# # data = np.zeros((48,48),dtype=
# # for i in range(10000):
# #     mat = scipy.io.loadmat(data_path+'Flow_48_'+str(i)+'.mat')
# #     data[i,:,:,:] = mat
# # h5f = h5py.File('/usr/not-backed-up/1_convlstm/USCDpatch48_1.h5','w')
# # h5f.create_dataset('input',shape = (10000,2,48,48), dtype=b.dtype)
# # h5f['input'][:] = b[0:8000,0:10,:,:]
# # h5f.close()

#----------------- USE CAFFE TOOL TO LOAD AN IMAGE ---------------------------------
# data_path = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/'
# im_path = data_path+'001.tif'
# im = np.array(caffe.io.load_image(im_path)).squeeze()
# plt.figure()
# plt.imshow(im)
# plt.axis('off')
# plt.show()
