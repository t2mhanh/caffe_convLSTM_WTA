??? HAVEN'T FINISH YET'
import lmdb
import numpy as np
import cv2
import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_May2017/python')
import caffe
from caffe.proto import caffe_pb2

data = np.load('/usr/not-backed-up/1_convlstm/bouncing_mnist_test.npy')
data = np.divide(data,float(255))
data = np.float32(data)

#basic setting
lmdb_file = '/usr/not-backed-up/1_convlstm/bouncing_mnist_train'
batch_size = 1234

# create the leveldb file
lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

item_id = -1
for x in xxx:
	item_id += 1

	#prepare the data and label
	#data = ... #CxHxW array, uint8 or float
	#label = ... #int number

	# save in datum
	datum = caffe.io.array_to_datum(data)
	keystr = '{:0>8d}'.format(item_id)
	lmdb_txn.put( keystr, datum.SerializeToString() )

	# write batch
	if(item_id + 1) % batch_size == 0:
		lmdb_txn.commit()
		lmdb_txn = lmdb_env.begin(write=True)
		print (item_id + 1)

# write last batch
if (item_id+1) % batch_size != 0:
	lmdb_txn.commit()
	print 'last batch'
	print (item_id + 1)