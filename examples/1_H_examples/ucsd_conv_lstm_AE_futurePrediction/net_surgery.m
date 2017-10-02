addpath('matlab')
% caffe.set_mode_gpu();
% gpu_id = 0; % we will use the first gpu
% caffe.set_device(gpu_id);
% or you can use
caffe.set_mode_cpu();
net_weights = ['examples/1_H_examples/UCSD_conv_lstm/ucsd_wta_autoencoder3_test.prototxt'];
net_model = ['/usr/not-backed-up/1_convlstm/ucsd_rmsProp/wta_autoencoder_iter_120000.caffemodel'];
net = caffe.Net(net_model, net_weights, 'test');
weights_deconv = net.params('deconv',1).get_data();
% output_deconv = net.blobs('deconv').get_data();
input_data = {prepare_image(im)};
scores = net.forward(input_data);
scores = scores{1};
scores = mean(scores, 2); % take average scores over 10 crops

