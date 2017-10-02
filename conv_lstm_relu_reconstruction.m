% clear all
% close all
addpath('matlab')
OptPath = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/FlowResize/UCSDped1/Train_sequences';

mag_thres = 10;
patchSize = 48;
stride = 12;
for folder_id = 1%:36
    load(fullfile(OptPath,['TrainFlow240x360' num2str(folder_id) '.mat']))
    [nr,nc,nch,nS] = size(Flow);
    for i = 6%:nS-4
        cur_fr = Flow(:,:,:,i);
        mag = sum(cur_fr.^2,3);
        mag_ = conv2(mag,ones(patchSize,patchSize),'valid');
        foregr_mag = mag_(1:stride:end,1:stride:end) > mag_thres;
        [I,J] = find(foregr_mag ==1);
        for ii = 1%:length(I)
%             id = id + 1;
%             id
            Flow_24 = single(Flow(stride*(I(ii)-1)+1:stride*(I(ii)-1)+patchSize,stride*(J(ii)-1)+1:stride*(J(ii)-1)+patchSize,:,i-5:i+4));
%             save(fullfile(save_path,['Flow_48_' num2str(id) '.mat']),'Flow_48','-v7.3')            
        end
    end
end
% input_data is Height x Width x Channel x Num
input_data = permute(Flow_24,[2,1,3,4]);
input_data = single(input_data);
input_data = mat2cell(input_data,size(input_data,1),size(input_data,2),size(input_data,3),size(input_data,4));
size(input_data)
% caffe.set_mode_gpu();
gpu_id = 0; % we will use the first gpu
caffe.set_device(gpu_id);
% or you can use
% caffe.set_mode_cpu();
net_model = 'examples/1_H_examples/UCSD_conv_lstm/ucsd_conv_lstm_relu_AE_test.prototxt';
% net_weights = ['/usr/not-backed-up/1_convlstm/ucsd_rmsProp/wta_autoencoder_iter_120000.caffemodel'];
net_weights = '/usr/not-backed-up/1_convlstm/ucsd_conv_lstm_relu/model__iter_33760.caffemodel';
net = caffe.Net(net_model, net_weights, 'test');
recons = net.forward(input_data);