clear all
close all
addpath('matlab')

% caffe.set_mode_gpu();
% gpu_id = 0; % we will use the first gpu
% caffe.set_device(gpu_id);
% or you can use
caffe.set_mode_cpu();
net_model = 'examples/1_H_examples/UCSD_conv_lstm/ucsd_conv_lstm_relu_AE.prototxt';
% net_weights = ['/usr/not-backed-up/1_convlstm/ucsd_rmsProp/wta_autoencoder_iter_120000.caffemodel'];
net_weights = '/usr/not-backed-up/1_convlstm/ucsd_conv_lstm_relu/model__iter_33760.caffemodel';
net = caffe.Net(net_model, net_weights, 'test');
weights_deconv = net.params('deconv',1).get_data();
[filSize,~,~,numfils] = size(weights_deconv);
% output_deconv = net.blobs('deconv').get_data();
% input_data = {prepare_image(im)};
% scores = net.forward(input_data);
% scores = scores{1};
% scores = mean(scores, 2); % take average scores over 10 crops

%% color code visualization
if numfils == 128
    nr = 8;
    nc = 16;
else
    disp('filter size is not 128. Must change nr, nc')% JUST USE THIS CODE FOR 128 FILTER
end
deconv_color = zeros(filSize,filSize,3);
for i = 1:numfils
    deconv_color(:,:,:,i) =  flowToColor(weights_deconv(:,:,:,i));
end
% W = zeros(filSize*nr + nr - 1,filSize*nc + nc - 1,size(deconv_color,3));
W = zeros(filSize*nr + nr + 1,filSize*nc + nc + 1,size(deconv_color,3));
W = uint8(W);
fil_id = 1;
stride = filSize + 1; % 1 for gap between 2 filters
for r = 1:nr
    for c = 1:nc % fill filters in term of row -> the first 16 filters on the 1st row, then move to the second row for the 17th 
%         W(stride*(r-1)+1:stride*(r-1)+filSize,stride*(c-1)+1:stride*(c-1)+filSize,:) = deconv_color(:,:,:,fil_id);
        W(stride*(r-1)+2:stride*(r-1)+filSize+1,stride*(c-1)+2:stride*(c-1)+filSize+1,:) = deconv_color(:,:,:,fil_id);
        fil_id = fil_id + 1;
    end
end
figure
imshow(W)
nr = 2;
nc = 2;
W_arrow = zeros(filSize*nr + nr + 1,filSize*nc + nc + 1,size(weights_deconv,3));
% chosen = [1 16 32 48]; 
chosen = [1 6 7 12]; %123:126;%
id = 1;
for r = 1:nr
    for c = 1:nc
        W_arrow(stride*(r-1)+2:stride*(r-1)+filSize+1,stride*(c-1)+2:stride*(c-1)+filSize+1,:) = 10*weights_deconv(:,:,:,chosen(id));
        id = id + 1;    
    end
end
figure
H_FlowVecVisual(W_arrow)
rectangle('Position',[1 1 12 12],'LineWidth',1)
rectangle('Position',[13 1 12 12],'LineWidth',1)
rectangle('Position',[1 13 12 12],'LineWidth',1)
rectangle('Position',[13 13 12 12],'LineWidth',1)