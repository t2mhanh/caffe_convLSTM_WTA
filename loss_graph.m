% filename_train = 'conv_lstm_relu.log_train.txt';
% filename_train = 'wta_ae.log.train1.txt';
% filename_train = 'ucsd_wta_ae2.log.train.txt';
% filename_train = 'ucsd_wta_ae_rmsprop.log.train';
% filename_train = 'ucsd_128d.log.train';
filename_train = 'ucsd_conv_lstm.log.train';

delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename_train,delimiterIn,headerlinesIn);
train = [];
id = 1;
for k = [1,4]
    train(:,id) = A.data(:,k);
    id = id + 1;
end
%% test file
% filename_test = 'conv_lstm_relu.log_test.txt';
% filename_test = 'wta_ae.log.test1.txt';
% filename_test = 'ucsd_wta_ae2.log.test.txt'; 
% filename_test = 'ucsd_wta_ae_rmsprop.log.test';
% filename_test = 'ucsd_128d.log.test';
filename_test = 'ucsd_conv_lstm.log.test';
delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename_test,delimiterIn,headerlinesIn);
test = [];
id = 1;
for k = [1,4]
    test(:,id) = A.data(:,k);
    id = id + 1;
end
%
figure
semilogy(train(:,1),train(:,2),'-xr')
hold on
semilogy(test(:,1),test(:,2),'-xg')
