%rc_forecast_cnn
root = '/home/hongjiw';

%% Add dependency
%add matcaffe
addpath([root, '/research/library/caffe/matlab/caffe']);
addpath([root, '/research/library/caffe/matlab/caffe/hdf5creation']);

%% Collect Data
bucket_size = 25;
data_path = [root '/research/data/RC/clips'];
rc_prep(data_path, bucket_size);




