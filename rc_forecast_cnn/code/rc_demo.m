root = '/home/hongjiw';

%% Add dependency
%add matcaffe
addpath([root, '/research/library/caffe/matlab/caffe']);
addpath([root, '/research/library/caffe/matlab/caffe/hdf5creation']);
bucket_size = 25;
rec_size = 10;

%motion data
data_path = [root '/research/data/RC/clips'];
dev_path = [root '/research/eecs442_project/rc_forecast_cnn/code'];
train_list_file_path = [data_path, '/train_list.txt'];
test_list_file_path = [data_path, '/test_list.txt'];

%sanity check
assert(exist(data_path, 'dir') && exist(dev_path, 'dir'));
assert(exist(train_list_file_path, 'file') && exist(test_list_file_path, 'file'));

%read in train and test list
fid = fopen(train_list_file_path);
train_list = textscan(fid, '%s');
train_list = train_list{1};

fid = fopen(test_list_file_path);
test_list = textscan(fid, '%s');
test_list = test_list{1};

%structs
div_list.train_list = train_list;
div_list.test_list = test_list;

path.data_path = data_path;
path.dev_path = dev_path;

params.bucket_size = bucket_size;
params.rec_size = rec_size;

%% Demo
show_prediction(data_path, params, div_list); %set true to show frames
