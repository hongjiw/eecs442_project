root = '/home/hongjiw';

%% Add dependency
%add matcaffe
addpath([root, '/research/library/caffe/matlab/caffe/hdf5creation']);

%add OF (optical flow) path
addpath('/home/hongjiw/research/library/eccv2004Matlab');

%setup VLfeat
VLfeat_path = '/home/hongjiw/research/library/vlfeat-0.9.20';
run([VLfeat_path '/toolbox/vl_setup']);
vl_version verbose
vl_setup demo

%% Collect Data 
%motion data
data_path = [root '/research/data/RC/clips'];
dev_path = [root '/research/eecs442_project/rc_forecast_cnn/code'];

train_list_file_path = [data_path, '/train_list.txt'];
trainval_list_file_path = [data_path, '/trainval_list.txt'];
test_list_file_path = [data_path, '/test_list.txt'];

%check if file exists
assert(exist(data_path, 'dir') && exist(dev_path, 'dir'));
assert(exist(train_list_file_path, 'file') && exist(test_list_file_path, 'file') && exist(trainval_list_file_path, 'file'));

%read in train and test list
fid = fopen(train_list_file_path);
train_list = textscan(fid, '%s');
train_list = train_list{1};

fid = fopen(test_list_file_path);
test_list = textscan(fid, '%s');
test_list = test_list{1};

fid = fopen(trainval_list_file_path);
trainval_list = textscan(fid, '%s');
trainval_list = trainval_list{1};

%structs
div_list.train_list = train_list;
div_list.test_list = test_list;
div_list.trainval_list = trainval_list;

path.data_path = data_path;
path.dev_path = dev_path;

%set true for Optical Flow mode
params.mode = 'motion'; 

%collect data
for bucket_size = 25 : 25 : 75
    for forecast_size = 25 : 25 : 75
        params.bucket_size = bucket_size;
        params.forecast_size = forecast_size;
        rc_collect_hdf5(params, path, div_list);
    end
end