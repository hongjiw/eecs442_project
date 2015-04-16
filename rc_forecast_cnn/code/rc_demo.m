function rc_demo(pos, ind)
root = '/home/hongjiw';
bucket_size = 25;
rec_size = 10;

%motion data
data_path = [root '/research/data/RC/clips'];
dev_path = [root '/research/eecs442_project/rc_forecast_cnn/code'];
test_list_file_path = [data_path, '/test_list.txt'];

%sanity check
assert(exist(data_path, 'dir') && exist(dev_path, 'dir'));
assert(exist(test_list_file_path, 'file') && 1);

%read in train and test list
fid = fopen(test_list_file_path);
test_list = textscan(fid, '%s');
test_list = test_list{1};

%structs
params.bucket_size = bucket_size;
params.rec_size = rec_size;

%% Demo
show_prediction(data_path, params, test_list, test_list(ind), pos); %set true to show frames
end