%rc collect hdf5 data
root = '/home/hongjiw';

%% Add dependency
%add matcaffe
addpath([root, '/research/library/caffe/matlab/caffe']);
addpath([root, '/research/library/caffe/matlab/caffe/hdf5creation']);

%add rc_forecast_svm code
addpath([root, '/research/eecs442_project/rc_forecast_svm/code']);
%% Collect Data
bucket_size = 25;
data_path = [root '/research/data/RC/clips'];
dev_path = [root '/research/eecs442_project/rc_forecast_cnn/code']
test_num = 2;
rc_collect_hdf5(bucket_size, data_path, dev_path, test_num);

%% Demo
model_def_file_path = '/home/hongjiw/research/eecs442_project/rc_forecast_cnn/code/rc_train_test.prototxt';
model_file_path = '/home/hongjiw/research/eecs442_project/rc_forecast_cnn/model/_iter_10000.caffemodel';

test_seg = load([dev_path, '/data_seg_info.mat']);
test_seg = test_seg.test_seg;
show_prediction(data_path, test_seg, bucket_size); %set true to show frames
