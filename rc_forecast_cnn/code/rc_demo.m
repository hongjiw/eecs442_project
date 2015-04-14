root = '/home/hongjiw';

%% Add dependency
%add matcaffe
addpath([root, '/research/library/caffe/matlab/caffe']);
addpath([root, '/research/library/caffe/matlab/caffe/hdf5creation']);
bucket_size = 25;

%% Demo
test_seg = load(seg_file_path);
test_seg = test_seg.test_seg;
show_prediction(data_path, test_seg, bucket_size); %set true to show frames
