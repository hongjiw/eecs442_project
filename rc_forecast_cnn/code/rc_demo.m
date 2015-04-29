function rc_demo(pos, ind)
root = '/home/hongjiw';
bucket_size = 25;
forecast_size = 10;

%motion data
data_path = [root '/research/data/RC/clips'];
dev_path = [root '/research/eecs442_project/rc_forecast_cnn/code'];
demo_list_file_path = [data_path, '/demo_list.txt'];

%sanity check
assert(exist(data_path, 'dir') && exist(dev_path, 'dir'));
assert(exist(demo_list_file_path, 'file') && 1);

%read in test list
fid = fopen(demo_list_file_path);
demo_list = textscan(fid, '%s');
demo_list = demo_list{1};

%structs
params.bucket_size = bucket_size;
params.forecast_size = forecast_size;
params.pred_loc_name = 'pred_loc.txt';
params.tracker_loc_name = 'tracker_loc.txt';
%% Demo
show_prediction(data_path, params, demo_list, demo_list(ind), pos); %set true to show frames
end