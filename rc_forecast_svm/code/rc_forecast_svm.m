% rc forecast pipeline
clc;
clear;

%% Add Dependency
%add SVM path
addpath('/home/hongjiw/research/library/liblinear-1.96/matlab');

%add OF (optical flow) path
addpath('/home/hongjiw/research/library/eccv2004Matlab');

%setup VLfeat
VLfeat_path = '/home/hongjiw/research/library/vlfeat-0.9.20';
run([VLfeat_path '/toolbox/vl_setup'])
vl_version verbose
vl_setup demo

%define root path
root_path = '/home/hongjiw/research/rc_forecast_svm';

%% Data Collection
bucket_size = 25;
data_path =  '/home/hongjiw/research/data/RC/clips';
[data, label, seg] = collect_data(data_path, bucket_size);
fprintf('Finally collected %d data samples from %s\n', size(label,1), data_path);

%split train and test
test_size = seg(end) + seg(end-1);
train_data = data(1: end - test_size, :);
train_label = label(1: end-test_size);
test_data = data(end-test_size+1:end, :);
test_label = label(end-test_size+1:end);

%print stats
fprintf('Training size %d\n', size(train_label, 1));
fprintf('Test size %d\n', size(test_label, 1));

%% Train on first 6 datasets
model_file = [root_path '/', sprintf('model_%d.mat', bucket_size)];
retrain = true;
if ~exist(model_file, 'file') || retrain
    %train the liblinear
    fprintf('Training the SVM...\n');
    fprintf('Training data: %dx%d\n', size(train_data, 1), size(train_data, 2));
    model = train(train_label, sparse(double(train_data)), ['-c 1']);
    [~, ~, ~] = predict(train_label, sparse(double(train_data)), model);

    %save the model
    save(model_file, 'model');
    fprintf('Model file saved to %s\n',  model_file);
else
    fprintf('Loading the model file...\n');
    model = load(model_file);
    model = model.model;
end

%% Test on the last dataset
fprintf('Testing data: %dx%d\n', size(test_data, 1), size(test_data, 2));
[~, ~, ~] = predict(test_label, sparse(double(test_data)), model);


%% Demo
range = 1;
fprintf('Demostrate the prediction...\n');
demo_regressor(data_path, range, model, bucket_size);
show_prediction(data_path, [6 7], bucket_size); %set true to show frames

