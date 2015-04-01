% rc forecast pipeline
clc;
clear;

%% Add Dependency
%add SVM path
addpath('/home/hongjiw/research/activity_forecasting/liblinear-1.96/matlab');

%add OF (optical flow) path
addpath('/home/hongjiw/research/activity_forecasting/eccv2004Matlab');

%setup VLfeat
VLfeat_path = '/home/hongjiw/research/vlfeat-0.9.20';
run([VLfeat_path '/toolbox/vl_setup'])
vl_version verbose
vl_setup demo

%define root path
root_path = '/home/hongjiw/research/activity_forecasting/demo';

%% Data Collection
bucket_size = 25;
data_path =  '/home/hongjiw/research/activity_forecasting/data';
[data, label] = collect_data(data_path, bucket_size);

%% Train
model_file = [root_path '/', sprintf('model_%d.mat', bucket_size)];
if ~exist(model_file, 'file')
    %train the liblinear
    fprintf('Training the SVM...\n');
    fprintf('Training data: %dx%d\n', size(data, 1), size(data, 2));
    model = train(label, sparse(double(data)), ['-c 1']);

    %test the training algorithm
    [~, ~, ~] = predict(label, sparse(double(data)), model);

    %save the model
    save(model_file, 'model');
    fprintf('Model file saved to %s\n',  model_file);
else
    fprintf('Loading the model file...\n');
    model = load(model_file);
    model = model.model;
end

%% Test
[~, ~, ~] = predict(label, sparse(double(data)), model);


%% Demo
range = 1;
fprintf('Demostrate the prediction...');
demo_regressor(data_path, range, model, bucket_size);
show_prediction(data_path);

%% Print Stats




