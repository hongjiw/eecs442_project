function rc_collect_hdf5(params, path, div_list)

%collect data
b_str = num2str(params.bucket_size);
f_str = num2str(params.forecast_size);
file_surffix = ['_b_', b_str, '_f_', f_str];

if strcmp(params.mode, 'ALEX')
    file_surffix = [file_surffix '_ALEX'];
    h5_save_path.train = [path.data_path, '/train', file_surffix, '.h5'];
    h5_save_path.trainval = [path.data_path, '/trainval', file_surffix, '.h5'];
    h5_save_path.test = [path.data_path, '/test', file_surffix, '.h5'];
    path.h5_save_path = h5_save_path;
    [train, trainval, test] = collect_data_ALEX(path.data_path, params, div_list);
    
    %convert to hdf5 fomat
    train_data_hdf5 = reshape(train.data, params.bucket_size, [], 1, size(train.data,2));
    train_label_hdf5 = train.label;
    trainval_data_hdf5 = reshape(trainval.data,  params.bucket_size, [], 1, size(trainval.data,2));
    trainval_label_hdf5 = trainval.label;
    test_data_hdf5 = reshape(test.data,  params.bucket_size, [], 1, size(test.data,2));
    test_label_hdf5 = test.label;
    
else
    file_surffix = [file_surffix '_MO'];
    h5_save_path.train = [path.data_path, '/train', file_surffix, '.h5'];
    h5_save_path.trainval = [path.data_path, '/trainval', file_surffix, '.h5'];
    h5_save_path.test = [path.data_path, '/test', file_surffix, '.h5'];
    path.h5_save_path = h5_save_path;
    [train, trainval, test] = collect_data_MO(path.data_path, params, div_list);
    
    %convert to hdf5 fomat
    train_data_hdf5 = reshape(train.data, size(train.data,1)/2, 2, 1, size(train.data,2));
    train_label_hdf5 = train.label;
    trainval_data_hdf5 = reshape(trainval.data, size(trainval.data,1)/2, 2, 1, size(trainval.data,2));
    trainval_label_hdf5 = trainval.label;
    test_data_hdf5 = reshape(test.data, size(test.data,1)/2, 2, 1, size(test.data,2));
    test_label_hdf5 = test.label;
end


%print progress
fprintf('Finally collected %d training data (%s) samples from %s\n', size(train_label_hdf5, 2), params.mode, path.data_path);
fprintf('Finally collected %d validation data (%s) samples from %s\n', size(trainval_label_hdf5, 2), params.mode, path.data_path);
fprintf('Finally collected %d testing data (%s) samples from %s\n', size(test_label_hdf5, 2), params.mode, path.data_path);

%write to hdf5
startloc=struct('data',[1,1,1,1], 'label', [1,1]);
if (~isempty(train_data_hdf5))
store2hdf5(path.h5_save_path.train, train_data_hdf5, train_label_hdf5, true, startloc, size(train_label_hdf5,2)); 
fprintf('Generated %s\n', path.h5_save_path.train);
end
if (~isempty(trainval_data_hdf5))
store2hdf5(path.h5_save_path.trainval, trainval_data_hdf5, trainval_label_hdf5, true, startloc, size(trainval_label_hdf5,2)); 
fprintf('Generated %s\n', path.h5_save_path.trainval);
end

if (~isempty(test_data_hdf5))
store2hdf5(path.h5_save_path.test, test_data_hdf5, test_label_hdf5, true, startloc, size(test_label_hdf5,2));
fprintf('Generated %s\n', path.h5_save_path.test);
end

%save the list info
FILE=fopen([path.dev_path ,'/', 'train_list.txt'], 'w');
fprintf(FILE, '%s', path.h5_save_path.train);
fclose(FILE);
fprintf('Generated %s\n', [path.dev_path ,'/', 'train_list.txt']);

FILE=fopen([path.dev_path, '/', 'test_list.txt'], 'w');
fprintf(FILE, '%s', path.h5_save_path.trainval);
fclose(FILE);
fprintf('Generated %s\n', [path.dev_path ,'/', 'test_list.txt']);
end