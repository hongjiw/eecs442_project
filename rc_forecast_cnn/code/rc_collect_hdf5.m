function rc_collect_hdf5(bucket_size, data_path, dev_path, test_num, seg_file_path)
[data, label, seg] = collect_data(data_path, bucket_size);
fprintf('Finally collected %d data samples from %s\n', size(label,2), data_path);

%convert to hdf5 fomat
data_hdf5 = reshape(data, size(data,1), 2, 1, size(data,2) / 2);
label_hdf5 = label;
assert(size(seg,2) > test_num)
train_size = 0;
for seg_ind = 1 : length(seg) - test_num
    train_size = train_size + seg(seg_ind).seg;
end
train_data_hdf5 = data_hdf5(:,:,:,1:train_size);
train_label_hdf5 = label_hdf5(:,1:train_size);
test_data_hdf5 = data_hdf5(:,:,:,train_size+1:end);
test_label_hdf5 = label_hdf5(:,train_size+1:end);

%write to hdf5
hdf5_train_file = [data_path, '/train.h5'];
hdf5_test_file = [data_path, '/test.h5'];
startloc=struct('data',[1,1,1,1], 'label', [1,1]);
store2hdf5(hdf5_train_file, train_data_hdf5, train_label_hdf5, true, startloc, size(train_label_hdf5,2)); 
store2hdf5(hdf5_test_file, test_data_hdf5, test_label_hdf5, true, startloc, size(test_label_hdf5,2)); 

%save the list info
FILE=fopen([dev_path ,'/', 'train_list.txt'], 'w');
fprintf(FILE, '%s', hdf5_train_file);
fclose(FILE);
FILE=fopen([dev_path, '/', 'test_list.txt'], 'w');
fprintf(FILE, '%s', hdf5_test_file);
fclose(FILE);

%save the seg info
test_seg = seg(end-test_num+1:end);
save(seg_file_path, 'test_seg');

%wrap up
fprintf('Generated %s\n', [dev_path ,'/', 'train_list.txt']);
fprintf('Generated %s\n', [dev_path ,'/', 'test_list.txt']);
fprintf('Generated %s\n', hdf5_train_file);
fprintf('Generated %s\n', hdf5_test_file);
fprintf('Generated %s\n', seg_file_path);
end