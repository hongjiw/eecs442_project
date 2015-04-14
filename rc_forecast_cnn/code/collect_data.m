function [train, test] = collect_data(data_path, params, div_list)
data_list = dir(data_path);
train_data_all = []; train_label_all = [];
test_data_all = []; test_label_all = [];

%iterate through all the training sets
for train_ind = 3 : size(data_list, 1)
    if ~data_list(train_ind).isdir
        continue;
    end
    %find the directory
    data_name = data_list(train_ind).name;
    data_dir = [data_path, '/', data_name];

    %print progress
    fprintf('Collecting motion data: %s\n', data_name);
    
    %read the tracker location file
    tracker_loc_file = [data_dir, '/', 'tracker_loc.txt'];
    fid = fopen(tracker_loc_file);
    tracker_loc = textscan(fid, '%d,%d,%d,%d');
    tracker_loc = cell2mat(tracker_loc);
        
    %get the number of frames
    num_frames = size(tracker_loc, 1);
    

    %extract motion features (x, y)
    data = []; label = [];
    
    for frame_ind = params.bucket_size+1 : num_frames - params.rec_size
        data_inst = tracker_loc(frame_ind-params.bucket_size : frame_ind, 1:2);
        %append label
        label_inst = [];
        for rec_ind = 1 : params.rec_size
            label_inst = [label_inst ; (tracker_loc(frame_ind+rec_ind,1:2) - data_inst(1,:))'];
        end
        label = [label label_inst];
        
        %append data
        data_inst = data_inst - repmat(data_inst(1,:), size(data_inst, 1), 1);
        data_inst = data_inst(2:end,:);
        data = [data data_inst];
    end
    
    %print progress
    fprintf('Collected %d data from %s\n', size(label,2), data_name);
    if sum(strcmp(data_name, div_list.train_list), 1) == 1
        train_data_all = [train_data_all data];
        train_label_all = [train_label_all label];
    end
    
    if sum(strcmp(data_name, div_list.test_list), 1) == 1
        test_data_all = [test_data_all data];
        test_label_all = [test_label_all label];
    end
end
train.data = train_data_all;
train.label = train_label_all;
test.data = test_data_all;
test.label = test_label_all;
end