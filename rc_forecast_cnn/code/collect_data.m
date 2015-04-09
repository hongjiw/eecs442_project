function [data_all, label_all, names] = collect_data(data_path, bucket_size)
data_list = dir(data_path);
data_all = []; label_all = [];
names = [];
%iterate through all the training sets
for train_ind = 3 : size(data_list, 1)
    if ~data_list(train_ind).isdir
        continue;
    end
    %find the directory
    data_name = data_list(train_ind).name;
    data_dir = [data_path, '/', data_name];

    %print progress
    fprintf('Collecting data: %s\n', data_name);
    
    %read the tracker location file
    tracker_loc_file = [data_dir, '/', 'tracker_loc.txt'];
    fid = fopen(tracker_loc_file);
    tracker_loc = textscan(fid, '%d,%d,%d,%d');
    tracker_loc = cell2mat(tracker_loc);
        
    %get the number of frames
    num_frames = size(tracker_loc, 1);
    
    data = []; label = [];
    
    %extract motion features (x, y)
    %extract label
    for frame_ind = bucket_size : num_frames - 1
        data_inst = tracker_loc(frame_ind-bucket_size+1 : frame_ind, 1:2);
        
        %append label
        label_inst = (tracker_loc(frame_ind+1,1:2) - data_inst(1,:))';
        label = [label label_inst];
        %append data
        data_inst = data_inst - repmat(data_inst(1,:), size(data_inst, 1), 1);
        data = [data data_inst];
    end
    
    %print progress
    fprintf('Collected %d data from %s\n', size(label,2), data_name);
    data_all = [data_all data];
    label_all = [label_all label];
    
    %return name and seg info
    seg_inst.seg = size(label,2);
    seg_inst.name = data_name;
    names = [names {seg_inst}];
end
end