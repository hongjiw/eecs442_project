function [train, trainval, test] = collect_data_ALEX(data_path, params, div_list)
data_list = dir(data_path);
train_data_all = []; train_label_all = [];
test_data_all = []; test_label_all = [];
trainval_data_all = []; trainval_label_all = [];

%iterate through all the training sets
for train_ind = 3 : size(data_list, 1)
    if ~data_list(train_ind).isdir
        continue;
    end
    %find the directory
    data_name = data_list(train_ind).name;
    data_dir = [data_path, '/', data_name];

    %read the tracker location file
    tracker_loc_file = [data_dir, '/tracker_loc.txt'];
    fid = fopen(tracker_loc_file);
    tracker_loc = textscan(fid, '%d,%d,%d,%d');
    tracker_loc = cell2mat(tracker_loc);
        
    %get the number of frames
    num_frames = size(tracker_loc, 1);

    %extract motion features (x, y)
    data = []; label = [];
    
    %load alex's mat feature
    alex_feature = load([data_dir, '/alex_feature.mat']);
    alex_feature = alex_feature.alex_feature;
    
    %make sure alex's feature is in good shape
    assert(size(alex_feature, 1) == size(tracker_loc, 1))
    
    %collect the feature under the current directory
    for frame_ind = params.bucket_size+1 : num_frames - params.forecast_size
        data_inst = tracker_loc(frame_ind-params.bucket_size : frame_ind, 1:2);
        normalizer =  double(tracker_loc(frame_ind-params.bucket_size, 3:4));
        
        %append label
        label_inst = [];
        for forecast_ind = 1 : params.forecast_size
            label_tmp = double(tracker_loc(frame_ind+forecast_ind,1:2) - data_inst(1,:))';
            %normalize
            label_tmp = label_tmp ./ normalizer';
            label_inst = [label_inst ; label_tmp];
        end
        label = [label label_inst];
        
        %append data
        data_inst = data_inst - repmat(data_inst(1,:), size(data_inst, 1), 1);
        data_inst = data_inst(2:end,:);
        %normalize
        data_inst = double(data_inst) ./ repmat(normalizer, size(data_inst, 1), 1);
        
        %add alex's feature (we try using alex's feature first without
        %appending the motion feature)
        alex_inst = alex_feature(frame_ind-params.bucket_size+1 : frame_ind, :);
        data_inst = [alex_inst data_inst];
      
        %append to the total data
        data = [data data_inst(:)];
    end
   
    %print progress
    fprintf('Collected %d %s data from %s\n', size(label,2), params.mode, data_name);
   if sum(strcmp(data_name, div_list.train_list), 1) == 1
        train_data_all = [train_data_all data];
        train_label_all = [train_label_all label];
    elseif sum(strcmp(data_name, div_list.test_list), 1) == 1
        test_data_all = [test_data_all data];
        test_label_all = [test_label_all label];
    elseif sum(strcmp(data_name, div_list.trainval_list), 1) == 1
        trainval_data_all = [trainval_data_all data];
        trainval_label_all = [trainval_label_all label];
    else
        assert(0);
    end
end
train.data = train_data_all;
train.label = train_label_all;
trainval.data = trainval_data_all;
trainval.label = trainval_label_all;
test.data = test_data_all;
test.label = test_label_all;
end