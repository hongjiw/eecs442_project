function [train, trainval, test] = collect_data_OF(data_path, params, div_list)
data_list = dir(data_path);
train_data_all = []; train_label_all = [];
test_data_all = []; test_label_all = [];
trainval_data_all = []; trainval_label_all = [];

for train_ind = 3 : size(data_list, 1)
    if ~data_list(train_ind).isdir
        continue;
    end
    %find the directory
    data_name = data_list(train_ind).name;
    data_dir = [data_path, '/', data_name];
    
    %if it does not exists, then extract the feature and save
    img_dir = [data_dir, '/', 'imgs'];
    
    %get label
    tracker_loc_file = [data_dir, '/', 'tracker_loc.txt'];
    fid = fopen(tracker_loc_file);
    tracker_loc = textscan(fid, '%d,%d,%d,%d');
    tracker_loc = cell2mat(tracker_loc);
    num_imgs = size(dir(img_dir), 1) - 2;
    %sanity check
    assert(num_imgs == size(tracker_loc, 1));
    %empty the bucket
    bucket = [];
    %get first frame num
    img_list = dir(img_dir);
    img_00 = img_list(3).name;
    ind_offset = str2double(img_00(4:8)) - 1;
    
    %get OF feature
    OF_all = extract_OF(img_dir, [data_dir, '/OF.mat']);
    
    %go through all images and extract optical flow
    data = []; label = [];

    for frame_ind = 2 : num_imgs-params.forecast_size
        
        %load the window
        img_ind_str_cur = sprintf('%05d', frame_ind+ind_offset);
        img_name_cur = sprintf('img%s.png', img_ind_str_cur);
        img_cur = imread([img_dir, '/', img_name_cur]);
        
        label_cur = double(tracker_loc(frame_ind, :));
        
        
        %center to computer SIFT (too huristic, change later)
        fc = [label_cur(1) + label_cur(3) / 2; label_cur(2) + label_cur(4) / 2; ...
            (label_cur(3)+label_cur(4)) / 20; 0];
        
        %test SIFT
        %{
        [f, d] = vl_sift(single(rgb2gray(im2double(img_cur))), 'frames', fc);
        imshow(img_cur);
        h3 = vl_plotsiftdescriptor(d,f);
        set(h3,'color','g');
        %}
        
        %compute the SIFT of OF (vertical and horizontal)
        OF = OF_all(:,:,:,frame_ind-1);
        OF = single(OF);
        [f_v, d_v] = vl_sift(OF(:,:,1),'frames',fc);
        [f_h, d_h] = vl_sift(OF(:,:,2),'frames',fc);
        
        %visualize
        %{
        imshow(OF(:,:,2));
        h3 = vl_plotsiftdescriptor(d_h,f_h);
        set(h3,'color','g');
        %}
        
        %fill the bucket
        if (size(bucket, 2) + 2) < params.bucket_size*2
            bucket = [bucket, [d_v d_h]];
        else
            %update bucket
            bucket = [bucket(:, end-(params.bucket_size-1)*2 + 1:end), [d_v d_h]];
            
            %append data
            data = [data bucket(:)];
            
            %append label
            label_inst = [];
            for rec_ind = 1 : params.forecast_size
                label_inst = [label_inst; (tracker_loc(frame_ind+rec_ind,1:2) -  ...
                    tracker_loc(frame_ind-params.bucket_size, 1:2))'];
            end
            label = [label label_inst];
        end
        %print process
        fprintf('%d/%d\n', frame_ind-1, num_imgs-2);
    end
    
    %print progress
    fprintf('Collected %d OF data from %s\n', size(label,2), data_name);
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
