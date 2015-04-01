function [data_all, label_all] = collect_data(data_path, bucket_size)
data_list = dir(data_path);
%iterate through all the training sets

data_all = [];
label_all = [];

for train_ind = 3 : size(data_list, 1)
    data_name = data_list(train_ind).name;
    data_dir = [data_path, '/', data_name];
    data_file = [data_dir, '/', data_name, '_', num2str(bucket_size), '.mat'];
    
    %initilization
    data = []; label = [];
    
    %if exists, just load the mat file
    if exist(data_file, 'file')
        data_inst = load(data_file);
        data_inst = data_inst.data_inst;
        data = [data; data_inst.data];
        label = [label; data_inst.label];
    else
        %if it does not exists, then extract the feature and save
        img_dir = [data_dir, '/', 'imgs'];
        %get label
        tracker_loc_file = [data_dir, '/', 'tracker_loc.txt'];
        fid = fopen(tracker_loc_file);
        bbox_tracker = textscan(fid, '%d,%d,%d,%d');
        bbox_tracker = cell2mat(bbox_tracker);
        num_imgs = size(bbox_tracker, 1);
        
        %empty the bucket
        bucket = [];
        
        %collect training data
        for img_ind = 2 : (num_imgs-1)
            %load the window
            img_ind_str_cur = sprintf('%05d', img_ind);
            img_name_cur = sprintf('img%s.png', img_ind_str_cur);
            img_cur = double(imread([img_dir, '/', img_name_cur]));
            
            img_ind_str_prev = sprintf('%05d', img_ind-1);
            img_name_prev = sprintf('img%s.png', img_ind_str_prev);
            img_prev = double(imread([img_dir, '/', img_name_prev]));
            
            img_ind_str_next = sprintf('%05d', img_ind+1);
            img_name_next = sprintf('img%s.png', img_ind_str_next);
            img_next = double(imread([img_dir, '/', img_name_next]));
            
            label_cur = double(bbox_tracker(img_ind, :));
            label_next = double(bbox_tracker(img_ind+1, :));
            
            %compute OF
            OF = mex_OF(img_prev, img_next);
            OF = single(OF);
            
            %center to computer SIFT (too huristic, change later)
            fc = [label_cur(1) + label_cur(3) / 2; label_cur(2) + label_cur(4) / 2; ...
                (label_cur(3)+label_cur(4)) / 20; 0];
            
            %test SIFT
            %{
                [f, d] = vl_sift(single(rgb2gray(img_cur)), 'frames', fc);
                imshow(img_cur);
                h3 = vl_plotsiftdescriptor(d,f);
                set(h3,'color','g');
            %}
            
            %compute the SIFT of OF (vertical and horizontal)
            [~, d_v] = vl_sift(OF(:,:,1),'frames',fc);
            [~, d_h] = vl_sift(OF(:,:,2),'frames',fc);
            
            %visualize
            %{
            imshow(OF(:,:,1));
            h3 = vl_plotsiftdescriptor(d_v,f_v);
            set(h3,'color','g');
             %}
            
            %fill the bucket
            if (size(bucket, 2) + 2) < bucket_size*2
                bucket = [bucket, [d_v d_h]];
            else
                %update bucket
                bucket = [bucket(:, end-(bucket_size-1)*2 + 1:end), [d_v d_h]];
                %append data and label
                data = [data; bucket(:)'];
                label = [label; get_direction(label_next, label_cur)];
            end
            
            %print process
            fprintf('Feature collection: %d/%d\n', img_ind, num_imgs);
        end
        
        %save train data
        data_inst.data = data;
        data_inst.label = label;
        save(data_file, 'data_inst');
    end
    data_all = [data_all; data];
    label_all = [label_all; label];
end
end
