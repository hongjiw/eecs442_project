function show_prediction(data_path, test_seg, bucket_size)
%get the demo list
data_list = dir(data_path);

%load predicted location
%get demo label
pred_loc_file = [data_path, '/', 'pred_loc.txt'];
pred_labels_test_all = load(pred_loc_file);
recurrent_size = size(pred_labels_test_all, 2) / 2;

%iterate through all the demo sets
test_set_ind = 1;
pred_index = 1;

for data_ind = 3 : size(data_list, 1)
    if(~data_list(data_ind).isdir || ~strcmp(data_list(data_ind).name, test_seg(test_set_ind).name))
        continue;
    end
    
    %get predicted label
    pred_labels = pred_labels_test_all(pred_index:pred_index+test_seg(test_set_ind).seg-1, :);
    pred_index = pred_index + test_seg(test_set_ind).seg;
    test_set_ind = test_set_ind + 1;

    %get demo set
    data_dir = [data_path, '/', data_list(data_ind).name];
    img_dir = [data_dir, '/', 'imgs'];
    
    %get gt label
    tracker_loc_file = [data_dir, '/', 'tracker_loc.txt'];
    fid = fopen(tracker_loc_file);
    labels = textscan(fid, '%d,%d,%d,%d');
    labels = cell2mat(labels);
    
    %get number of images
    num_imgs = size(labels, 1);
    
    %get first frame num
    img_list = dir(img_dir);
    img_00 = img_list(3).name;
    ind_offset = str2double(img_00(4:8)) - 1;
    %print progress
    fprintf('Demo %s\n', data_dir);
    
    %show frames
    for img_ind = 1 : num_imgs-1
        %load the window
        img_ind_str = sprintf('%05d', img_ind+ind_offset);
        img_name = sprintf('img%s.png', img_ind_str);
        img = imread([img_dir, '/', img_name]);
        
        %get tracker loc
        tracker_loc_inst = labels(img_ind,:);
      
        %show the image
        figure(1); imshow(img, []);
        
        %plot prediction on previous image as green
        if img_ind >= bucket_size
            for recur_ind = 1 : recurrent_size
                bbox_pred = int32([pred_labels(img_ind-bucket_size+1, 2*recur_ind-1:2*recur_ind) 0 0])...
                    + labels(img_ind-bucket_size+recur_ind,:);
                rectangle('position',bbox_pred, 'EdgeColor', 'g');
            end
        end
        %plot tracker on previous image as red
        rectangle('position',tracker_loc_inst, 'EdgeColor', 'r', 'LineWidth', 1);
        pause(0.1);
    end
    
    %go to the next demo data
    fprintf('Hit any key to continue to next demo set\n');
    pause;
end
end