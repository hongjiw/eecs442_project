function show_prediction(data_path, params, test_list, demo_list, pos)
%get the demo list
data_list = dir(data_path);
while (1)
%load predicted location
%get demo label
pred_loc_file = [data_path, '/', 'pred_loc.txt'];
pred_labels_test_all = load(pred_loc_file);
assert(size(pred_labels_test_all, 2) / 2 == params.rec_size);

%iterate through all the demo sets
test_set_ind = 1;
pred_index = 1;

for data_ind = 3 : size(data_list, 1) 
    if(~data_list(data_ind).isdir || ~sum(strcmp(data_list(data_ind).name, test_list), 1) == 1)
        continue;
    end
    
    %get demo set
    data_dir = [data_path, '/', data_list(data_ind).name];
    img_dir = [data_dir, '/', 'imgs'];
    
    %get predicted location
    seg_offset = size(dir(img_dir), 1) - params.bucket_size - 2 - params.rec_size; %two is . and ..
    pred_loc = pred_labels_test_all(pred_index:pred_index+seg_offset-1, :);
    pred_index = pred_index +seg_offset;
    test_set_ind = test_set_ind + 1;
    %get tracker location
    tracker_loc_file = [data_dir, '/', 'tracker_loc.txt'];
    fid = fopen(tracker_loc_file);
    tracker_loc = textscan(fid, '%d,%d,%d,%d');
    tracker_loc = cell2mat(tracker_loc);
    
    %get first frame num
    img_list = dir(img_dir);
    img_00 = img_list(3).name;
    ind_offset = str2double(img_00(4:8)) - 1;
    %print progress
    fprintf('Demo %s\n', data_dir);
    
    %get number of images
    num_imgs = size(img_list, 1) - 2;
    assert(num_imgs == size(tracker_loc, 1));
    close all;
    fh = figure('Position', pos); %left bottom width height
    %show frames
    
    % label DEMO
    if (~sum(strcmp(data_list(data_ind).name, demo_list), 1) == 1)
        continue
    end
    for img_ind = 1 : num_imgs-params.rec_size
        %load the window
        img_ind_str = sprintf('%05d', img_ind+ind_offset);
        img_name = sprintf('img%s.png', img_ind_str);
        img = imread([img_dir, '/', img_name]);
        
        %show the image
        figure(fh); 
        imshow(img);
        
        %plot prediction on previous image as green
    %{
        if img_ind >= params.bucket_size+1
            for recur_ind = 1 : params.rec_size
                bbox_pred = int32([pred_loc(img_ind-params.bucket_size, 2*recur_ind-1:2*recur_ind) 0 0])...
                    + tracker_loc(img_ind-params.bucket_size,:);
                rectangle('position',bbox_pred, 'EdgeColor', 'g');
            end
        end
%}
        m = 10;
        assert(m <= params.rec_size);
        if img_ind >= params.bucket_size+1 + m
            %show next 5th frame
            bbox_pred = int32([pred_loc(img_ind-params.bucket_size, 2*m-1:2*m) 0 0])...
                + tracker_loc(img_ind-params.bucket_size,:);
            rectangle('position',bbox_pred, 'EdgeColor', 'g');
            %show prev 5th pred
            bbox_pred = int32([pred_loc(img_ind-m-params.bucket_size, 2*m-1:2*m) 0 0])...
                + tracker_loc(img_ind-m-params.bucket_size,:);
            rectangle('position',bbox_pred, 'EdgeColor', 'y');
        end
        
        %plot tracker on previous image as red
        tracker_loc_inst = tracker_loc(img_ind,:);
        rectangle('position',tracker_loc_inst, 'EdgeColor', 'r', 'LineWidth', 1);
        
        if img_ind >= params.bucket_size+1 + m
            pause(0.01);
        else
            pause(0.01);
        end
    end
    
    %go to the next demo data
    %fprintf('Hit any key to continue to next demo set\n');
    %pause;
end
end
end