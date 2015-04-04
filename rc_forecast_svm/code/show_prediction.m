function show_prediction(data_path, demo_ind, bucket_size)
%get the demo list
data_list = dir(data_path);

%iterate through all the demo sets
for data_ind = 3 : size(data_list, 1)
    %get demo set
    data_dir = [data_path, '/', data_list(data_ind).name];
    img_dir = [data_dir, '/', 'imgs'];
    
    %get gt label
    tracker_loc_file = [data_dir, '/', 'tracker_loc.txt'];
    fid = fopen(tracker_loc_file);
    labels = textscan(fid, '%d,%d,%d,%d');
    labels = cell2mat(labels);
    
    %get demo label
    pred_loc_file = [data_dir, '/', 'pred_loc.txt'];
    fid = fopen(pred_loc_file);
    pred_labels = textscan(fid, '%d,%d,%d,%d,');
    pred_labels = cell2mat(pred_labels);
    
    %get number of images
    num_imgs = size(labels, 1);
    
    %get first frame num
    img_list = dir(img_dir);
    img_00 = img_list(3).name;
    ind_offset = str2double(img_00(4:8)) - 1;
    
    if sum((data_ind-2) == demo_ind) > 0
        %print progress
        fprintf('Demo %s\n', data_dir);
        
        %show trajectory
        show_trajectory(imread([img_dir, '/', img_00]), tracker_loc_file, pred_loc_file, bucket_size);
        
        %print progress
        fprintf('Hit any key to show the frames\n');
        pause;
        
        %show frames
        for img_ind = 1 : num_imgs
            %load the window
            img_ind_str = sprintf('%05d', img_ind+ind_offset);
            img_name = sprintf('img%s.png', img_ind_str);
            img = imread([img_dir, '/', img_name]);

            %get labels
            bbox_tracker = labels(img_ind,:);
            bbox_pred = pred_labels(img_ind,:);

            %show the image
            figure(1); imshow(img, []); 

            %plot prediction on previous image as green
            if bbox_pred(1) ~= -1
                rectangle('position',bbox_pred, 'EdgeColor', 'g');
            end
            %plot tracker on previous image as red
            rectangle('position',bbox_tracker, 'EdgeColor', 'r', 'LineWidth', 1);
            pause(0.1);
        end
        
        %go to the next demo data
        fprintf('Hit any key to continue to next demo set\n');
        pause;
    end 
end
end