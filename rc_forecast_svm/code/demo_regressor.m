function demo_regressor(data_path, range, model, bucket_size)
%This function performs SVM prediction and outputs a pred.txt in the proper
%directory

%get the demo list
data_list = dir(data_path);

%iterate through all the demo sets
for demo_ind = 3 : size(data_list, 1)
    %load feature
    data_name = data_list(demo_ind).name;
    data_dir = [data_path, '/', data_name];
    data_file = [data_dir, '/', data_name, '_', num2str(bucket_size), '.mat'];
    data_inst = load(data_file);
    data_inst = data_inst.data_inst;
    data = data_inst.data;
    label = data_inst.label;
    
    bbox = [];
    
    %load tracker location
    label_file = [data_dir, '/', 'tracker_loc.txt'];
    fid = fopen(label_file);
    bbox_tracker = textscan(fid, '%d,%d,%d,%d');
    bbox_tracker = cell2mat(bbox_tracker);
    num_imgs = size(bbox_tracker, 1);
    
    for img_ind = 1 : num_imgs
        if img_ind > bucket_size && img_ind < num_imgs
            bbox_pred = [];
            bbox_tr = bbox_tracker(img_ind,:);
            
            for range_ind = 1 : range
                %predict next frame
                [predicted_label, ~, ~] =...
                    predict(label(img_ind-bucket_size), sparse(double(data(img_ind-bucket_size, :))), model);
                
                %get the bbox next frame (change later)
                vec = 10*direction2vec(predicted_label);
                bbox_next(1) = bbox_tr(1) + vec(1);
                bbox_next(2) = bbox_tr(2) + vec(2); %note y is opposite in this case
                bbox_pred = [bbox_pred; [bbox_next bbox_tr(3:4)]];
                
                %update bucket_pred if range > 1
            end
            bbox = [bbox; bbox_pred];
        else
            bbox = [bbox; [-1, -1, -1, -1]];
        end
    end
    
    %save the bounding box to file
    bbox = int32(bbox);
    bbox_file = [data_dir, '/', 'pred_loc.txt'];
    fid = fopen(bbox_file', 'w'); % Open for writing
    for box_ind =1:size(bbox,1)
        fprintf(fid, '%d,', bbox(box_ind, :));
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    %print progress
    fprintf('Save trajectory prediction to: %s\n', bbox_file);
end
end


