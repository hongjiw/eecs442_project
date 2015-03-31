clc;
clear;
%add SVM path
addpath('/home/hongjiw/research/activity_forecasting/liblinear-1.96/matlab');

%add OF (optical flow) path
addpath('/home/hongjiw/research/activity_forecasting/eccv2004Matlab');

%setup VLfeat
VLfeat_path = '/home/hongjiw/research/vlfeat-0.9.20';
run([VLfeat_path '/toolbox/vl_setup'])
vl_version verbose
vl_setup demo
%vl_demo_sift_basic

%% train
train_path = '/home/hongjiw/research/activity_forecasting/demo/data/train';
train_list = dir(train_path);
% memoery of the regressor
bucket_size = 5; % 5 training

%iterate through all the training sets
for train_ind = 3 : size(train_list, 1)
    %get training set
    train_ins_dir = [train_path, '/', train_list(train_ind).name];
    img_dir = [train_ins_dir, '/', 'imgs'];
    
    %get label
    label_file = [train_ins_dir, '/', 'label.txt'];
    label_id = fopen(label_file);
    labels = textscan(label_id, '%d,%d,%d,%d');
    labels = cell2mat(labels);
    
    num_imgs = size(labels, 1);

    %empty the bucket
    bucket = [];
    train_data = [];
    train_label = [];
    
    %collect training data
    for img_ind = 2 : 7%(num_imgs-1)
        %load the window
        img_ind_str_cur = sprintf('%05d', img_ind);
        img_name_cur = sprintf('img%s.png', img_ind_str_cur);
        img_cur = imread([img_dir, '/', img_name_cur]);
        
        img_ind_str_prev = sprintf('%05d', img_ind-1);
        img_name_prev = sprintf('img%s.png', img_ind_str_prev);
        img_prev = double(imread([img_dir, '/', img_name_prev]));
        
        img_ind_str_next = sprintf('%05d', img_ind+1);
        img_name_next = sprintf('img%s.png', img_ind_str_next);
        img_next = double(imread([img_dir, '/', img_name_next]));
        
        label_cur = double(labels(img_ind, :));
        label_prev = double(labels(img_ind-1, :));
        
        %compute OF
        OF = mex_OF(img_prev, img_next);
        OF = single(OF);
        
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
        [f_v, d_v] = vl_sift(OF(:,:,1),'frames',fc);
        [f_h, d_h] = vl_sift(OF(:,:,2),'frames',fc);
        
        %fill the bucket
        if size(bucket, 2) < bucket_size*2
            bucket = [bucket, [d_v d_h]];
        else
            %add training sample
            train_data = [train_data; bucket(:)'];
            train_label = [train_label; get_direction(label_cur, label_prev)];
            bucket = [bucket(:,3:end), [d_v d_h]];
        end
    end
end

%train the liblinear
model = train(train_label, sparse(double(train_data)), '-c 1');











