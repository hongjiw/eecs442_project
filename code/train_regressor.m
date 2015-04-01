function model = train_regressor(train_path, bucket_size, liblinear_options, train_data_file)
    train_list = dir(train_path);
    %iterate through all the training sets
    if exist(train_data_file, 'file')
        data = load(train_data_file);
        data = data.data;
        train_data = data.train_data;
        train_label = data.train_label;
    else
        train_data = [];
        train_label = [];

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

            %collect training data
            for img_ind = 2 : (num_imgs-1)
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

                %center to computer SIFT
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
                %imshow(OF(:,:,1));
                %h3 = vl_plotsiftdescriptor(d_v,f_v);
                %set(h3,'color','g');

                %fill the bucket
                if size(bucket, 2) < bucket_size*2
                    bucket = [bucket, [d_v d_h]];
                else
                    %add training sample
                    train_data = [train_data; bucket(:)'];
                    train_label = [train_label; get_direction(label_cur, label_prev)];
                    bucket = [bucket(:,3:end), [d_v d_h]];
                end

                %print process
                fprintf('Feature collection: %d/%d\n', img_ind, num_imgs);
            end
        end
        
        %save train data
        data.train_data = train_data;
        data.train_label = train_label;
        save(train_data_file, 'data');
    end
    
    
   
end