function OF_all = extract_OF (img_dir, save_OF_to)
%get first frame num
img_list = dir(img_dir);
img_00 = img_list(3).name;
ind_offset = str2double(img_00(4:8)) - 1;
num_imgs = size(dir(img_dir), 1) - 2;

img_ind_str_inst = sprintf('%05d', ind_offset+1);
img_name_inst= sprintf('img%s.png', img_ind_str_inst);
img_inst = double(imread([img_dir, '/', img_name_inst]));

OF_all = zeros(size(img_inst, 1), size(img_inst, 2), 2, num_imgs-2);

if ~exist(save_OF_to, 'file')
    fprintf('Extracting OF feature from: %s\n', img_dir);
    for frame_ind = 2 : num_imgs - 1
        %load the window
        img_ind_str_prev = sprintf('%05d', frame_ind-1+ind_offset);
        img_name_prev = sprintf('img%s.png', img_ind_str_prev);
        img_prev = double(imread([img_dir, '/', img_name_prev]));

        img_ind_str_next = sprintf('%05d', frame_ind+1+ind_offset);
        img_name_next = sprintf('img%s.png', img_ind_str_next);
        img_next = double(imread([img_dir, '/', img_name_next]));

        %compute OF
        OF = mex_OF(img_prev, img_next);    
        OF_all(:,:,:,frame_ind-1) =  OF;
        
        %print process
        fprintf('%d/%d\n', frame_ind-1, num_imgs-2);
    end
    save(save_OF_to, 'OF_all', '-v7.3');
else
    fprintf('Read OF feature from: %s\n', img_dir);
    OF_all = load(save_OF_to);
    OF_all = OF_all.OF_all;
end