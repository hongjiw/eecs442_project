function show_prediction(frame, ground_truth, prediction_result)
%find how many images
temp_frame = [frame, '/*.png'];
list_of_imgs = dir(temp_frame);
number_of_imgs = numel(list_of_imgs);
%load the ground truth
gt_data = importdata(ground_truth);
red = uint8([255 0 0]);
shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',red);
%load the prediction result
temp_frame2 = [prediction_result, '/*.txt'];
list_of_preds = dir(temp_frame2);
number_of_preds = numel(list_of_preds);
for i = 1 : number_of_preds
    pred_name = [prediction_result list_of_preds(i).name];
    predict_data(i,:,:) = importdata(pred_name);
end
yellow = uint8([255 255 0]);
shapeInserter2 = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',yellow);
for i = 1 : number_of_imgs
    img_name = [frame '/' list_of_imgs(i).name];
    current_im = imread(img_name);
    rectangle = int32(gt_data(i, :));
    current_im = step(shapeInserter, current_im, rectangle);
    for j = 1 : number_of_preds
         rectangle = int32(reshape(predict_data(j,i,:), 1, 4));
         current_im = step(shapeInserter2, current_im, rectangle); 
    end
    imshow(current_im);
end