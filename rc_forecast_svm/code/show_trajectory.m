function show_trajectory(im, actual_path, predict_path, bucket_size)
%load the data
actual_data = load(actual_path);
predict_data = load(predict_path);

%get rid of no predictions
actual_data = actual_data(bucket_size+1:end-1,:);
predict_data = predict_data(bucket_size+1:end-1,:);

%calculate the average distance
 sum = calculate_distance(actual_data, predict_data);
%show the image
str = ['Average error is ', num2str(sum)];
position = [10, 0];
%im = insertText(im,position,text,'FontSize',18,'BoxColor','yellow', 'BoxOpacity',0.9);
imshow(im);
hold on;
h = text(position(1), position(2)+20, str, 'Color', 'yellow', 'LineWidth', 10, 'FontWeight', 'bold');
h.FontSize = 12;
%plot two lines
for i = 2 : size(actual_data, 1)
    x1 = actual_data(i - 1, 1) + actual_data(i - 1, 3) / 2;
    x2 = actual_data(i, 1) + actual_data(i, 3) / 2;
    x = [x1, x2];
    y1 = actual_data(i - 1, 2) + actual_data(i - 1, 4) / 2;
    y2 = actual_data(i, 2) + actual_data(i, 4) / 2;
    y = [y1, y2];
    plot(x, y, 'r','linewidth', 2);
end
for i = 2 : size(predict_data, 1)
    x1 = predict_data(i - 1, 1) + predict_data(i - 1, 3) / 2;
    x2 = predict_data(i, 1) + predict_data(i, 3) / 2;
    x = [x1, x2];
    y1 = predict_data(i - 1, 2) + predict_data(i - 1, 4) / 2;
    y2 = predict_data(i, 2) + predict_data(i, 4) / 2;
    y = [y1, y2];
    plot(x, y, 'g','linewidth', 2);
end

end