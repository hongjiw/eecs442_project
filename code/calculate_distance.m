function sum = calculate_distance(ground_truth, tracker)
sum = 0;
total_distance = 0;
for i = 1 : size(ground_truth, 1)
    sum = sum + sqrt((ground_truth(i, 1) - tracker(i, 1))^2 + (ground_truth(i, 2) - tracker(i, 2))^2);
end
for i = 1 : size(ground_truth, 1) - 1
    total_distance = total_distance + sqrt((ground_truth(i + 1, 1) - ground_truth(i, 1))^2 + (ground_truth(i + 1, 2) - ground_truth(i, 2))^2);
end
sum = sum / total_distance;
end