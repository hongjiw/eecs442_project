function direction = get_direction(label_cur, label_prev)
    direction = 1;
    direction_vec = [label_cur(1) - label_prev(1), label_cur(2) - label_prev(2)];
    
    %direction bucket (sin)
    direction_bucket = [0 ];
end