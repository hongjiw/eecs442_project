function direction = get_direction(label_cur, label_prev)
    direction = 1;
    vec = [label_cur(1) - label_prev(1), label_cur(2) - label_prev(2)];
    if vec(1) == 0 && vec(2) == 0
        direction = 0;
    elseif vec(1) == 0 && vec(2) < 0
        direction = 5;
    elseif vec(1) == 0 && vec(2) > 0
         direction = 1;
    elseif vec(1) > 0 && vec(2) == 0
        direction = 3;
    elseif vec(1) < 0 && vec(2) == 0
        direction = 7;
    else
        dir = atan2(vec(2), vec(1)) * 180 / pi;
        if abs(dir) <= 22.5
            direction = 3;
        elseif abs(dir) > 22.5 &&  abs(dir) <= 67.5
            if dir > 0
                direction = 2;
            else
                direction = 4;
            end
        elseif abs(dir) > 67.5 &&  abs(dir) <= 112.5
            if dir > 0
                direction = 1;
            else
                direction = 5;
            end
        elseif abs(dir) > 112.5 &&  abs(dir) <= 157.5
             if dir > 0
                direction = 8;
            else
                direction = 6;
            end
        else
            direction = 7;
        end
    end
end