function vec = direction2vec(label)
    if label == 0
        vec = [0 0];
    elseif label == 1
        vec = [0 1];
    elseif label == 2
        vec = [1 1];
    elseif label == 3
        vec = [1 0];
    elseif label == 4
        vec = [1 -1];
    elseif label == 5
        vec = [0 -1];
    elseif label == 6
        vec = [-1 -1];
    elseif label == 7
        vec = [-1 0];
    elseif label == 8
        vec = [-1 1];
    end
end