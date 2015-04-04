function coord = bbox2coord(bbox)
    coord00 = [bbox(1), bbox(2)];
    coord01 = [bbox(1)+bbox(3), bbox(2)];
    coord10 = [bbox(1), bbox(2)+bbox(4)];
    coord11 = [bbox(1)+bbox(3), bbox(2)+bbox(4)];
    coord = [coord00;coord01;coord10;coord11];
end