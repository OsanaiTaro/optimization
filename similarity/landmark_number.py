with open('1129_minmap_1.txt') as f:
    file = open('landmark_number_1129_minmap_11129_minmap_1p.txt', 'w')
    for line in f:
        line = eval(line.rstrip())
        file.write(str(sum(line)) + "\n")
    file.close()