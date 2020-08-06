#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
for line in matrix:
    a = []
    a.append(line[2])
    a.append(line[3])
    the_middle.append(a)
print("The middle columns of the matrix are: {}".format(the_middle))
