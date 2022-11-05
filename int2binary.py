# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.

class Int2BinaryTransformer:
    def __init__(self, data_space):
        self.data_space = data_space
        self.count = [self.int_bin_count(data) for data in data_space]

    def transfer(self, input_point):
        num = input_point
        int2bin = ['{:032b}'.format(num[i])[32 - self.count[i]:] for i in range(len(num))]

        bin_list = [[] for i in range(len(num))]
        for i, string in enumerate(int2bin):
            for j in string:
                bin_list[i].append(int(j))
        return bin_list

    def int_bin_count(self, integer):
        int2bin = '{:032b}'.format(integer)
        count = 0
        for ch in int2bin:
            if (int(ch) != 0):
                break
            else:
                count += 1
        count = 32 - count
        return count
