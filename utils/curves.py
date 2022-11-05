# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.

from pyzorder import ZOrderIndexer




'''
Z order Module supporting z-order curve value output
'''

class ZorderModule:
    def __init__(self, data_space):
        self.indexer = ZOrderIndexer((0, data_space[0]), (0, data_space[1]))

    def output(self, datapoint):
        return self.indexer.zindex(datapoint[0], datapoint[1])

class Z_Curve_Module:
    def __init__(self, transfer):
        '''here store some orders follow restrict rule, jiangneng'''
        self.possible_orders = []


        self.transfer = transfer
        self.bit_length = transfer.count
        # print(self.bit_length)

        self.order = []

        for i in range(20):
            self.order.append('x')
            self.order.append('y')


    def output(self, datapoint):
        binary_form = self.transfer.transfer(datapoint)
        # print(binary_form)
        x_num = 0
        y_num = 0
        value = 0
        # print(binary_form)

        for i in range(len(self.order)):
            if self.order[i] == 'x':
                value = value * 2 + int(binary_form[0][x_num])
                x_num += 1
            else:
                value = value * 2 + int(binary_form[1][y_num])
                y_num += 1
        # print(value)
        return




def recur_enumerate(num_x, num_y):
    if num_x == 0 and num_y == 0:
        return [[]]
    orders = []
    if num_x > 0:
        order = ['x']
        sub_orders = recur_enumerate(num_x - 1, num_y)
        for sub_order in sub_orders:
            orders.append(order + sub_order)
    if num_y > 0:
        order = ['y']
        sub_orders = recur_enumerate(num_x, num_y - 1)
        for sub_order in sub_orders:
            orders.append(order + sub_order)

    return orders


class DesignQuiltsModule:
    def __init__(self, transfer):
        '''here store some orders follow restrict rule, jiangneng'''
        self.possible_orders = []

        self.transfer = transfer
        self.bit_length = transfer.count
        # print(self.bit_length)

        self.order = []

        sub_orders = recur_enumerate(2, 4)

        sub_order_2 = []
        for i in range(4):
            sub_order_2.append('x')

        for i in range(2):
            sub_order_2.append('y')
        for i in range(4):
            sub_order_2.append('x')
            sub_order_2.append('y')

        # sub_orders = recur_enumerate(4,6)

        sub_order_3 = []
        for i in range(10):
            sub_order_3.append('x')
            sub_order_3.append('y')

        for sub_order_1 in sub_orders:
            self.possible_orders.append(sub_order_1 + sub_order_2 + sub_order_3)

        '''initial the self.order with the first 1'''
        self.order = self.possible_orders[0]


    def output(self, datapoint):
        binary_form = self.transfer.transfer(datapoint)
        # print(binary_form)
        x_num = 0
        y_num = 0
        value = 0
        # print(binary_form)

        for i in range(len(self.order)):
            if self.order[i] == 'x':
                value = value * 2 + int(binary_form[0][x_num])
                x_num += 1
            else:
                value = value * 2 + int(binary_form[1][y_num])
                y_num += 1
        # print(value)
        return value

class HighQuiltsModule:
    def __init__(self, transfer, input_dim):
        self.dim = input_dim
        self.transfer = transfer
        length = {2: 20, 3: 14, 4: 10, 5: 8, 6: 7}
        area = [30, 32]

        self.order = []

        dim_len = length[input_dim]
        query_len = int(area[0] / input_dim)

        for _ in range(dim_len - query_len - 1):
            for i in range(input_dim):
                self.order.append(i)
        for i in range(input_dim - 2):
            self.order.append(i + 2)
        self.order.append(0)
        self.order.append(1)
        for _ in range(query_len):
            for i in range(input_dim):
                self.order.append(i)


    def output(self, datapoint):
        binary_form = self.transfer.transfer(datapoint)
        # print(binary_form)
        count = [0 for i in range(self.dim)]
        value = 0

        for i in range(len(self.order)):
            value = value * 2 + int(binary_form[self.order[i]][count[self.order[i]]])
            count[self.order[i]] += 1

        # print(value)
        return value


class QuiltsAllModule:
    def __init__(self, transfer):
        '''here store some orders follow restrict rule, jiangneng'''
        self.possible_orders = []


        self.transfer = transfer
        self.bit_length = transfer.count
        # print(self.bit_length)

        self.order = []

        sub_orders = recur_enumerate(4, 2)

        sub_order_2 = []
        for i in range(4):
            sub_order_2.append('y')

        sub_order_3 = []
        for i in range(2):
            sub_order_3.append('x')
        for i in range(14):
            sub_order_3.append('x')
            sub_order_3.append('y')

        for sub_order_1 in sub_orders:
            self.possible_orders.append(sub_order_1 + sub_order_2 + sub_order_3)

        '''initial the self.order with the first 1'''
        self.order = self.possible_orders[0]


    def output(self, datapoint):
        binary_form = self.transfer.transfer(datapoint)
        # print(binary_form)
        x_num = 0
        y_num = 0
        value = 0
        # print(binary_form)

        for i in range(len(self.order)):
            if self.order[i] == 'x':
                value = value * 2 + int(binary_form[0][x_num])
                x_num += 1
            else:
                value = value * 2 + int(binary_form[1][y_num])
                y_num += 1
        # print(value)
        return value
