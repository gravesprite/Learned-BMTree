# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.

import numpy as np


class Query:
    '''
    Query:
    Area with 16 64 256 1024 4096, height/width ratio with 16 4 1 1/4 1/16
    Query distribution follows Gaussian with rng generated mu. Should be an object
    '''
    def __init__(self, query, if_point=False):
        """
        This function initiate the query object, need to mention it's only for two dimensional data
        :param query:
        :param if_point:
        """

        '''When the query is from [x1, x2, y1, y2 stuff]'''
        if len(query)> 3:
        # if if_point:
            self.min_point = query[0: int(len(query)/2)]
            self.max_point = query[int(len(query)/2) :]
            # self.start_position = [query[0], query[2]]
            # self.length = query[1] - query[0]
            # self.width = query[3] - query[2]
            # self.area = self.length * self.width
            # self.ratio = self.length / self.width
            return

        self.area = query[0]
        self.ratio = query[1]
        self.min_point = query[2]
        self.width = np.sqrt(self.area / self.ratio)
        self.length = self.area / self.width
        self.max_point = [query[2][0] + self.length, query[2][1] + self.width]
        return

    def inside(self, data):
        # check if a datapoint inside the query rectangle
        location: object = data['data']
        for i in range(len(location)):
            if self.min_point[i] <= location[i] < self.max_point[i]:
                continue
            else:
                return False
        return True

        # if (self.start_position[0] <= location[0] < self.start_position[0] + self.length) and \
        #         (self.start_position[1] <= location[1] < self.start_position[1] + self.width):
        #     return True
        # else:
        #     return False
