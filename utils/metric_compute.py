# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.


'''
ExperimentEnv class set up the experimental environment, including ordering generate,
computing scanrange with Query input.
'''

class ExperimentEnv:
    def __init__(self, dataset, module=None, pagesize=5, module_name=None, core_num=20):
        # initialize dataset #
        self.dataset = list([])
        for datapoint in dataset:
            self.dataset.append(
                {'data': datapoint, 'value': 0, 'order': 0, 'page': 0})  # (dim0,dim1, value, order, pagenum)

        if module != None:
            self.module = module

        self.module_name = module_name

        self.pagesize = pagesize  # data points per page

        self.core_num = core_num

        self.value_page = []
        return

    '''Change the mapping module'''
    def change_module(self, module, module_name):
        # change order generating module
        self.module = module
        self.module_name = module_name

    def order_generate(self, compute_range=-1):
        # use module generate ordering, partition/mark #
        sort_list = []
        '''if not mention, regenerate whole order, otherwise generate order of the range accordingly.'''
        if compute_range == -1:
            for i in range(len(self.dataset)):
                value = self.module.output(self.dataset[i]['data'])
                self.dataset[i]['value'] = value

            self.dataset.sort(key=lambda x: x['value'])
        else:
            for i in range(compute_range[0], compute_range[1] + 1):
                value = self.module.output(self.dataset[i]['data'])
                self.dataset[i]['value'] = value
            self.dataset[compute_range[0]: compute_range[1]] = sorted(self.dataset[compute_range[0]: compute_range[1]],
                                                                      key=lambda x: x['value'])

        

        # free and reallocate
        del self.value_page[:]
        del self.value_page
        self.value_page = []

        current_page = 0
        self.value_page.append([current_page, self.dataset[0]['value']])
        for i in range(len(self.dataset)):
            self.dataset[i]['order'] = i
            self.dataset[i]['page'] = int(i / self.pagesize)

            '''Now we store the vlaue-block mapping information'''
            if current_page < self.dataset[i]['page']:
                current_page = self.dataset[i]['page']
                self.value_page.append([current_page, self.dataset[i]['value']])
        return

    def value_to_position(self, value):
        """
        This function get the position within an order
        :param value:
        :return:
        """
        for i in range(len(self.dataset)):
            if value == self.dataset[i]['value']:
                return i
            elif value < self.dataset[i]['value']:
                return [i - 1, i]
        '''if the scan is end'''
        return len(self.dataset) - 1

    def run_query_fast(self, query):
        """
        This function only on scan range objective, as fast as possible
        :param query:
        :return:
        """

        if self.module_name != 'hilbert':
            # print(self.module_name)
            # data_min = query.start_position
            data_min = query.min_point
            value_min = self.module.output(data_min)
            # data_max = [int(query.start_position[0] + query.length - 1), int(query.start_position[1] + query.width - 1)]
            data_max = query.max_point
            value_max = self.module.output(data_max)

            start_scan = 0
            end_scan = 0

            start_flag = 0
            for i in range(len(self.value_page) - 1):
                current_page = self.value_page[i][0]
                value = self.value_page[i][1]
                next_value = self.value_page[i+1][1]

                if start_flag == 0:
                    if value_min >=  value and value_min < next_value:
                        start_scan = current_page
                        start_flag = 1
                        if i == len(self.value_page) - 2:
                            end_scan = self.value_page[i + 1][0]
                else:
                    if value_max >= value and value_max < next_value:
                        end_scan = current_page
                        break
                    if i == len(self.value_page) - 2:
                        end_scan = self.value_page[i + 1][0]

            scan_range = end_scan - start_scan + 1
            # print(start_scan, end_scan)
        return scan_range

    def fast_compute_scan_range(self, queries):
        scan_range = 0

        for i in range(len(queries)):
            scan_range += self.run_query_fast(queries[i])

        scan_range /= len(queries)
        return scan_range


if __name__ == '__main__':

    pass