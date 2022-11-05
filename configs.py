# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.

import argparse


def set_config():
    """
    This function describe the configuration of a specific experiment
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='put into which device', type=int, default=0)
    parser.add_argument('--result_appendix', help='add the string at the end of the file', default='')

    parser.add_argument('--train_episode', help='tree generating epochs', type=int, default=300)
    parser.add_argument('--data', help='which dataset to use', default='uniform_1000000')
    parser.add_argument('--query', help='which query workload to use, (uniform, normal, skew, design)',
                        default='skew_1000_dim2')


    parser.add_argument('--method',
                        help='in which method you use, quilts, mcts \
                             or prioritized experience replay', default='mcts')

    parser.add_argument('--query_split_rate', help='ratio query workload split for training and testing',
                        type=float, default=0.8)

    # parser.add_argument('--data_sample_rate', help='ratio data is sampled during training', type=float,
    #                     default=0.05)
    # parser.add_argument('--data_sample_rate', help='ratio data is sampled during training', type=float,
    #                     default=0.1)
    parser.add_argument('--data_sample_points', help='data points sampled', type=int,
                        default=10**5)

    parser.add_argument('--lr_actor', help='actor net\'s learning rate', type=float, default=0.001)
    parser.add_argument('--lr_critic', help='critic net\'s learning rate', type=float, default=0.05)
    parser.add_argument('--smallest_split_card', help='stop split and put to heuristic node, 0 means split all',
                        type=int, default=0)

    '''hyperparameters that heuristic opt method, and prioritized replay method could have'''
    parser.add_argument('--max_depth', help='maximum depth the agent give influence to', type=int, default=8)
    parser.add_argument('--split_depth', help='maximum depth the agent give influence to', type=int, default=5)
    parser.add_argument('--action_depth', help='maximum action depth the agent give influence to', type=int, default=10)

    parser.add_argument('--rollouts', help='maximum action depth the agent give influence to', type=int, default=10)
    parser.add_argument('--discount_rate_hierarchical',
                        help='discount rate of bellman equation computing, following branching',
                        type=float, default=0.9)

    parser.add_argument('--if_hilbert', help='If test hilbert curve', type=int, default=0)

    '''setting of experiment environment, data space's scale'''
    parser.add_argument('--bit_length', help='the binary length of data space ', type=int, nargs="+", default=[20, 20])

    parser.add_argument('--page_size', help='the number of data records per block', type=int, default=50)

    parser.add_argument('--core_num', help='muptiprocessing the query excute', type=int, default=30)

    '''settings for postgresql experiment test'''
    parser.add_argument('--bmtree', help='which bmtree to use', default='bmtree_uni_skew')
    parser.add_argument('--quilts_ind', help='which quilts curve to use', type=int, default=0)
    parser.add_argument('--pg_test_method', help='what method to test (z_curve, quilts, bmtree)',
                        default='bmtree')
    parser.add_argument('--knn_number', help='knn query query neighbor number', type=int, default=25)
    parser.add_argument('--warm_cache_repeat', help='knn query query neighbor number', type=int, default=3)
    parser.add_argument('--db_password', help='the password of the dbsystem', default='1234')



    args = parser.parse_args()

    return args
