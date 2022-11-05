# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.

import numpy as np
import pickle
from bmtree.tree import Tree


class BMTEnv:
    def __init__(self, dataset, query_workload, bit_length, best_tree_file_path, binary_transfer, smallest_split_card, max_depth):
        """
        This object providing a Reinforcement Learning Experiment Environment, which support
        the BMTREE constructing task
        :param dataset: as input information, dataset involves data_points with multi_dimension
        :param query_workload: as input information, query_workload involves with Range Query Rectangle
        :param best_tree_file_path: File Path saving the best tree construct, wrt. the comparing metrics
        """

        self.dataset = dataset
        self.query_workload = query_workload

        self.bit_length = bit_length
        self.binary_transfer = binary_transfer

        self.smallest_split_card = smallest_split_card
        self.max_depth = max_depth

        '''Initiate a Tree object'''
        '''Here we add a hyper-parameter for future tuning'''
        self.tree = Tree(bit_length=bit_length, binary_transfer=binary_transfer,
                         dataset=self.dataset, smallest_split_card=self.smallest_split_card, max_depth=self.max_depth)

        '''Support the saving and loading procedure of A Tree Object'''
        self.best_tree_path = best_tree_file_path

        if best_tree_file_path is None:
            self.best_tree = None
        else:
            self.best_tree = None  # TODO tree structure saving and loading methodology
            self.tree = Tree(bit_length=self.bit_length, binary_transfer=self.binary_transfer,
                             dataset=self.dataset, smallest_split_card=self.smallest_split_card, max_depth=self.max_depth)
            '''load the actions, and build the tree identically, jiangnengß'''
            with open(best_tree_file_path, 'r') as f:
                lines = f.readlines()
                lines = lines[2:]
                for line in lines:
                    line = line.split(" ")
                    # print(line)
                    if int(line[3]) == -1:
                        break
                    else:
                        if len(line) < 7:
                            _, _, _, _ = self.tree.take_action(int(line[3]))
                        else:
                            if int(line[-1]) == -1:
                                _, _, _, _ = self.tree.take_action(int(line[3]))
                            else:
                                _, _, _, _ = self.tree.take_action(int(line[3]), 0)

        return

    def reset(self):
        """
        This function reset the Environment which
        POSSIBLY save the current tree (Maybe), but set a new tree, etc. to make a new env
        :return:
        """

        '''Initiate a Tree object, replace the former one'''
        self.tree = Tree(bit_length=self.bit_length, binary_transfer=self.binary_transfer,
                         dataset=self.dataset, smallest_split_card=self.smallest_split_card, max_depth=self.max_depth)

        return None

    def step(self, action):
        """
        This function provides action to the env(Tree Object) to split tree nodes, return states/obs
        and perhaps reward if the tree construction is finished( done = True)
        :param action: the output of ppo algorithm, mostly a discrete probability distribution towards
        different actions(arms).
        :return: obs(states) rew(rewards) done(if an episode is finished) info(information, optional)
        """

        '''Feed action to the Tree, get tree's return result'''
        obs, rew, done, curve_range = self.tree.take_action(action)

        '''
        Fetch the mask of current, None if done = True
        ** Here we add count to say how many available actions
        '''
        mask, count = self.tree.get_mask()

        return obs, mask, rew, done, count, curve_range
    
    def unstep(self):
        self.tree.unaction()
        return
    
    def generate_tree_zorder(self, initial_action):

        '''With the new ZNode node class, we could change to a simple implementation for z order'''
        self.tree.root_node.chosen_bit = initial_action

        '''Now if child is None, then it will link to the heuristic node automatically'''
        # '''assign left and right children as the heuristic node, which will simply do the thing'''
        # self.tree.root_node.add_children(self.tree.heuristic_node, 'left')
        # self.tree.root_node.add_children(self.tree.heuristic_node, 'right')

        return

    def update_action(self, action):
        """
        This is the function connect agent with bmtree, update the bit of current node
        :param action:
        :return:
        """
        states, done = self.tree.update_bit(action)

        mask, count = self.tree.get_mask()

        return states, mask, done, count


    def save_if_best(self):
        """
        This function is called when tree construction is finished, compare new tree with the exists
        best tree
        :return: return signal if the best
        """

        '''Get a value of the new generated tree'''
        new_value = self.tree.get_value()

        '''Decide if generated tree is the best'''
        signal = False
        if self.best_tree_path is None:
            self.best_value = new_value
            self.best_tree = self.tree
            signal = True
        else:
            if self.best_value < new_value:
                self.best_value = new_value
                self.best_tree = self.tree
                signal = True

        return signal

    def get_state(self):
        """
        This function returns the observation/state of the environment right now
        :return: state (which is mostly a tensor)
        """
        return self.tree.get_state()

    def compute_reward(self, query_rewards, query_data_points):
        """
        This function calls tree's add_reward and recurse_reward function to generate rewards
        of inner nodes
        :param query_rewards:
        :param query_data_points:
        :return:
        """

        '''iterate for different query'''
        for i in range(len(query_rewards)):
            reward = query_rewards[i]
            data_points = query_data_points[i]

            self.tree.add_reward(data_points, reward)

        '''Call the recurse function'''
        result_rewards = self.tree.recurse_reward()

        return result_rewards

    def save(self, path, max_depth):
        """
        This function save the generated BMTree
        :param path:
        :param max_depth:
        :return:
        """
        self.tree.save(path, max_depth)
