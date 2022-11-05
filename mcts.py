# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.

from abc import ABC, abstractmethod
from collections import defaultdict
from torch.distributions import Categorical
import torch
import math
import random
from bmtree.tree import Tree
from random import randrange

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, initial_scanrange, bmtree, split_depth=5, max_depth=10, exploration_weight=1, roll_greedy=True):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.G = defaultdict(bool)  # if greedy instance is generated
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.initial_scanrange = initial_scanrange + 300
        self.split_depth = split_depth
        self.max_depth = max_depth

        self.roll_greedy = roll_greedy


        state, _, _ = bmtree.get_all_states()
        self.children[state] = set()
        self.N[state] += 1
        self.Q[state] += self.initial_scanrange

    def choose(self, tree: Tree):  # or bestaction function in the pseudo code

        actions, _, _ = tree.get_all_states()
        state = actions

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[state], key=score)[len(actions):], self.initial_scanrange - max(
            score(n) for n in self.children[state])

    def do_rollout(self, tree, env, query):  # in our LSFC problem can limit a length of the path
        "Make the tree one layer better. (Train for one iteration.)"

        state, _, _ = tree.get_all_states()

        path = self._select(tree, env, query)

        self._expand(path)

        reward = self._simulate(tree, env, query)

        self._backpropagate(path, reward)

        tree.unstep(
            int((len(path[-1]) - len(state))/2))

        return path, self.initial_scanrange - reward


    def _select(self, tree: Tree, env, query):
        "Find an unexplored descendent of `node`"
        path = []

        # we give a try to dig the rollout deeper
        # dig_more_step = 1

        while True:
            actions, masks, _ = tree.get_all_states()
            state = actions

            path.append(state)

            if state not in self.children:
                return path



            '''mark the greedy step'''
            if self.roll_greedy and self.G[state] is False:
                taken_actions = ""
                for mask in masks:
                    greed_perf = [0 for i in range(len(mask))]
                    for i in range(len(mask)):
                        if mask[i] == 1:
                            action = i

                            _, _, _, curve_range = tree.take_action(action)

                            # get the performance
                            env.order_generate(curve_range)
                            scan_range = env.fast_compute_scan_range(query)
                            greed_perf[i] = scan_range

                            tree.unaction()  # remove the current step
                        else:
                            greed_perf[i] = float("inf")
                    action = greed_perf.index(min(greed_perf))

                    tree.take_action(action)
                    taken_actions = taken_actions + str(action)
                self.G[state] = True

                continue

            if len(masks) < 10:
                children_actions = tree.generate_children_action()  # TODO: bmtree generate all child nodes
            else:
                children_actions = tree.sample_children_action(1000)

            if tree.current_node.depth <= self.split_depth:
                split = [1 for i in range(len(masks))]
            elif tree.current_node.depth <= self.max_depth:
                split = [randrange(2) for i in range(len(masks))]
            else:
                split = [0 for i in range(len(masks))]

            children = set()
            for children_action in children_actions:
                children_state = ''
                for i, ac in enumerate(children_action):
                    children_state += ac
                    children_state += str(split[i])
                children_state = state + children_state
                children.add(children_state)

            unexplored = children - self.children[state]

            if unexplored:
                prob = random.uniform(0, 1)
                if prob >= 0.0 or len(self.children[state]) == 0:  # Set the probability on explore unexplored node
                    n = unexplored.pop()

                    tree.multi_action(n[len(actions):])

            else:
                node = self._uct_select(tree)  # TODO: here we do not store all possible nodes
                tree.multi_action(node[len(actions):])

    def _expand(self, path):
        "Update the `children` dict with the children of `node`"
        if path[-1] in self.children:
            return  # already expanded
        if len(path) == 1:  # if it is the root node, no need to expand
            self.children[path[-1]] = set()
            return
        self.children[path[-2]].add(path[-1])
        self.children[path[-1]] = set()

    def _simulate(self, tree, env, query):
        "Returns the reward for a random simulation (to completion) of `node`"
        # invert_reward = True

        env.change_module(tree, module_name='bmtree')
        env.order_generate()
        scan_range = env.fast_compute_scan_range(query)

        reward = self.initial_scanrange - scan_range
        return reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):

            if self.Q[node] == 0:
                self.Q[node] += reward
            else:
                mean_reward = self.Q[node] / self.N[node]
                if mean_reward > reward:
                    self.Q[node] += mean_reward
                else:
                    # self.Q[node] += reward
                    self.Q[node] = reward * (self.N[node] + 1)
            self.N[node] += 1

    def _uct_select(self, tree):
        "Select a child of node, balancing exploration & exploitation"
        actions, _, _ = tree.get_all_states()
        state = actions
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[state]), '{},{}'.format(self.children,
                                                                                     self.children[state])

        log_N_vertex = math.log(self.N[state])

        def uct(n):  # TODO: check if there is requirement of reward fo uct search
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[state], key=uct)
