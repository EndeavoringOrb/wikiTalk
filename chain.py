import os
import pickle
from itertools import product
from tqdm import tqdm
import random
from time import perf_counter, sleep
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

#################################################
#################################################
#################################################
#################################################
#################################################
#################################################

class Value:
    def __init__(self) -> None:
        self.num_children = 0
        self.children = {}

    def add(self, num):
        self.num_children += 1
        if num not in self.children:
            self.children[num][0] = 1
        else:
            self.children[num][0] += 1

class Nodes:
    def __init__(self) -> None:
        self.sequences = [[0,1,2], [3,1,4]]

        self.root = Value()

        self.nodes = []
        self.create_chain()

    def create_chain(self):
        for sequence in self.sequences:
            node = self.root
            for tokIdx, token in enumerate(tqdm(sequence)):
                if token not in node.children:
                    if tokIdx == len(sequence) - 1:
                        node.children[token] = (1, -1)
                    else:
                        node.children[token] = (1, len(self.nodes))
                        node = Value()
                        self.nodes.append(node)
                else:
                    node.children[token][0] += 1
                    node = self.nodes[node.children[token][1]]

        print(f"Finished creating {len(self.nodes)} nodes.")

if __name__ == "__main__":
    nodes = Nodes()
    print()