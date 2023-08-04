from random import randint

def build_label_distribution(DATASET, node_num, label_initialization):
    if DATASET=="MNIST" or DATASET=="CIFAR10":
        if label_initialization=="fixed_value":
            if node_num==3:
                train_mask_list = {
                "node1": [True, True, True, False, False, False, False, False, False, False],
                "node2": [False, False, False, True, True, True, False, False, False, False],
                "node3": [False, False, False, False, False, False, True, True, True, True]
                }
            elif node_num==5:
                train_mask_list = {
                "node0": [True, True, False, False, False, False, False, False, False, False],
                "node1": [False, False, True, True, False, False, False, False, False, False],
                "node2": [False, False, False, False, True, True, False, False, False, False],
                "node3": [False, False, False, False, False, False, True, True, False, False],
                "node4": [False, False, False, False, False, False, False, False, True, True]
                }
            elif node_num==10:
                train_mask_list = {
                "node0": [True, False, False, False, False, False, False, False, False, False],
                "node1": [False, True, False, False, False, False, False, False, False, False],
                "node2": [False, False, True, False, False, False, False, False, False, False],
                "node3": [False, False, False, True, False, False, False, False, False, False],
                "node4": [False, False, False, False, True, False, False, False, False, False],
                "node5": [False, False, False, False, False, True, False, False, False, False],
                "node6": [False, False, False, False, False, False, True, False, False, False],
                "node7": [False, False, False, False, False, False, False, True, False, False],
                "node8": [False, False, False, False, False, False, False, False, True, False],
                "node9": [False, False, False, False, False, False, False, False, False, True]
                }
        elif label_initialization=="random":
            train_mask_list = {}
            while True:
                flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for node in range(node_num):
                    dist_list = [False, False, False, False, False, False, False, False, False, False]
                    for x in range(randint(2, 5)):
                        y = randint(0, 9)
                        dist_list[y] = True
                        flag[y] = 1
                    train_mask_list['node{}'.format(node)] = dist_list
                if sum(flag)==10:
                    break
    
    if DATASET=="IMdb":
        if label_initialization=="fixed_value":
            if node_num==3:
                train_mask_list={
                "node1": [0, 0.81, 0, 0.21],
                "node2": [0.81, 0.97, 0.21, 0.59],
                "node3": [0.97, 1, 0.59, 1]
                }
            elif node_num==5:
                train_mask_list={
                "node0": [0, 0.63, 0, 0.09],
                "node1": [0.63, 0.86, 0.09, 0.028],
                "node2": [0.86, 0.95, 0.28, 0.51],
                "node3": [0.95, 0.98, 0.51, 0.75],
                "node4": [0.98, 1, 0.75, 1]
                }
            elif node_num==10:
                train_mask_list={
                "node0": [0, 0.39, 0, 0.02],
                "node1": [0.39, 0.63, 0.02, 0.09],
                "node2": [0.63, 0.77, 0.09, 0.18],
                "node3": [0.77, 0.86, 0.18, 0.28],
                "node4": [0.86, 0.92, 0.28, 0.39],
                "node5": [0.92, 0.95, 0.39, 0.51],
                "node6": [0.95, 0.97, 0.51, 0.63],
                "node7": [0.97, 0.98, 0.63, 0.75],
                "node8": [0.98, 0.99, 0.75, 0.87],
                "node9": [0.99, 1, 0.87, 1]
                }
    return train_mask_list