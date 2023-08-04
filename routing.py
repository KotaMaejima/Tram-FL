from random import randint
import statistics

def static_routing_algorithm(node_num, node_id, routing_id):
    if node_num==3:
        static_routing_dict = {
            1:[0, 1, 2], 2:[0, 2, 1]
        }
    elif node_num==5:
        static_routing_dict = {
            1:[0, 1, 2, 3, 4], 2:[0, 1, 2, 4, 3], 3:[0, 1, 3, 2, 4],
            4:[0, 1, 3, 4, 2], 5:[0, 1, 4, 2, 3], 6:[0, 1, 4, 3, 2],
            7:[0, 2, 1, 3, 4], 8:[0, 2, 1, 4, 3], 9:[0, 2, 3, 1, 4],
            10:[0, 2, 3, 4, 1], 11:[0, 2, 4, 1, 3], 12:[0, 2, 4, 3, 1],
            13:[0, 3, 1, 2, 4], 14:[0, 3, 1, 4, 2], 15:[0, 3, 2, 1, 4],
            16:[0, 3, 2, 4, 1], 17:[0, 3, 4, 1, 2], 18:[0, 3, 4, 2, 1],
            19:[0, 4, 1, 2, 3], 20:[0, 4, 1, 3, 2], 21:[0, 4, 2, 1, 3],
            22:[0, 4, 2, 3, 1], 23:[0, 4, 3, 1, 2], 24:[0, 4, 3, 2, 1]
        }

    static_routing = static_routing_dict[routing_id]
    index = static_routing.index(node_id)
    if index==node_num-1:
        next_node = static_routing[0]
    else:
        next_node = static_routing[index+1]
    return next_node

def random_routing_algorithm(node_num):
    if node_num==3:
        next_node = randint(0, 2)
    elif node_num==5:
        next_node = randint(0, 4)
    elif node_num==10:
        next_node = randint(0, 9)
    return next_node

def proposed_routing_algorithm(node_num, label_log, train_mask_list, BATCHSIZE, INTERVAL):
    variance = []
    for i in range(node_num):
        label_expectation = label_log.copy()
        label = [j for j, x in enumerate(train_mask_list["node{}".format(i)]) if x]
        for j in label:
            label_expectation[j] += (BATCHSIZE*INTERVAL/len(label))
        variance.append(statistics.pvariance(label_expectation))
    next_node = variance.index(min(variance))
    return next_node
    