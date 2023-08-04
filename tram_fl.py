import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import node
import build_model
import label_distribution

DATASET = "MNIST" #select "MNIST", "CIFAR10" or "IMdb"
NODE_AMOUNT = 5 #select 3, 5 or 10
TRANSMISSION_NUM = 100
INTERVAL = 1
BATCHSIZE = 100
TEST_INTERVAL = 20
LEARNING_RATE = 0.005
LABEL_DISTRIBUTION = "fixed_value" #select "fixed_value" or "random"(MNIST, CIFAR10 only)
ROUTING_ALGORITHM = 1 #select model routing algorithm from 0 to 25 (IMdb can't select 0)

path = "result/{}/{}nodes/".format(DATASET, NODE_AMOUNT)

def test(model, x_test, y_test, test_accuracy_list):
    testdata = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCHSIZE)   
    test_accuracy = tf.keras.metrics.Accuracy()

    if DATASET=="MNIST" or DATASET=="CIFAR10":
        class_correct = [0,0,0,0,0,0,0,0,0,0]
        class_total = [0,0,0,0,0,0,0,0,0,0]
    elif DATASET=="IMdb":
        class_correct = [0,0]
        class_total = [0,0]

    for (x, y) in testdata:
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
        for i in range(BATCHSIZE):
            label = int(y[i])
            if int(prediction[i])==label:
                class_correct[label] += 1
            class_total[label] += 1

    for i in range(len(class_correct)):
        class_result = class_correct[i] / class_total[i]
        print("Test sett accuracy of {}: {:.3%}".format(i, class_result))   

    print("Test set accuracy of All: {:.3%}".format(test_accuracy.result()))
    test_accuracy_list.append(test_accuracy.result()*100)
    return test_accuracy_list


def plot(test_accuracy_list, transmision_num_list, iter_num):
    x1 = list(range(TEST_INTERVAL, iter_num, TEST_INTERVAL))
    if x1[-1]!=iter_num:
        x1.append(iter_num)
    x2 = transmision_num_list
    fig, ax1 = plt.subplots()

    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(right=0.85)
    fig.subplots_adjust(left=0.15)

    ax1.plot(x1,test_accuracy_list)
    ax1.set_xlabel("update")

    ax2 = ax1.twiny()

    ax2.plot(x2,test_accuracy_list)
    ax2.set_xlabel("communication")

    plt.savefig(path+"convergence_curve.png")



def main():
    train_mask_list = label_distribution.build_label_distribution(DATASET, NODE_AMOUNT, LABEL_DISTRIBUTION)
    if DATASET=="MNIST":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = x_train/255.0, x_test/255.0
        model = build_model.build_mnist_model()
        label_log = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if NODE_AMOUNT==3 or NODE_AMOUNT==5 or NODE_AMOUNT==10:
            node0 = node.MNIST_Node("node0", 0, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
            node1 = node.MNIST_Node("node1", 1, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
            node2 = node.MNIST_Node("node2", 2, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
            if NODE_AMOUNT==5 or NODE_AMOUNT==10:
                node3 = node.MNIST_Node("node3", 3, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                node4 = node.MNIST_Node("node4", 4, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                if NODE_AMOUNT==10:
                    node5 = node.MNIST_Node("node5", 5, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                    node6 = node.MNIST_Node("node6", 6, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                    node7 = node.MNIST_Node("node7", 7, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                    node8 = node.MNIST_Node("node8", 8, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                    node9 = node.MNIST_Node("node9", 9, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
    elif DATASET=="CIFAR10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train, x_test = x_train/255.0, x_test/255.0
        model = build_model.build_cifar10_model()
        label_log = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if NODE_AMOUNT==3 or NODE_AMOUNT==5 or NODE_AMOUNT==10:
            node0 = node.CIFAR10_Node("node0", 0, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
            node1 = node.CIFAR10_Node("node1", 1, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
            node2 = node.CIFAR10_Node("node2", 2, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
            if NODE_AMOUNT==5 or NODE_AMOUNT==10:
                node3 = node.CIFAR10_Node("node3", 3, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                node4 = node.CIFAR10_Node("node4", 4, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                if NODE_AMOUNT==10:
                    node5 = node.CIFAR10_Node("node5", 5, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                    node6 = node.CIFAR10_Node("node6", 6, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                    node7 = node.CIFAR10_Node("node7", 7, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                    node8 = node.CIFAR10_Node("node8", 8, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                    node9 = node.CIFAR10_Node("node9", 9, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
    elif DATASET=="IMdb":
        (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=80)
        model = build_model.build_imdb_model()
        label_log = [0, 0]
        if NODE_AMOUNT==3 or NODE_AMOUNT==5:
            node0 = node.IMdb_Node("node0", 0, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
            node1 = node.IMdb_Node("node1", 1, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
            node2 = node.IMdb_Node("node2", 2, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
            if NODE_AMOUNT==5:
                node3 = node.IMdb_Node("node3", 3, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
                node4 = node.IMdb_Node("node4", 4, NODE_AMOUNT, x_train, y_train, train_mask_list, BATCHSIZE, ROUTING_ALGORITHM, INTERVAL)
    
    model.summary()

    test_accuracy_list = []
    next_node = 0
    transmission_num = 0
    transmision_num_list = []
    iter_num = 0

    while True:
        if next_node==0:
            model, label_log, next_node = node0.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=0:
                transmission_num += 1
        elif next_node==1:
            model, label_log, next_node = node1.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=1:
                transmission_num += 1
        elif next_node==2:
            model, label_log, next_node = node2.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=2:
                transmission_num += 1
        elif next_node==3:
            model, label_log, next_node = node3.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=3:
                transmission_num += 1
        elif next_node==4:
            model, label_log, next_node = node4.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=4:
                transmission_num += 1
        elif next_node==5:
            model, label_log, next_node = node5.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=5:
                transmission_num += 1
        elif next_node==6:
            model, label_log, next_node = node6.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=6:
                transmission_num += 1
        elif next_node==7:
            model, label_log, next_node = node7.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=7:
                transmission_num += 1
        elif next_node==8:
            model, label_log, next_node = node8.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=8:
                transmission_num += 1
        elif next_node==9:
            model, label_log, next_node = node9.train(model, label_log, INTERVAL, LEARNING_RATE)
            if next_node!=9:
                transmission_num += 1
        iter_num += INTERVAL

        if (iter_num)%TEST_INTERVAL==0:
            transmision_num_list.append(transmission_num-1)
            print("Iter{}".format(iter_num))
            test_accuracy_list = test(model, x_test, y_test, test_accuracy_list)
        
        if transmission_num-1==TRANSMISSION_NUM:
            transmision_num_list.append(transmission_num-1)
            print("Iter{}".format(iter_num))
            test_accuracy_list = test(model, x_test, y_test, test_accuracy_list)
            break
    
    plot(test_accuracy_list, transmision_num_list, iter_num)
    model.save(path+'my_model')

if __name__=="__main__":
    main()