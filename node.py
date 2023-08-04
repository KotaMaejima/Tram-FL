import tensorflow as tf
from tensorflow import keras
import routing

class Node:
    def __init__(
        self, name, node_id, node_num, train_mask_list,
         routing_algorithm, BATCHSIZE, INTERVAL
         ):
        self.node_id = node_id
        self.node_num = node_num
        self.train_mask_list = train_mask_list
        self.mask_list = train_mask_list[name]
        self.routing_algorithm = routing_algorithm
        self.BATCHSIZE = BATCHSIZE
        self.INTERVAL = INTERVAL
        
    def decide_next_node(self, label_log):
        if self.routing_algorithm==0:
            next_node = routing.proposed_routing_algorithm(
                self.node_num, label_log, self.train_mask_list, self.BATCHSIZE, self.INTERVAL
                )
        elif 1<=self.routing_algorithm and self.routing_algorithm<=24:
            next_node = routing.static_routing_algorithm(self.node_num, self.node_id, self.routing_algorithm)
        elif self.routing_algorithm==25:
            next_node = routing.random_routing_algorithm(self.node_num)
        return next_node

    def loss(self, model, x, y, training):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)

        return loss_object(y_true=y, y_pred=y_)

    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, model, label_log, INTERVAL, lr):
        optimizer = keras.optimizers.SGD(learning_rate=lr)
        # Training loop - using batches of 32
        for i in range(INTERVAL):
            x, y = next(self.train_iter)
            
            # Optimize the model
            loss_value, grads = self.grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            for j in y:
                label_log[j] += 1
        next_node = self.decide_next_node(label_log)
        return model, label_log, next_node


class MNIST_Node(Node):
    def __init__(self, name, node_id, node_num, x_train, y_train, train_mask_list, BATCHSIZE, routing_algorithm, INTERVAL):
        super().__init__(name, node_id, node_num, train_mask_list, routing_algorithm, BATCHSIZE, INTERVAL)
        self.x_train = [x_train[i] for i, data in enumerate(y_train) if self.mask_list[data]]
        self.y_train = [i for i in y_train if self.mask_list[i]]
        self.data_num = len(self.x_train)
        
        self.traindata = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(self.data_num).repeat().batch(BATCHSIZE)
        self.train_iter = iter(self.traindata)


class CIFAR10_Node(Node):
    def __init__(self, name, node_id, node_num, x_train, y_train, train_mask_list, BATCHSIZE, routing_algorithm, INTERVAL):
        super().__init__(name, node_id, node_num, train_mask_list, routing_algorithm, BATCHSIZE, INTERVAL)
        self.x_train = [x_train[i] for i, data in enumerate(y_train) if self.mask_list[data[0]]]
        self.y_train = [i[0] for i in y_train if self.mask_list[i[0]]]
        self.data_num = len(self.x_train)
        
        self.traindata = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(self.data_num).repeat().batch(BATCHSIZE)
        self.train_iter = iter(self.traindata)


class IMdb_Node(Node):
    def __init__(self, name, node_id, node_num, x_train, y_train, train_mask_list, BATCHSIZE, routing_algorithm, INTERVAL):
        super().__init__(name, node_id, node_num, train_mask_list, routing_algorithm, BATCHSIZE, INTERVAL)

        x_train_0 = [x_train[i] for i, data in enumerate(y_train) if data==0]
        x_train_1 = [x_train[i] for i, data in enumerate(y_train) if data==1]
        y_train_0 = [i for i in y_train if i==0]
        y_train_1 = [i for i in y_train if i==1]

        x_train_0 = x_train_0[int(len(x_train_0)*self.mask_list[0]):int(len(x_train_0)*self.mask_list[1])]
        x_train_1 = x_train_1[int(len(x_train_1)*self.mask_list[2]):int(len(x_train_1)*self.mask_list[3])]
        y_train_0 = y_train_0[int(len(y_train_0)*self.mask_list[0]):int(len(y_train_0)*self.mask_list[1])]
        y_train_1 = y_train_1[int(len(y_train_1)*self.mask_list[2]):int(len(y_train_1)*self.mask_list[3])]

        x_train = x_train_0+x_train_1
        y_train = y_train_0+y_train_1

        data_num = len(x_train)
        
        self.traindata = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(data_num).repeat().batch(BATCHSIZE)
        self.train_iter = iter(self.traindata)
        

    def decide_next_node(self, label_log):
        if 1<=self.routing_algorithm and self.routing_algorithm<=24:
            next_node = routing.static_routing_algorithm(self.node_num, self.node_id, self.routing_algorithm)
        elif self.routing_algorithm==25:
            next_node = routing.random_routing_algorithm(self.node_num)
        return next_node

