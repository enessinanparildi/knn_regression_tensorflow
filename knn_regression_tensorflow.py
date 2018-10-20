import tensorflow as tf
import numpy as np
import matplotlib.pyplot as py

def make_dataset():
    np.random.seed(521)
    Data = np.linspace(1.0, 10.0, num=10000)[:, np.newaxis]
    Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(10000, 1)
    randIdx = np.arange(10000)
    np.random.shuffle(randIdx)
    traindata = Data[randIdx[:8000]], Target[randIdx[:8000]]
    validdata = Data[randIdx[8000:9000]], Target[randIdx[8000:9000]]
    testdata = Data[randIdx[9000:10000]], Target[randIdx[9000:10000]]
    return traindata, validdata, testdata


def euclid_distance(X, Z):
    N1 = tf.shape(X)
    N2 = tf.shape(Z)

    a = tf.matmul(X, tf.transpose(X))
    a = tf.diag_part(a)
    a = tf.reshape(a, [N1[0], 1])
    x = tf.tile(a, [1, N2[0]])

    b = tf.matmul(Z, tf.transpose(Z))
    b = tf.diag_part(b)
    z = tf.tile(b, [N1[0]])
    z = tf.reshape(z, [N1[0], N2[0]])

    c = 2 * tf.matmul(X, tf.transpose(Z))

    return x - c + z


# responsibility for  single test point
def hardresponsibility(D, k, graph):
    with graph.as_default():
        length = tf.shape(D)
        (maxkvals, maxkindicis) = tf.nn.top_k(D, k)
        scatter = tf.scatter_nd(maxkindicis, tf.to_float(tf.ones(tf.shape(maxkvals))) / tf.to_float(k), length)
        return scatter

def softresponsibility(D,lambd,graph):
    with graph.as_default():
        expdistance = tf.exp(-lambd*D)
        wholesum = tf.reduce_sum(expdistance)
        normalizedexp = tf.divide(expdistance,wholesum)
        return normalizedexp



def mseloss(prediction, target):
    return tf.losses.mean_squared_error(prediction, target)


def main_operation(i, distancesettrain, kvalue, lambdaval, n_train, training_target, graph, hard_mode):
    with graph.as_default():
        finalval = tf.cond(hard_mode, lambda: hard_op(distancesettrain[:,i], kvalue, graph, training_target, n_train),
                                      lambda: soft_op(distancesettrain[:,i], lambdaval, graph, training_target, n_train))
    return finalval

def hard_op(distances,kvalue,graph,training_target,n_train):
    with graph.as_default():
        resp = tf.reshape(hardresponsibility(-1 * distances, kvalue, graph), [n_train, 1])
        val = tf.reshape(tf.matmul(tf.transpose(tf.cast(training_target, dtype=tf.float32)), resp), [1])
    return val

def soft_op(distances,lvalue,graph,training_target,n_train):
    with graph.as_default():
        resp = tf.reshape(softresponsibility(distances, lvalue, graph), [n_train, 1])
        val = tf.reshape(tf.matmul(tf.transpose(tf.cast(training_target, dtype=tf.float32)), resp), [1])
    return val


def main_graph(traindata, validdata, testdata, kset, lambdaset):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():

        n_train = len(traindata[0])
        training_dataset = tf.convert_to_tensor(traindata[0], dtype=tf.float32)
        training_target = tf.convert_to_tensor(traindata[1], dtype=tf.float32)

        hard_mode = tf.placeholder(dtype=tf.bool, shape=())
        lvalue = tf.placeholder_with_default(input = tf.zeros((),dtype=tf.float32) ,  shape=())
        kvalue = tf.placeholder_with_default(input = tf.zeros((),dtype=tf.int32),  shape=())
        dataset = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        target = tf.placeholder(dtype=tf.float32, shape=[None,1])

        distancesettrain = euclid_distance(training_dataset, dataset)

        results = tf.map_fn(lambda i: main_operation(i, distancesettrain, kvalue,  lvalue , n_train, training_target, graph,hard_mode),
                            tf.range(tf.shape(dataset)[0]), dtype=(tf.float32))

        loss = mseloss(target, results)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init)
        datatargets = traindata[1].astype(np.float32)

        traininglossset = np.zeros(len(kset))
        validationlossset = np.zeros(len(kset))


        for index, k in enumerate(kset):
            traininglossset[index] = sess.run(loss, feed_dict={kvalue: k,
                                                               hard_mode:True,
                                                               dataset: traindata[0],
                                                               target:datatargets })
        # # validation loop
        for index, k in enumerate(kset):
            validationlossset[index] = sess.run(loss, feed_dict={kvalue: k,
                                                                 hard_mode: True,
                                                                 dataset: validdata[0],
                                                                 target: validdata[1].astype(np.float32)})
        chosenk = kset[np.argmin(validationlossset)]
        testlosshard, testpredhard = sess.run([loss, results], feed_dict={kvalue: chosenk,
                                                                  hard_mode: True,
                                                                  dataset: testdata[0],
                                                                  target: testdata[1].astype(np.float32)})

        traininglossset = np.zeros(len(lambdaset))
        validationlossset = np.zeros(len(lambdaset))


        for index, l in enumerate(lambdaset):
            traininglossset[index] = sess.run(loss, feed_dict={lvalue: l,
                                                               hard_mode:False,
                                                               dataset: traindata[0],
                                                               target:datatargets })
        # # validation loop
        for index, l in enumerate(lambdaset):
            validationlossset[index] = sess.run(loss, feed_dict={lvalue: l,
                                                                 hard_mode:False,
                                                                 dataset: validdata[0],
                                                                 target: validdata[1].astype(np.float32)})
        chosenl = lambdaset[np.argmin(validationlossset)]
        testlosssoft, testpredsoft = sess.run([loss, results], feed_dict={lvalue: chosenl,
                                                                          hard_mode:False,
                                                                          dataset: testdata[0],
                                                                          target: testdata[1].astype(np.float32)})

    return testlosshard, testpredhard,testlosssoft, testpredsoft, chosenk , chosenl


def plot_results(testdata, testpred, chosenk):
    py.scatter(testdata[0], testpred, s=5, c='g')
    py.scatter(testdata[0], testdata[1], s=5, c='b')
    py.title('k = ' + str(chosenk))
    py.xlabel('Data')
    py.ylabel('Target')
    py.grid(True)
    py.show()


