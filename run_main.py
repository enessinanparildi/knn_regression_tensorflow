import numpy as np
import knn_regression_tensorflow


if __name__ == "__main__":
    kset  = np.array([1, 3, 5, 10, 50])
    traindata, validdata, testdata = knn_regression_tensorflow.make_dataset()
    testloss, testpred = knn_regression_tensorflow.main_graph(traindata, validdata, testdata, kset)
    print('Test mse loss= ' + str(testloss))
    knn_regression_tensorflow.plot_results(testdata, testpred)