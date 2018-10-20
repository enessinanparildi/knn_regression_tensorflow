import numpy as np
import knn_regression_tensorflow


if __name__ == "__main__":
    kset  = np.array([1, 3, 5, 10, 50])
    lambdaset= np.linspace(0.1, 50 , 100)

    traindata, validdata, testdata  = knn_regression_tensorflow.make_dataset()
    testlosshard, testpredhard, testlosssoft, testpredsoft, chosenk , chosenl = knn_regression_tensorflow.main_graph(traindata, validdata, testdata, kset, lambdaset)
    print('Test mse loss hard = ' + str(testlosshard))
    print('Test mse loss soft = ' + str(testlosssoft))

    knn_regression_tensorflow.plot_results(testdata, testpredhard, chosenk)
    knn_regression_tensorflow.plot_results(testdata, testpredsoft, chosenl)
