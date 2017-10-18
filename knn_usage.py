import cifar10
import knn
import numpy as np

if __name__=="__main__":
    cf=cifar10.cifar10_data()
    img_data=cf.readCIFRData()
    split_data=cf.splitTrainTest(img_data,5000,500)
    x_train=split_data[0]
    y_train=split_data[1]
    x_test=split_data[2]
    y_test_actual=split_data[3]
    knn=knn.knn()
    knn.train(x_train,y_train)
    y_test_pred=knn.predict(x_test)
    y_diff=y_test_actual==y_test_pred
    y_diff.astype(int)
    accuracy=np.sum(y_diff)/y_diff.shape[0]
    print("Prediction Accuracy:{:04.2f}% ".format(accuracy*100))
