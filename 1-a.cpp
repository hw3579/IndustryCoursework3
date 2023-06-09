#include <stdbool.h>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include "CV.h"
#include"tensorflow.h"
#include <Python.h>


using namespace std;
using namespace cv;
using namespace cv::dnn;


int main() {
    string def_front_path = "C:/Users/A/Desktop/daima/Industry_CW3/train/def_front";
    string ok_front_path = "C:/Users/A/Desktop/daima/Industry_CW3/train/ok_front";

    vector<pair<Mat, int>> imagesTrain_def = load_images_labels(def_front_path, 1);
    vector<pair<Mat, int>> imagesTrain_ok = load_images_labels(ok_front_path, 0);

    vector<pair<Mat, int>> imagesTrain;
    imagesTrain.insert(imagesTrain.end(), imagesTrain_def.begin(), imagesTrain_def.end());
    imagesTrain.insert(imagesTrain.end(), imagesTrain_ok.begin(), imagesTrain_ok.end());

    // Stack all images in a single Mat variable
    Mat imagesTrainMat;
    for (const auto& image_label : imagesTrain) {
        imagesTrainMat.push_back(image_label.first.reshape(1, 1)); // Flatten and add the image
    }


    /// 1-a


    // Normalize the data
    imagesTrainMat.convertTo(imagesTrainMat, CV_32F, 1.0 / 255.0);

    // Split the training data into training and validation sets
    float test_ratio = 0.25;
    int total_samples = imagesTrainMat.rows;
    int test_samples = static_cast<int>(total_samples * test_ratio);
    int train_samples = total_samples - test_samples;

    Mat imagesTrainSubset = imagesTrainMat(Rect(0, 0, imagesTrainMat.cols, train_samples));
    Mat imagesValSubset = imagesTrainMat(Rect(0, train_samples, imagesTrainMat.cols, test_samples));

    // Assuming you have labels in a single Mat
    Mat labelsTrain;
    for (const auto& image_label : imagesTrain) {
        labelsTrain.push_back(image_label.second);
    }

    Mat imageLabelsTrainSubset = labelsTrain(Rect(0, 0, labelsTrain.cols, train_samples));
    Mat imageLabelsValSubset = labelsTrain(Rect(0, train_samples, labelsTrain.cols, test_samples));

    /*
    
    完成数据录入 
    
    */

    // Convert cv::Mat to TF_Tensor
    TF_Tensor* images_train_tensor = convert_mat_to_tf_tensor(imagesTrainSubset);
    TF_Tensor* images_val_tensor = convert_mat_to_tf_tensor(imagesValSubset);

    // Convert cv::Mat labels data to TensorFlow tensors
    TF_Tensor* labels_train_tensor = convert_labels_to_tf_tensor(imageLabelsTrainSubset);
    TF_Tensor* labels_val_tensor = convert_labels_to_tf_tensor(imageLabelsValSubset);


    /*
    数据转换完毕
    */


    Py_Initialize();

    PyRun_SimpleString("print ('hello')");

    PyRun_SimpleString("import numpy as np");

    Py_Finalize();

    system("pause");









    // Don't forget to delete the tensors after you're done
    TF_DeleteTensor(images_train_tensor);
    TF_DeleteTensor(images_val_tensor);

    TF_DeleteTensor(labels_train_tensor);
    TF_DeleteTensor(labels_val_tensor);






    system("pause");
    // Note: For loading test data, the process would be the same as the training data.
    // You will need to call 'load_images_labels' function with the test folder paths.
    return 0;
}

