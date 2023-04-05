#include "CV.h"
#include <stdbool.h>
#include <stdio.h>
#include <tensorflow/c/c_api.h>


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

    // Normalize the data
    imagesTrainMat.convertTo(imagesTrainMat, CV_32F, 1.0 / 255.0);

    // Split the training data into training and validation sets
    float test_ratio = 0.25;
    int total_samples = imagesTrainMat.rows;
    int test_samples = static_cast<int>(total_samples * test_ratio);
    int train_samples = total_samples - test_samples;

    Mat X_train = imagesTrainMat(Rect(0, 0, imagesTrainMat.cols, train_samples));
    Mat X_val = imagesTrainMat(Rect(0, train_samples, imagesTrainMat.cols, test_samples));

    // Assuming you have labels in a single Mat
    Mat labelsTrain; // Load your imageLabelsTrain data here
    for (const auto& image_label : imagesTrain) {
        labelsTrain.push_back(image_label.second);
    }

    Mat y_train = labelsTrain(Rect(0, 0, labelsTrain.cols, train_samples));
    Mat y_val = labelsTrain(Rect(0, train_samples, labelsTrain.cols, test_samples));

    // Reshape the images to add channel information
    X_train = X_train.reshape(1, X_train.rows);
    X_val = X_val.reshape(1, X_val.rows);








    system("pause");
    // Note: For loading test data, the process would be the same as the training data.
    // You will need to call 'load_images_labels' function with the test folder paths.
    return 0;
}

