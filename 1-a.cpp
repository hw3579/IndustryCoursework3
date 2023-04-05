#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/core/types.hpp>



#include <stdio.h>
#include <tensorflow/c/c_api.h>


using namespace std;
using namespace cv;
using namespace cv::dnn;


vector<pair<Mat, int>> load_images_labels(const string& folder_path, int label) {
    vector<pair<Mat, int>> images_labels;
    vector<String> file_names;
    glob(folder_path + "/*", file_names);

    for (const auto& file_name : file_names) {
        Mat img = imread(file_name, IMREAD_GRAYSCALE);
        if (!img.empty()) {
            Mat img_resized;
            resize(img, img_resized, Size(150, 150));
            images_labels.push_back(make_pair(img_resized, label));
        }
    }
    return images_labels;
}

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

    // Note: For loading test data, the process would be the same as the training data.
    // You will need to call 'load_images_labels' function with the test folder paths.



    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, session_opts, status);


    // Add the first Conv2D layer
    {
        // Create the Conv2D operation and its attributes
        TF_OperationDescription* conv2d_op_desc = TF_NewOperation(graph, "Conv2D", "conv2d");
        // Set the attributes and input tensors
        // ...
    }

    // Add the first MaxPooling2D layer
    {
        // Create the MaxPool operation and its attributes
        TF_OperationDescription* maxpool_op_desc = TF_NewOperation(graph, "MaxPool", "maxpool");
        // Set the attributes and input tensors
        // ...
    }

    // Add the second Conv2D layer
    {
        // Create the Conv2D operation and its attributes
        TF_OperationDescription* conv2d_op_desc = TF_NewOperation(graph, "Conv2D", "conv2d_1");
        // Set the attributes and input tensors
        // ...
    }

    // Add the second MaxPooling2D layer
    {
        // Create the MaxPool operation and its attributes
        TF_OperationDescription* maxpool_op_desc = TF_NewOperation(graph, "MaxPool", "maxpool_1");
        // Set the attributes and input tensors
        // ...
    }

    // Add the Flatten layer
    {
        // Create the Reshape operation and its attributes
        TF_OperationDescription* reshape_op_desc = TF_NewOperation(graph, "Reshape", "flatten");
        // Set the attributes and input tensors
        // ...
    }

    // Add the first Dense layer
    {
        // Create the MatMul operation and its attributes
        TF_OperationDescription* matmul_op_desc = TF_NewOperation(graph, "MatMul", "dense");
        // Set the attributes and input tensors
        // ...
    }

    // Add the second Dense layer
    {
        // Create the MatMul operation and its attributes
        TF_OperationDescription* matmul_op_desc = TF_NewOperation(graph, "MatMul", "dense_1");
        // Set the attributes and input tensors
        // ...
    }

    // Add the Sigmoid activation layer
    {
        // Create the Sigmoid operation and its attributes
        TF_OperationDescription* sigmoid_op_desc = TF_NewOperation(graph, "Sigmoid", "sigmoid");
        // Set the attributes and input tensors
        // ...
    }

    system("pause");

    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    // Train the CNN model
    //train_cnn_model(cnn_model, X_train, y_train, X_val, y_val);

    // TODO: Evaluate the CNN model on the test set using your deep learning library

    return 0;
}

