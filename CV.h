#pragma once
#ifndef CV_H
#define CV_H

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


using namespace std;
using namespace cv;
using namespace cv::dnn;

extern vector<pair<Mat, int>> load_images_labels(const string& folder_path, int label) {
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









#endif