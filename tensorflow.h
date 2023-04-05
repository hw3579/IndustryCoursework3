#pragma once
#ifndef TENSORFLOW_H
#define TENSORFLOW_H

#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>

using namespace cv;

extern TF_Tensor* convert_mat_to_tf_tensor(const Mat& mat_data) {
    // Reshape the Mat data to add channel information
    Mat reshaped_mat = mat_data.reshape(1, mat_data.rows);

    // Get the size of the reshaped Mat
    int64_t dims[] = { reshaped_mat.rows, reshaped_mat.cols };

    // Allocate a TensorFlow tensor with the same dimensions and data type as the reshaped Mat
    TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, reshaped_mat.total() * reshaped_mat.elemSize());

    // Copy the reshaped Mat data to the TensorFlow tensor
    memcpy(TF_TensorData(tensor), reshaped_mat.data, reshaped_mat.total() * reshaped_mat.elemSize());

    return tensor;
}

extern TF_Tensor* convert_labels_to_tf_tensor(const Mat& mat_labels) {
    // Get the size of the labels Mat
    int64_t dims[] = { mat_labels.rows };

    // Allocate a TensorFlow tensor with the same dimensions and data type as the labels Mat
    TF_Tensor* tensor = TF_AllocateTensor(TF_INT32, dims, 1, mat_labels.total() * mat_labels.elemSize());

    // Copy the labels Mat data to the TensorFlow tensor
    memcpy(TF_TensorData(tensor), mat_labels.data, mat_labels.total() * mat_labels.elemSize());

    return tensor;
}






#endif