//
// Created by pointer on 17-12-6.
//

#ifndef DEPTHOPTIMIZATION_STATIC_PARA_H
#define DEPTHOPTIMIZATION_STATIC_PARA_H

#include <Eigen/Core>

const int kCamHeight = 1024;
const int kCamWidth = 1280;
const int kCamVecSize = kCamHeight * kCamWidth;
const int kProHeight = 800;
const int kProWidth = 1280;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ImgMatrix;

#endif //DEPTHOPTIMIZATION_STATIC_PARA_H
