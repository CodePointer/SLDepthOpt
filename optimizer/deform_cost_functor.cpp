//
// Created by pointer on 17-12-12.
//

#include "deform_cost_functor.h"

DeformCostFunctor::DeformCostFunctor(
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> &pattern, double img_k,
    Eigen::Matrix<double, 3, 1> vec_M, Eigen::Matrix<double, 3, 1> vec_D,
    double epi_A, double epi_B, Eigen::Matrix<double, 2, 2> range,
    Eigen::Matrix<double, 2, 1> pos_k) : pattern_(pattern) {
  this->img_k_ = img_k;
  this->vec_M_ = vec_M;
  this->vec_D_ = vec_D;
  this->epi_A_ = epi_A;
  this->epi_B_ = epi_B;
  this->range_ = range;
  this->pos_k_ = pos_k;
}
