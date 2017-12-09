//
// Created by pointer on 17-12-6.
//

#include "image_cost_functor.h"

ImageCostFunctor::ImageCostFunctor(
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & weight,
    int idx_k, double img_k, Eigen::Matrix<double, 3, 1> vec_M,
    Eigen::Matrix<double, 3, 1> vec_D, double epi_A, double epi_B)
    : pattern_(pattern), weight_(weight) {
  this->idx_k_ = idx_k;
  this->img_k_ = img_k;
  this->vec_M_ = vec_M;
  this->vec_D_ = vec_D;
  this->epi_A_ = epi_A;
  this->epi_B_ = epi_B;
}

//template <class T>
//bool ImageCostFunctor::operator()(const T* const depth_vec, T* sResiduals) const {
//  // Get x_pro, y_pro
//  Eigen::Map<Eigen::Matrix<T, 1, 1> const> const depth_k_vec(depth_vec);
//  Eigen::Matrix<T, 3, 1> M = this->vec_M_.cast<T>();
//  Eigen::Matrix<T, 3, 1> D = this->vec_D_.cast<T>();
//  T depth_k = depth_k_vec(0, 0);
//  T x_pro = (M(1)*depth_k + D(1)) / (M(3)*depth_k + D(3));
//  T y_pro = T(-this->epi_A_/this->epi_B_) * x_pro + T(1/this->epi_B_);
//
//  // Get img_k_head
//  T img_k_head;
//  this->pattern_.Evaluate(y_pro, x_pro, &img_k_head);
//
//  // Get weight
//  T weight_k;
//  this->weight_.Evaluate(y_pro + T(0.5), x_pro + T(0.5), &weight_k);
//
//  // Set residual
//  sResiduals[0] = weight_k * (img_k_head - T(this->img_k_));
//
//  return true;
//}