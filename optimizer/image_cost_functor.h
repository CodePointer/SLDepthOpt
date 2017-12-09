//
// Created by pointer on 17-12-6.
//

#ifndef DEPTHOPTIMIZATION_IMAGE_COST_FUNCTOR_H
#define DEPTHOPTIMIZATION_IMAGE_COST_FUNCTOR_H

#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

class ImageCostFunctor {
public:
  ImageCostFunctor(
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & weight,
      int idx_k, double img_k, Eigen::Matrix<double, 3, 1> vec_M,
      Eigen::Matrix<double, 3, 1> vec_D, double epi_A, double epi_B);

  template <class T>
  bool operator()(const T* const depth_vec, T* sResiduals) const {
    // Get x_pro, y_pro
    Eigen::Map<Eigen::Matrix<T, 1, 1> const> const depth_k_vec(depth_vec);
    Eigen::Matrix<T, 3, 1> M = this->vec_M_.cast<T>();
    Eigen::Matrix<T, 3, 1> D = this->vec_D_.cast<T>();
    T depth_k = depth_k_vec(0, 0);
    T x_pro = (M(0)*depth_k + D(0)) / (M(2)*depth_k + D(2));
    T y_pro = T(-this->epi_A_/this->epi_B_) * x_pro + T(1/this->epi_B_);

    // Get img_k_head
    T img_k_head;
    this->pattern_.Evaluate(y_pro, x_pro, &img_k_head);

    // Get weight
    T weight_k;
    this->weight_.Evaluate(y_pro + T(0.5), x_pro + T(0.5), &weight_k);

    // Set residual
    sResiduals[0] = (img_k_head - T(this->img_k_));

    if (ceres::IsNaN(sResiduals[0])) {
      system("PAUSE");
    }

    return true;
  }

  // the pattern and weight
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern_;
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & weight_;

  // Parameters for calculation
  int idx_k_;
  double img_k_;
  Eigen::Matrix<double, 3, 1> vec_M_;
  Eigen::Matrix<double, 3, 1> vec_D_;
  double epi_A_;
  double epi_B_;
};

#endif //DEPTHOPTIMIZATION_IMAGE_COST_FUNCTOR_H
