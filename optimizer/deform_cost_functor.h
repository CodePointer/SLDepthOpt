//
// Created by pointer on 17-12-12.
//

#ifndef DEPTHOPTIMIZATION_DEFORM_COST_FUNCTOR_H
#define DEPTHOPTIMIZATION_DEFORM_COST_FUNCTOR_H


#include <ceres/cubic_interpolation.h>

class DeformCostFunctor {
public:
  DeformCostFunctor(
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
      double img_k, Eigen::Matrix<double, 3, 1> vec_M,
      Eigen::Matrix<double, 3, 1> vec_D, double epi_A, double epi_B,
      Eigen::Matrix<double, 2, 2> range, Eigen::Matrix<double, 2, 1> pos_k);

  template <class T>
  bool operator()(const T* const d_ul, const T* const d_ur,
                  const T* const d_dl, const T* const d_dr, T* residuals) const {
    // get d_k
    double kBlockHeight = this->range_(1, 0) - this->range_(0, 0);
    double kBlockWidth = this->range_(1, 1) - this->range_(0, 1);
    double dis_lf = (this->pos_k_(1) - this->range_(0, 1)) / kBlockWidth;
    double dis_rt = (this->range_(1, 1) - this->pos_k_(1)) / kBlockWidth;
    double dis_up = (this->pos_k_(0) - this->range_(0, 0)) / kBlockHeight;
    double dis_dn = (this->range_(1, 0) - this->pos_k_(0)) / kBlockHeight;
    T d_up = d_ul[0] * T(1 - dis_lf) + d_ur[0] * T(1 - dis_rt);
    T d_dn = d_dl[0] * T(1 - dis_lf) + d_dr[0] * T(1 - dis_rt);
    T d_lf = d_ul[0] * T(1 - dis_up) + d_dl[0] * T(1 - dis_dn);
    T d_rt = d_ur[0] * T(1 - dis_up) + d_dr[0] * T(1 - dis_dn);
    T d_k = 0.5 * (d_up * (1 - dis_up) + d_dn * (1 - dis_dn))
            + 0.5 * (d_lf * (1 - dis_lf) + d_rt * (1 - dis_rt));
    // Get x_pro, y_pro
    Eigen::Matrix<T, 3, 1> M = this->vec_M_.cast<T>();
    Eigen::Matrix<T, 3, 1> D = this->vec_D_.cast<T>();
    T x_pro = (M(0)*d_k + D(0)) / (M(2)*d_k + D(2));
    T y_pro = T(-this->epi_A_/this->epi_B_) * x_pro + T(1/this->epi_B_);
    // Get img_k_head
    T img_k_head;
    this->pattern_.Evaluate(y_pro, x_pro, &img_k_head);
    residuals[0] = img_k_head - T(this->img_k_);
  }

  // pattern & weight
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern_;
  // ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & weight_;
  // For depth color calculation
  double img_k_;
  Eigen::Matrix<double, 3, 1> vec_M_;
  Eigen::Matrix<double, 3, 1> vec_D_;
  double epi_A_;
  double epi_B_;
  // For interpolation
  Eigen::Matrix<double, 2, 2> range_;
  Eigen::Matrix<double, 2, 1> pos_k_;
};


#endif //DEPTHOPTIMIZATION_DEFORM_COST_FUNCTOR_H
