//
// Created by pointer on 17-12-6.
//

#ifndef DEPTHOPTIMIZATION_IMAGE_OPTIMIZER_H
#define DEPTHOPTIMIZATION_IMAGE_OPTIMIZER_H

#include <opencv2/core.hpp>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <static_para.h>
#include <image_cost_functor.h>
#include <fstream>

class ImageOptimizer {
public:
  ceres::Problem * problem_;
  ImgMatrix depth_mat_;

  ImgMatrix & pattern_img_;
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> * pattern_;
  ImgMatrix & weight_img_;
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> * weight_;
  double alpha_;
  ImgMatrix img_obs_;
  ImgMatrix img_mask_;
  ImgMatrix epi_mat_A_;
  ImgMatrix epi_mat_B_;
  ImgMatrix mat_M_;
  ImgMatrix mat_D_;

  ImageOptimizer(
      ImgMatrix depth_mat, ImgMatrix & pattern_img, ImgMatrix & weight_img,
      double alpha, ImgMatrix img_obs, ImgMatrix img_mask,
      ImgMatrix epi_mat_A, ImgMatrix epi_mat_B,
      ImgMatrix mat_M, ImgMatrix mat_D);

  ~ImageOptimizer();

  void AddDataResidualBlock(int h, int w);

  void AddRegularResidualBlock(int h, int w);

  void optimize();

  ImgMatrix Run();
};


#endif //DEPTHOPTIMIZATION_IMAGE_OPTIMIZER_H
