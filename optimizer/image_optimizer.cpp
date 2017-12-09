//
// Created by pointer on 17-12-6.
//

#include "image_optimizer.h"
#include "regular_cost_functor.h"

ImageOptimizer::ImageOptimizer(
    ImgMatrix depth_mat, ImgMatrix & pattern_img, ImgMatrix & weight_img,
    double alpha, ImgMatrix img_obs, ImgMatrix img_mask,
    ImgMatrix epi_mat_A, ImgMatrix epi_mat_B,
    ImgMatrix mat_M, ImgMatrix mat_D) : pattern_img_(pattern_img),
                                        weight_img_(weight_img) {
  this->problem_ = nullptr;
  this->depth_mat_ = depth_mat;
  this->img_obs_ = img_obs;
  this->img_mask_ = img_mask;
  this->epi_mat_A_ = epi_mat_A;
  this->epi_mat_B_ = epi_mat_B;
  this->mat_M_ = mat_M;
  this->mat_D_ = mat_D;

  this->alpha_ = alpha;
  ceres::Grid2D<double, 1> pat_grid(this->pattern_img_.data(),
                                    0, kProHeight, 0, kProWidth);
  this->pattern_ = new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(pat_grid);
  ceres::Grid2D<double, 1> wet_grid(this->weight_img_.data(),
                                    0, kProHeight, 0, kProWidth);
  this->weight_ = new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(wet_grid);
}

ImageOptimizer::~ImageOptimizer() {
  if (this->pattern_ != nullptr) {
    delete this->pattern_;
    this->pattern_ = nullptr;
  }
  if (this->weight_ != nullptr) {
    delete this->weight_;
    this->weight_ = nullptr;
  }
}

ImgMatrix ImageOptimizer::Run() {
  if (this->problem_ != nullptr) {
    delete this->problem_;
    this->problem_ = nullptr;
  }
  this->problem_ = new ceres::Problem;

  int block_num = 0;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (this->img_mask_(h, w) <= 0.5) {
        continue;
      }
      this->AddDataResidualBlock(h, w);
      if ((h >= 1) && (h < kCamHeight - 1) && (w >= 1) && (w < kCamWidth - 1)) {
        this->AddRegularResidualBlock(h, w);
      }
      block_num++;
    }
  }
  this->optimize();

  return this->depth_mat_;
}

void ImageOptimizer::AddDataResidualBlock(int h, int w) {
  int cvec_idx = h*kCamWidth + w;
  double img_k = this->img_obs_(h, w);
  Eigen::Matrix<double, 3, 1> vec_M, vec_D;
  vec_M = this->mat_M_.block<3, 1>(0, cvec_idx);
  vec_D = this->mat_D_.block<3, 1>(0, cvec_idx);
  double epi_A, epi_B;
  epi_A = this->epi_mat_A_(h, w);
  epi_B = this->epi_mat_B_(h, w);

  ceres::CostFunction * cost_fun =
      new ceres::AutoDiffCostFunction<ImageCostFunctor, 1, 1>(
          new ImageCostFunctor(*pattern_, *weight_, cvec_idx, img_k, vec_M,
                               vec_D, epi_A, epi_B));
  this->problem_->AddResidualBlock(cost_fun, NULL,
                                   &this->depth_mat_.data()[cvec_idx]);
}

void ImageOptimizer::AddRegularResidualBlock(int h, int w) {
  int idx_k = h * kCamWidth + w;
  int idx_up = (h - 1) * kCamWidth + w;
  int idx_lt = h * kCamWidth + w - 1;
  int idx_rt = h * kCamWidth + w + 1;
  int idx_dn = (h + 1) * kCamWidth + w;

  ceres::CostFunction * cost_fun =
      new ceres::AutoDiffCostFunction<RegularCostFunctor, 1, 1, 1, 1, 1, 1>(
          new RegularCostFunctor(this->alpha_));
  this->problem_->AddResidualBlock(cost_fun, NULL,
                                   &this->depth_mat_.data()[idx_k],
                                   &this->depth_mat_.data()[idx_up],
                                   &this->depth_mat_.data()[idx_lt],
                                   &this->depth_mat_.data()[idx_rt],
                                   &this->depth_mat_.data()[idx_dn]);
}

void ImageOptimizer::optimize() {
  ceres::Solver::Options options;
  options.gradient_tolerance = 1e-10;
  options.function_tolerance = 1e-10;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;

  ceres::Solve(options, this->problem_, &summary);
  std::fstream file;
  file.open("Ceres_Report.txt", std::ios::out);
  file << summary.FullReport() << std::endl;
  // file << this->depth_vec_ << std::endl;
  file.close();

  std::cout << summary.FullReport() << std::endl;
}