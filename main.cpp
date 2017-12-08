#include <iostream>
#include <static_para.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "image_optimizer.h"

ImgMatrix LoadTxtFileToMatrix(std::string file_name, int kHeight, int kWidth) {
  ImgMatrix result(kHeight, kWidth);
  std::fstream file(file_name, std::ios::in);
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      double tmp;
      file >> tmp;
      result(h, w) = tmp;
    }
  }
  file.close();
  return result;
}

ImgMatrix LoadPngFileToMatrix(std::string file_name, int kHeight, int kWidth) {
  ImgMatrix result(kHeight, kWidth);
  cv::Mat img;
  img = cv::imread(file_name);
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      uchar tmp;
      tmp = img.at<uchar>(h, w);
      result(h, w) = (double)tmp;
    }
  }
  return result;
}

void SaveMatrxToTxtFile(std::string file_name, ImgMatrix img_mat,
                        int kHeight, int kWidth) {
  std::fstream file(file_name, std::ios::out);
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      file << img_mat(h, w) << " ";
    }
    file << std::endl;
  }
  file.close();
}

int main() {
  // Load: pattern, grid(from pattern.txt, grid.txt)
  double alpha = 1.0;
  ImgMatrix pattern_img = LoadTxtFileToMatrix("pattern.txt",
                                              kProHeight, kProWidth);
  ImgMatrix weight_img = LoadTxtFileToMatrix("weight.txt",
                                             kProHeight, kProWidth);
  ImgMatrix img_obs = LoadPngFileToMatrix("img.png",
                                          kCamHeight, kCamWidth);
  ImgMatrix depth_mat = LoadTxtFileToMatrix("depth.txt",
                                            kCamHeight, kCamWidth);
  ImgMatrix mask_mat = LoadTxtFileToMatrix("mask.txt",
                                           kCamHeight, kCamWidth);
  ImgMatrix epi_A_mat = LoadTxtFileToMatrix("EpiLineA.txt",
                                            kCamHeight, kCamWidth);
  ImgMatrix epi_B_mat = LoadTxtFileToMatrix("EpiLineB.txt",
                                            kCamHeight, kCamWidth);
  ImgMatrix mat_M = LoadTxtFileToMatrix("Mat_M.txt",
                                        3, kCamHeight*kCamWidth);
  ImgMatrix mat_D = LoadTxtFileToMatrix("Mat_D.txt",
                                        3, kCamHeight*kCamWidth);

  // Optimization
  ImageOptimizer opt(depth_mat, pattern_img, weight_img, alpha,
                     img_obs, mask_mat, epi_A_mat, epi_B_mat, mat_M, mat_D);
  ImgMatrix final_depth = opt.Run();
  SaveMatrxToTxtFile("output_depth.txt", final_depth, kCamHeight, kCamWidth);

  return 0;
}