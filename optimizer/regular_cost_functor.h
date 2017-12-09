//
// Created by pointer on 17-12-8.
//

#ifndef DEPTHOPTIMIZATION_REGULAR_COST_FUNCTOR_H
#define DEPTHOPTIMIZATION_REGULAR_COST_FUNCTOR_H


#include "image_cost_functor.h"

class RegularCostFunctor {
public:
  RegularCostFunctor(double alpha) : alpha_(alpha) {};

  template <class T>
  bool operator()(const T* const d_k, const T* const d_up,
                  const T* const d_lf, const T* const d_rt,
                  const T* const d_dn, T* residual) const {
    residual[0] = T(this->alpha_) * (T(4.0) * d_k[0] - d_up[0] - d_dn[0] - d_lf[0] - d_rt[0]);
  }

  double& alpha_;
};


#endif //DEPTHOPTIMIZATION_REGULAR_COST_FUNCTOR_H
