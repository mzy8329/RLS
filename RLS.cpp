#include "RLS.h"

RLS::RLS(int _N, int _M, RLS_TYPE_e _type, float _rho = 0.99, int _p = 10)
{

}

RLS::~RLS()
{
}

Eigen::VectorXf RLS::initRegression(const Eigen::MatrixXf& _H, const Eigen::MatrixXf& _Y)
{
    H_ = _H;
    Y_ = _Y;

    if (p_ != 0)
    {
        G_.resize(p_, p_);
        G_.setZero();
        for (size_t i = 0; i < p_; i++)
        {
            G_(i, i) = std::pow(rho_, p_ - i - 1);
        }
    }
    P_ = (H_.transpose() * G_ * H_).inverse();

}

Eigen::VectorXf RLS::recursiveRegression(const Eigen::MatrixXf& _new_h, const Eigen::MatrixXf& _new_y)
{

}