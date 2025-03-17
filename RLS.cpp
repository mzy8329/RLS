#include "RLS.h"
#include <iostream>


RLS::RLS(int _M, RLS_TYPE_e _type, float _rho, int _p)
    : M_(_M), type_(_type), rho_(_rho), p_(_p) {
    H_.resize(p_, M_);
    Y_.resize(p_, 1);
    H_.setZero();
    Y_.setZero();
    rho_p_ = std::pow(rho_, p_);
    P_ = Eigen::MatrixXf::Identity(M_, M_) * 1024;
}

RLS::~RLS() {}

Eigen::VectorXf RLS::initRegression(const Eigen::MatrixXf& _H, const Eigen::MatrixXf& _Y)
{
    H_.bottomRows(_H.rows()) = _H;
    Y_.bottomRows(_Y.rows()) = _Y;

    H_.leftCols(M_) = _H.replicate(p_, 1);
    Y_.leftCols(1) = _Y.replicate(p_, 1);

    if (p_ != 0)
    {
        G_.resize(p_, p_);
        G_.setZero();
        for (size_t i = 0; i < p_; i++)
        {
            G_(i, i) = std::pow(rho_, p_ - i - 1);
        }
    }
    theta_ = P_.inverse() * H_.transpose() * G_ * Y_;
    is_init_ = true;
    return theta_;
}

Eigen::VectorXf RLS::recursiveRegression(const Eigen::RowVectorXf& _new_h, const Eigen::RowVectorXf& _new_y)
{
    if (!is_init_)
    {
        return initRegression(_new_h, _new_y);
    }

    Eigen::MatrixXf Q = (P_ - (P_ * _new_h.transpose() * _new_h * P_) / (rho_ + _new_h * P_ * _new_h.transpose())) / rho_;
    P_ = Q + (rho_p_ * Q * H_.row(0).transpose() * H_.row(0) * Q) / (1.0f - rho_p_ * H_.row(0) * Q * H_.row(0).transpose());
    theta_ = theta_ + P_ * _new_h.transpose() * (_new_y - _new_h * theta_)
        - P_ * rho_p_ * H_.row(0).transpose() * (Y_.row(0) - H_.row(0) * theta_);

    H_.topRows(p_ - 1) = H_.bottomRows(p_ - 1);
    Y_.topRows(p_ - 1) = Y_.bottomRows(p_ - 1);
    H_.row(p_ - 1) = _new_h;
    Y_.row(p_ - 1) = _new_y;
    return theta_;
}