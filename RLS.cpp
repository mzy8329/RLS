#include "RLS.h"
#include <iostream>


RLS::RLS(int _M, RLS_TYPE_e _type, float _rho, int _p)
    : M_(_M), type_(_type), rho_(_rho), p_(_p) {
    H_.resize(p_, M_);
    Y_.resize(p_, 1);
    H_.setZero();
    Y_.setZero();
    rho_p_ = std::pow(rho_, p_);
    P_ = Eigen::MatrixXd::Identity(M_, M_) * 1024;
}

RLS::~RLS() {}

Eigen::VectorXd RLS::initRegression(const Eigen::MatrixXd& _H, const Eigen::MatrixXd& _Y)
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

Eigen::VectorXd RLS::recursiveRegression(const Eigen::RowVectorXd& _new_h, const Eigen::RowVectorXd& _new_y)
{
    if (!is_init_)
    {
        return initRegression(_new_h, _new_y);
    }

    Eigen::MatrixXd Q = (P_ - (P_ * _new_h.transpose() * _new_h * P_) / (rho_ + _new_h * P_ * _new_h.transpose())) / rho_;
    P_ = Q + (rho_p_ * Q * H_.row(0).transpose() * H_.row(0) * Q) / (1.0f - rho_p_ * H_.row(0) * Q * H_.row(0).transpose());
    theta_ = theta_ + P_ * _new_h.transpose() * (_new_y - _new_h * theta_)
        - P_ * rho_p_ * H_.row(0).transpose() * (Y_.row(0) - H_.row(0) * theta_);

    H_.topRows(p_ - 1) = H_.bottomRows(p_ - 1);
    Y_.topRows(p_ - 1) = Y_.bottomRows(p_ - 1);
    H_.row(p_ - 1) = _new_h;
    Y_.row(p_ - 1) = _new_y;
    return theta_;
}