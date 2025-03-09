#include "RLS.h"

RLS::RLS(int _N, int _M, RLS_TYPE_e _type, float _rho = 0.99, int _p = 10)
{

}

RLS::~RLS()
{
}

Eigen::VectorXf RLS::initRegression(const Eigen::MatrixXf& _H, const Eigen::MatrixXf& _Y)
{
    H_.resize(p_, M_);
    Y_.resize(p_, N_);
    H_.bottomRows(_H.rows()) = _H;
    Y_.bottomRows(_Y.rows()) = _Y;

    if (p_ != 0)
    {
        G_.resize(p_, p_);
        G_.setZero();
        for (size_t i = 0; i < p_; i++)
        {
            G_(i, i) = std::pow(rho_, p_ - i - 1);
        }
    }
    P_ = (Eigen::MatrixXf::Identity(M_, M_) + H_.transpose() * G_ * H_).inverse();
    theta_ = P_.inverse() * H_.transpose() * G_ * Y_;
    is_init_ = true;
    return theta_;
}

Eigen::VectorXf RLS::recursiveRegression(const Eigen::RowVectorXf& _new_h, const Eigen::RowVectorXf& _new_y)
{
    Eigen::MatrixXf Q = (P_ - (P_ * _new_h.transpose() * _new_h * P_) / (1.0f + _new_h * P_ * _new_h.transpose())) / rho_;
    P_ = Q + (std::pow(rho_, p_) * Q * _new_h.transpose() * _new_h * Q) / (1.0f - std::pow(rho_, p_) * _new_h * Q * _new_h.transpose());
    theta_ = theta_ + P_ * _new_h.transpose() * (_new_y - _new_h * theta_)
        - P_ * std::pow(rho_, p_) * _new_h.transpose() * (Y_.row(0) - H_.row(0));

    H_.topRows(p_ - 1) = H_.bottomRows(p_ - 1);
    Y_.topRows(p_ - 1) = Y_.bottomRows(p_ - 1);
    H_.row(p_) = _new_h;
    Y_.row(p_) = _new_y;
    return theta_;
}