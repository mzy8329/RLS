#ifndef _RLS_H_
#define _RLS_H_

#include <eigen3/Eigen/Dense>
#include <vector>

typedef enum
{
    W_RLS = 0,
    EF_RLS,
    WEF_RLS
} RLS_TYPE_e;


/**
 * @brief Rescursive Least Square regression
 * @param _M number of theta
 * @param _type LS_TYPE_e
 * @param _rho forgetting factor (default 0.99)
 * @param _p length of time windows (default 1)
 */
class RLS
{
public:
    RLS(int _M, RLS_TYPE_e _type, float _rho = 1.0, int _p = 0);
    ~RLS();

    /**
     * @brief LS regression
     * @param _H T*M matrix
     * @param _Y T*N matrix
     * @return N*1 vector
     */
    Eigen::VectorXd initRegression(const Eigen::MatrixXd& _H, const Eigen::MatrixXd& _Y);
    Eigen::VectorXd recursiveRegression(const Eigen::RowVectorXd& _new_h, const Eigen::RowVectorXd& _new_y);

public:
    bool is_init_ = false;

private:
    int M_;
    RLS_TYPE_e type_;
    int p_;
    float rho_;
    float rho_p_;
    Eigen::MatrixXd H_;
    Eigen::MatrixXd Y_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd G_;
    Eigen::MatrixXd theta_;

};

#endif // _RLS_H_