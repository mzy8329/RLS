#ifndef _RLS_H_
#define _RLS_H_

#include <eigen3/Eigen/Dense>
#include <vector>

typedef enum
{
    RLS,
    W_RLS,
    EF_RLS,
    WEF_RLS
} RLS_TYPE_e;


/**
 * @brief Rescursive Least Square regression
 * @param _N number of output
 * @param _M number of theta
 * @param _type LS_TYPE_e
 * @param _rho forgetting factor (default 0.99)
 * @param _p length of time windows (default 1)
 */
class RLS
{
public:
    RLS(int _N, int _M, RLS_TYPE_e _type, float _rho = 1.0, int _p = 0);
    ~RLS();

    /**
     * @brief LS regression
     * @param _H T*M matrix
     * @param _Y T*N matrix
     * @return N*1 vector
     */
    Eigen::VectorXf initRegression(const Eigen::MatrixXf& _H, const Eigen::MatrixXf& _Y);
    Eigen::VectorXf recursiveRegression(const Eigen::RowVectorXf& _new_h, const Eigen::RowVectorXf& _new_y);

private:
    bool is_init_ = false;
    int N_;
    int M_;
    int p_;
    float rho_;
    Eigen::MatrixXf H_;
    Eigen::MatrixXf Y_;
    Eigen::MatrixXf P_;
    Eigen::MatrixXf G_;
    Eigen::MatrixXf theta_;

};


#endif // _RLS_H_