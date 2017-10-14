#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace { // anon namespace

// angle normalization
void normalize_angle(VectorXd& v, int i) {
    const double psi = v(i);
    v(i) = atan2(sin(psi), cos(psi));
}

} // anon namespace

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF(double std_a, double std_yawdd) :
    is_initialized_(false),
    use_laser_(true),
    use_radar_(true),
    n_x_(5),
    n_aug_(n_x_ + 2),
    n_sigma_points_(2 * n_aug_ + 1),
    lambda_(3 - n_aug_),
    weights_(n_sigma_points_),
    x_(n_x_),
    P_(n_x_, n_x_),
    Xsig_pred_(n_x_, n_sigma_points_),
    previous_timestamp_(-1),
    std_a_(std_a),          // unit: m/s^2
    std_yawdd_(std_yawdd),  // unit: rad/s^2
    std_laspx_(0.15),
    std_laspy_(0.15),
    std_radr_(0.3),
    std_radphi_(0.03),
    std_radrd_(0.3) {

    // initialize sigma point weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < n_sigma_points_; ++i)
        weights_(i) = 0.5 / (n_aug_ + lambda_);
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
    if (!is_initialized_) {
        double px, py, v, phi, d_phi;
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            const double rho = measurement_pack.raw_measurements_[0];
            phi = measurement_pack.raw_measurements_[1];
            px = rho * cos(phi);
            py = rho * sin(phi);
            v = 5.0;
            d_phi = 0;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            px = measurement_pack.raw_measurements_[0];
            py = measurement_pack.raw_measurements_[1];
            v = 5.0;
            phi = M_PI/2;
            d_phi = 0;
        }

        x_ << px, py, v, phi, d_phi;
        
        P_.fill(0.0);
        for (int i = 0; i < n_x_; ++i)
            P_(i, i) = 1.0;
        
        // timestamp
        previous_timestamp_ = measurement_pack.timestamp_;
        
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }
    
    /*****************************************************************************
     *  Prediction
     ****************************************************************************/
    const double delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;
    Prediction(delta_t);
    
    /*****************************************************************************
     *  Update
     ****************************************************************************/
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR && use_radar_ )
        UpdateRadar(measurement_pack);
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER && use_laser_ )
        UpdateLidar(measurement_pack);
    
    // print states
    // cout << "x_ = " << x_ << endl;
    // cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    //
    // generate augmented sigma points
    //

    // create augmented mean state
    VectorXd x_aug(n_aug_);
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0.0;
    x_aug(n_x_ + 1) = 0.0;

    // create augmented covariance matrix
    MatrixXd P_aug(n_aug_, n_aug_);
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
    const MatrixXd L = P_aug.llt().matrixL();

    // create augmented sigma points from augmented mean state and covariance matrix
    MatrixXd Xsig_aug(n_aug_, n_sigma_points_);
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    //
    // predict augmented sigma points
    //
    for (int i = 0; i < n_sigma_points_; ++i) {
        const double p_x = Xsig_aug(0, i);
        const double p_y = Xsig_aug(1, i);
        const double v = Xsig_aug(2, i);
        const double yaw = Xsig_aug(3, i);
        const double yawd = Xsig_aug(4, i);
        const double nu_a = Xsig_aug(5, i);
        const double nu_yawdd = Xsig_aug(6, i);
    
        // predicted state values
        double px_p, py_p, v_p, yaw_p, yawd_p;

        // avoid division by zero
        if (std::abs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }
        v_p = v;
        yaw_p = yaw + yawd*delta_t;
        yawd_p = yawd;

        // add noise
        px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p += nu_a * delta_t;
        yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p += + nu_yawdd * delta_t;
        
        // write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

    //
    // predict state
    //
    x_ = Xsig_pred_ * weights_;

    //
    // predict state covariance matrix
    //
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_points_; ++i) {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        normalize_angle(x_diff, 3);
        P_ += weights_(i) * x_diff * x_diff.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    assert(meas_package.sensor_type_ == MeasurementPackage::LASER);

    const int n_z = 2;
    MatrixXd Zsig(n_z, n_sigma_points_);

    // transform sigma points into measurement space
    for (int i = 0; i < n_sigma_points_; ++i) {
        const double p_x = Xsig_pred_(0, i);
        const double p_y = Xsig_pred_(1, i);
        
        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    // mean predicted measurement
    const VectorXd z_pred = Zsig * weights_;

    // measurement covariance matrix S
    MatrixXd S(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < n_sigma_points_; ++i) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S += weights_(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    MatrixXd R(n_z, n_z);
    R << std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;
    S += R;

    // incoming radar measurement
    const VectorXd z(meas_package.raw_measurements_);
    assert(z.size() == n_z);

    // create matrix for cross correlation Tc
    MatrixXd Tc(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < n_sigma_points_; ++i) {
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    
    // Kalman gain K;
    const MatrixXd K = Tc * S.inverse();
    
    // residual
    VectorXd z_diff = z - z_pred;
    
    // update state mean and covariance matrix
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();

    // NIS (Normzlied Innovation Squared) 
    const double nis = z_diff.transpose() * S.inverse() * z_diff;
    static int n_outside = 0, n_inside = 0;
    const double chi_square_5 = 5.991;
    if (nis > chi_square_5) ++n_outside;
    else ++n_inside;

    const double ratioInside = double(n_inside) / (n_outside + n_inside) * 100;
    cout << "Lidar NIS: " << ratioInside << "%" << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    assert(meas_package.sensor_type_ == MeasurementPackage::RADAR);

    const int n_z = 3;
    MatrixXd Zsig(n_z, n_sigma_points_);

    // transform sigma points into measurement space
    for (int i = 0; i < n_sigma_points_; ++i) {
        const double p_x = Xsig_pred_(0, i);
        const double p_y = Xsig_pred_(1, i);
        const double v  = Xsig_pred_(2, i);
        const double yaw = Xsig_pred_(3, i);
        const double v1 = cos(yaw) * v;
        const double v2 = sin(yaw) * v;
        const double r = sqrt(p_x * p_x + p_y * p_y);
        
        // measurement model
        Zsig(0, i) = r;                         // r
        Zsig(1, i) = atan2(p_y, p_x);           // phi
        Zsig(2, i) = (p_x * v1 + p_y * v2) / r; // r_dot
    }

    // mean predicted measurement
    const VectorXd z_pred = Zsig * weights_;

    // measurement covariance matrix S
    MatrixXd S(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < n_sigma_points_; ++i) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        normalize_angle(z_diff, 1);
        S += weights_(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    MatrixXd R(n_z, n_z);
    R << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0,std_radrd_ * std_radrd_;
    S += R;

    // incoming radar measurement
    const VectorXd z(meas_package.raw_measurements_);
    assert(z.size() == n_z);

    // create matrix for cross correlation Tc
    MatrixXd Tc(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < n_sigma_points_; ++i) {
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        normalize_angle(z_diff, 1);
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        normalize_angle(x_diff, 3);

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    
    // Kalman gain K;
    const MatrixXd K = Tc * S.inverse();
    
    // residual
    VectorXd z_diff = z - z_pred;
    normalize_angle(z_diff, 1);
    
    // update state mean and covariance matrix
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();

    // NIS (Normzlied Innovation Squared) 
    const double nis = z_diff.transpose() * S.inverse() * z_diff;
    static int n_outside = 0, n_inside = 0;
    const double chi_square_5 = 7.815;
    if (nis > chi_square_5) ++n_outside;
    else ++n_inside;

    const double ratioInside = double(n_inside) / (n_outside + n_inside) * 100;
    cout << "Radar NIS: " << ratioInside << "%" << endl;
}
