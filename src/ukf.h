#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct UKF {
  bool is_initialized_;
  bool use_laser_;          // if this is false, laser measurements will be ignored (except for init)
  bool use_radar_;          // if this is false, radar measurements will be ignored (except for init)
  const int n_x_;           // State dimension
  const int n_aug_;         // Augmented state dimension
  const int n_sigma_points_;// Augmented state dimension
  const double lambda_;     // Sigma point spreading parameter
  VectorXd weights_;        // Weights of sigma points
  VectorXd x_;              // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  MatrixXd P_;              // state covariance matrix
  MatrixXd Xsig_pred_;      // predicted sigma points matrix
  long previous_timestamp_; // previous timestamp
  const double std_a_;      // Process noise standard deviation longitudinal acceleration in m/s^2
  const double std_yawdd_;  // Process noise standard deviation yaw acceleration in rad/s^2
  const double std_laspx_;  // Laser measurement noise standard deviation position1 in m
  const double std_laspy_;  // Laser measurement noise standard deviation position2 in m
  const double std_radr_;   // Radar measurement noise standard deviation radius in m
  const double std_radphi_; // Radar measurement noise standard deviation angle in rad
  const double std_radrd_ ; // Radar measurement noise standard deviation radius change in m/s

  UKF(double std_a, double std_yawdd);

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
