#if !defined(__SWL_GPS_AIDED_IMU_FILTER_APP__GPS_AIDED_IMU_SYSTEM__H_)
#define __SWL_GPS_AIDED_IMU_FILTER_APP__GPS_AIDED_IMU_SYSTEM__H_ 1


#include "DataDefinition.h"
#include "swl/rnd_util/DiscreteNonlinearStochasticSystem.h"
#include <gsl/gsl_blas.h>


namespace swl {

class GpsAidedImuSystem: public DiscreteNonlinearStochasticSystem
{
public:
	typedef DiscreteNonlinearStochasticSystem base_type;

public:
	GpsAidedImuSystem(const double Ts, const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim, const ImuData::Accel &initial_gravity, const ImuData::Gyro &initial_angular_velocity_of_the_earth);
	~GpsAidedImuSystem();

private:
	GpsAidedImuSystem(const GpsAidedImuSystem &rhs);
	GpsAidedImuSystem & operator=(const GpsAidedImuSystem &rhs);

public:
	// the stochastic differential equation: f = f(k, x(k), u(k), v(k))
	/*virtual*/ gsl_vector * evaluatePlantEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const;
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const  // Phi(k) = exp(A(k) * Ts) where A(k) = df(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const  // Bu(k) = Bd(k) * u(k)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const  {  return W_;  }  // W(k) = df(k, x(k), u(k), 0)/dw

	// the observation equation: h = h(k, x(k), u(k), v(k))
	/*virtual*/ gsl_vector * evaluateMeasurementEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const;
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const  // Cd(k) = dh(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const  // Du(k) = D(k) * u(k) (D == Dd)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const  {  return V_;  }  // V(k) = dh(k, x(k), u(k), 0)/dv

	// noise covariance matrices
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  // Qd = W * Q * W^T, but not Q
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  // Rd = V * R * V^T, but not R
	{  throw std::runtime_error("this function doesn't have to be called");  }

	void setImuMeasurement(const ImuData::Accel &measuredAccel, const ImuData::Gyro &measuredAngularVel)
	{
		measuredAccel_ = measuredAccel;
		measuredAngularVel_ = measuredAngularVel;
	}

private:
	const double Ts_;

	// evalution of the plant equation: f = f(k, x(k), u(k), 0)
	gsl_vector *f_eval_;
	// evalution of the measurement equation: h = h(k, x(k), u(k), 0)
	gsl_vector *h_eval_;

	// the initial gravity
	const ImuData::Accel &initial_gravity_;
	// the initial angular velocity of the Earth
	const ImuData::Gyro &initial_angular_velocity_of_the_earth_;

	// the time-delayed (by N samples due to sensor latency) 3D navigation-frame position vector of the vehicle
	gsl_vector *p_k_N_;
	// the time-delayed (by N samples due to sensor latency) 3D navigation-frame velocity vector of the vehicle
	gsl_vector *v_k_N_;
	// the time-delayed (by N samples due to sensor latency) 3D body-frame angular velocity vector of the vehicle
	gsl_vector *w_k_N_;

	// the location of the GPS antenna in the body frame relative to the IMU location
	gsl_vector *r_GPS_;

	ImuData::Accel measuredAccel_;  //  expressed in body coordinates
	ImuData::Gyro measuredAngularVel_;  //  expressed in body coordinates
};

}  // namespace swl


#endif  // __SWL_GPS_AIDED_IMU_FILTER_APP__GPS_AIDED_IMU_SYSTEM__H_
