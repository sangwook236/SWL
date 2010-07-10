#if !defined(__SWL_RND_UTIL_TEST__IMU_KALMAN_FILTER__H_)
#define __SWL_RND_UTIL_TEST__IMU_KALMAN_FILTER__H_ 1


#include "swl/rnd_util/KalmanFilter.h"
#include <gsl/gsl_blas.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
//

class ImuKalmanFilter: public KalmanFilter
{
public:
	typedef KalmanFilter base_type;

protected:
	ImuKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: base_type(x0, P0, stateDim, inputDim, outputDim)
	{}

public:
	bool runStep(size_t step, const gsl_vector *Bu, const gsl_vector *Du, const gsl_vector *actualMeasurement, double &prioriEstimate, double &posterioriEstimate);

public:
	virtual const gsl_matrix * getInputMatrix() const = 0;
};

//--------------------------------------------------------------------------
//

class AccelKalmanFilter: public ImuKalmanFilter
{
public:
	typedef ImuKalmanFilter base_type;

public:
	static const size_t stateDim = 3;
	static const size_t inputDim = 1;
	static const size_t outputDim = 1;

public:
	AccelKalmanFilter(const double Ts, const double beta, const double Qv, const double Qa, const double Qb, const double Ra, const gsl_vector *x0, const gsl_matrix *P0);
	~AccelKalmanFilter();

private:
	AccelKalmanFilter(const AccelKalmanFilter &rhs);
	AccelKalmanFilter & operator=(const AccelKalmanFilter &rhs);

public:
	/*virtual*/ const gsl_matrix * getInputMatrix() const  {  return Bd_;  }

private:
	// for continuous Kalman filter
	/*virtual*/ gsl_matrix * doGetSystemMatrix(const double time, const gsl_vector *state) const  // A(t)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	// for discrete Kalman filter
	/*virtual*/ gsl_matrix * doGetStateTransitionMatrix(const size_t step, const gsl_vector *state) const  {  return Phi_;  }  // Phi(k) = exp(A(k) * Ts)
	/*virtual*/ gsl_matrix * doGetOutputMatrix(const size_t step, const gsl_vector *state) const  {  return C_;  }  // Cd(k) (C == Cd)

	/*virtual*/ gsl_matrix * doGetProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd = W * Q * W^T, but not Q
	/*virtual*/ gsl_matrix * doGetMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return Rd_;  }  // Rd = V * R * V^T, but not R

protected:
	const double Ts_;  // sampling time
	const double beta_;  // correlation time (time constant)
	const double Qv_;  // variance of a process noise related to a velocity state
	const double Qa_;  // variance of a process noise related to an acceleration state
	const double Qb_;  // variance of a process noise related to an acceleration's bias state
	const double Ra_;  // variance of a measurement noise

	gsl_matrix *Phi_;
	gsl_matrix *C_;  // Cd = C
	gsl_matrix *Qd_;
	gsl_matrix *Rd_;

	// control input: Bu = Bd * u where Bd = integrate(exp(A * t), {t, 0, Ts}) * B or A^-1 * (Ad - I) * B if A is nonsingular
	gsl_matrix *Bd_;
};

//--------------------------------------------------------------------------
//

class GyroKalmanFilter: public ImuKalmanFilter
{
public:
	typedef ImuKalmanFilter base_type;

public:
	static const size_t stateDim = 2;
	static const size_t inputDim = 1;
	static const size_t outputDim = 1;

public:
	GyroKalmanFilter(const double Ts, const double beta, const double Qw, const double Qb, const double Rg, const gsl_vector *x0, const gsl_matrix *P0);
	~GyroKalmanFilter();

private:
	GyroKalmanFilter(const GyroKalmanFilter &rhs);
	GyroKalmanFilter & operator=(const GyroKalmanFilter &rhs);

public:
	/*virtual*/ const gsl_matrix * getInputMatrix() const
	{  throw std::runtime_error("this function doesn't have to be called");  }

private:
	// for continuous Kalman filter
	/*virtual*/ gsl_matrix * doGetSystemMatrix(const double time, const gsl_vector *state) const  // A(t)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	// for discrete Kalman filter
	/*virtual*/ gsl_matrix * doGetStateTransitionMatrix(const size_t step, const gsl_vector *state) const  {  return Phi_;  }  // Phi(k) = exp(A(k) * Ts)
	/*virtual*/ gsl_matrix * doGetOutputMatrix(const size_t step, const gsl_vector *state) const  {  return C_;  }  // Cd(k) (C == Cd)

	/*virtual*/ gsl_matrix * doGetProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd = W * Q * W^T, but not Q
	/*virtual*/ gsl_matrix * doGetMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return Rd_;  }  // Rd = V * R * V^T, but not R

protected:
	const double Ts_;  // sampling time
	const double beta_;  // correlation time (time constant)
	const double Qw_;  // variance of a process noise related to an angular velocity state
	const double Qb_;  // variance of a process noise related to an angular velocity's bias state
	const double Rg_;  // variance of a measurement noise

	gsl_matrix *Phi_;
	gsl_matrix *C_;  // Cd = C
	gsl_matrix *Qd_;
	gsl_matrix *Rd_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL_TEST__IMU_KALMAN_FILTER__H_
