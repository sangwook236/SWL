#if !defined(__SWL_RND_UTIL_TEST__IMU_SYSTEM__H_)
#define __SWL_RND_UTIL_TEST__IMU_SYSTEM__H_ 1


#include "swl/rnd_util/DiscreteLinearStochasticSystem.h"
#include <gsl/gsl_blas.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

class DiscreteKalmanFilter;

//--------------------------------------------------------------------------
//

class ImuSystem: public DiscreteLinearStochasticSystem
{
public:
	typedef DiscreteLinearStochasticSystem base_type;

protected:
	ImuSystem(const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: base_type(stateDim, inputDim, outputDim, (size_t)-1, (size_t)-1)
	{}

public:
	bool runStep(DiscreteKalmanFilter &filter, size_t step, const gsl_vector *Bu, const gsl_vector *Du, const gsl_vector *actualMeasurement, double &prioriEstimate, double &posterioriEstimate) const;

public:
	virtual const gsl_matrix * getInputMatrix() const = 0;
};

//--------------------------------------------------------------------------
//

class AccelSystem: public ImuSystem
{
public:
	typedef ImuSystem base_type;

public:
	static const size_t stateDim = 3;
	static const size_t inputDim = 1;
	static const size_t outputDim = 1;

public:
	AccelSystem(const double Ts, const double beta, const double Qv, const double Qa, const double Qb, const double Ra);
	~AccelSystem();

private:
	AccelSystem(const AccelSystem &rhs);
	AccelSystem & operator=(const AccelSystem &rhs);

public:
	// the stochastic differential equation: x(k+1) = Phi(k) * x(k) + Bd(k) * u(k) + W(k) * w(k)
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const  {  return Phi_;  }  // Phi(k) = exp(A(k) * Ts)
	/*virtual*/ gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const  // Bu(k) = Bd(k) * u(k)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const  {  return W_;  }  // W(k)

	// the observation equation: y(k) = Cd(k) * x(k) + Dd(k) * u(k)
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const  {  return C_;  }  // Cd(k) (C == Cd)
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const  // Du(k) = D(k) * u(k) (D == Dd)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const  {  return V_;  }  // V(k)

	// noise covariance matrices
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd = W * Q * W^T, but not Q
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return Rd_;  }  // Rd = V * R * V^T, but not R

	//
	/*virtual*/ const gsl_matrix * getInputMatrix() const  {  return Bd_;  }

private:
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

class GyroSystem: public ImuSystem
{
public:
	typedef ImuSystem base_type;

public:
	static const size_t stateDim = 2;
	static const size_t inputDim = 1;
	static const size_t outputDim = 1;

public:
	GyroSystem(const double Ts, const double beta, const double Qw, const double Qb, const double Rg);
	~GyroSystem();

private:
	GyroSystem(const GyroSystem &rhs);
	GyroSystem & operator=(const GyroSystem &rhs);

public:
	// the stochastic differential equation: x(k+1) = Phi(k) * x(k) + Bd(k) * u(k) + W(k) * w(k)
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const  {  return Phi_;  }  // Phi(k) = exp(A(k) * Ts)
	/*virtual*/ gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const  // Bu(k) = Bd(k) * u(k)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const  {  return W_;  }  // W(k)

	// the observation equation: y(k) = Cd(k) * x(k) + Dd(k) * u(k)
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const  {  return C_;  }  // Cd(k) (C == Cd)
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const  // Du(k) = D(k) * u(k) (D == Dd)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const  {  return V_;  }  // V(k)

	// noise covariance matrices
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd = W * Q * W^T, but not Q
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return Rd_;  }  // Rd = V * R * V^T, but not R

	//
	/*virtual*/ const gsl_matrix * getInputMatrix() const
	{  throw std::runtime_error("this function doesn't have to be called");  }

private:
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


#endif  // __SWL_RND_UTIL_TEST__IMU_SYSTEM__H_
