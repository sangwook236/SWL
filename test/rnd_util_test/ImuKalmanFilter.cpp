#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/KalmanFilter.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>
#include <ctime>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

class ImuKalmanFilter: public swl::KalmanFilter
{
public:
	typedef swl::KalmanFilter base_type;

protected:
	ImuKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: base_type(x0, P0, stateDim, inputDim, outputDim)
	{}

public:
	virtual gsl_vector * getControlInput(const size_t step) const = 0;  // Bu(k) = Bd(k) * u(k)
	virtual gsl_vector * getMeasurementInput(const size_t step) const = 0;  // Du(k) = D(k) * u(k) (D == Dd)

	// actual measurement
	virtual gsl_vector * getMeasurement(const size_t step) const = 0;
};

class AccelKalmanFilter: public ImuKalmanFilter
{
public:
	typedef ImuKalmanFilter base_type;

public:
	static const size_t stateDim = 3;
	static const size_t inputDim = 1;
	static const size_t outputDim = 1;

protected:
	AccelKalmanFilter(const double Ts, const double Qv, const double Qa, const double Qb, const double Ra, const gsl_vector *x0, const gsl_matrix *P0)
	: base_type(x0, P0, stateDim, inputDim, outputDim),
	  Ts_(Ts), Qv_(Qv), Qa_(Qa), Qb_(Qb), Ra_(Ra),
	  Phi_(NULL), C_(NULL), Qd_(NULL), Rd_(NULL), Bd_(NULL), Bu_(NULL), Du_(NULL), y_tilde_(NULL)
	{
		// Phi = exp(A * Ts) = [ 1 Ts 0 ; 0 1 0 ; 0 0 1 ]
		Phi_ = gsl_matrix_alloc(stateDim, stateDim);
		gsl_matrix_set_identity(Phi_);
		gsl_matrix_set(Phi_, 0, 1, Ts_);

		// C = [ 0 1 1 ]
		C_ = gsl_matrix_alloc(outputDim, stateDim);
		gsl_matrix_set(C_, 0, 0, 0.0);  gsl_matrix_set(C_, 0, 1, 1.0);  gsl_matrix_set(C_, 0, 2, 1.0);

		// continuous system  -->  discrete system
		// Qd = integrate(Phi(t) * W(t) * Q(t) * W(t)^T * Phi(t)^T, {t, 0, Ts})
		//	Q = [ Qv 0 0 ; 0 Qa 0 ; 0 0 Qb ]
		//	W = I
		//	Qd = [ (Qa*Ts^3)/3 + Qv*Ts, (Qa*Ts^2)/2,     0 ]
		//	     [         (Qa*Ts^2)/2,       Qa*Ts,     0 ]
		//	     [                   0,           0, Qb*Ts ]
		gsl_vector_set_zero(Bu_);
		Qd_ = gsl_matrix_alloc(stateDim, stateDim);
		gsl_matrix_set(Qd_, 0, 0, (Qa_*Ts_*Ts_*Ts_)/3 + Qv_*Ts_);  gsl_matrix_set(Qd_, 0, 1, (Qa_*Ts_*Ts_)/2);  gsl_matrix_set(Qd_, 0, 2, 0);
		gsl_matrix_set(Qd_, 1, 0, (Qa_*Ts_*Ts_)/2);  gsl_matrix_set(Qd_, 1, 1, Qa_*Ts_);  gsl_matrix_set(Qd_, 1, 2, 0);
		gsl_matrix_set(Qd_, 2, 0, 0);  gsl_matrix_set(Qd_, 2, 1, 0);  gsl_matrix_set(Qd_, 2, 2, Qb_*Ts_);

		// Rd = V * R * V^T since V = I
		Rd_ = gsl_matrix_alloc(outputDim, outputDim);
		gsl_matrix_set_all(Rd_, Ra_);

		// input matrix: Bd = integrate(exp(A * t), {t, 0, Ts}) * B
		//	B = [ 1 ; 0 ; 0 ]
		//	integrate(exp(A * t), {t, 0, Ts}) = [ Ts Ts^2/2 0 ; 0 Ts 0 ; 0 0 Ts ]
		//	Bd = [ Ts ; 0 ; 0 ]
		Bd_ = gsl_matrix_alloc(stateDim, inputDim);
		gsl_matrix_set(Bd_, 0, 0, Ts_);  gsl_matrix_set(Bd_, 0, 1, 0.0);  gsl_matrix_set(Bd_, 0, 2, 0.0);

		// control input: Bu = Bd * u where u(t) = g_x + a_Fx
		Bu_ = gsl_vector_alloc(stateDim);
		gsl_vector_set_zero(Bu_);

		// no measurement input
		Du_ = gsl_vector_alloc(outputDim);
		gsl_vector_set_zero(Du_);

		//
		y_tilde_ = gsl_vector_alloc(outputDim);
		gsl_vector_set_zero(y_tilde_);
	}
public:
	~AccelKalmanFilter()
	{
		gsl_matrix_free(Phi_);  Phi_ = NULL;
		gsl_matrix_free(C_);  C_ = NULL;
		gsl_matrix_free(Qd_);  Qd_ = NULL;
		gsl_matrix_free(Rd_);  Rd_ = NULL;

		gsl_matrix_free(Bd_);  Bd_ = NULL;
		gsl_vector_free(Bu_);  Bu_ = NULL;
		gsl_vector_free(Du_);  Du_ = NULL;

		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;
	}

private:
	AccelKalmanFilter(const AccelKalmanFilter &rhs);
	AccelKalmanFilter & operator=(const AccelKalmanFilter &rhs);

public:
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step) const  {  return Du_;  }  // Du(k) = D(k) * u(k) (D == Dd)

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
	gsl_vector *Bu_;
	// measurement input: Du = Dd * u where Dd = D
	gsl_vector *Du_;

	// actual measurement
	gsl_vector *y_tilde_;
};

class XAccelKalmanFilter: public AccelKalmanFilter
{
public:
	typedef AccelKalmanFilter base_type;

public:
	XAccelKalmanFilter(const double Ts, const double Qv, const double Qa, const double Qb, const double Ra, const gsl_vector *x0, const gsl_matrix *P0)
	: base_type(Ts, Qv, Qa, Qb, Ra, x0, P0)
	{}

public:
	/*virtual*/ gsl_vector * getControlInput(const size_t step) const  // Bu(k) = Bd(k) * u(k)
	{
		// FIXME [implement] >>
		assert(false);

		// g_x: the x-component of gravity, a_Fx: the x-component of the acceleration exerted by the robot's input force
		// Bu = Bd * (g_x + a_Fx)
		//gsl_vector_set_zero(Bu_);
		const double g_x = 0.0;
		const double a_Fx = 0.0;
		gsl_vector_set(Bu_, 0, Ts_ * (g_x + a_Fx));

		return Bu_;
	}

	// actual measurement
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		// FIXME [implement] >>
		assert(false);

		// y_tilde = a measured x-axis acceleration
		gsl_vector_set(y_tilde_, 0, 0.0);

		return y_tilde_;
	}
};

class YAccelKalmanFilter: public AccelKalmanFilter
{
public:
	typedef AccelKalmanFilter base_type;

public:
	YAccelKalmanFilter(const double Ts, const double Qv, const double Qa, const double Qb, const double Ra, const gsl_vector *x0, const gsl_matrix *P0)
	: base_type(Ts, Qv, Qa, Qb, Ra, x0, P0)
	{}

public:
	/*virtual*/ gsl_vector * getControlInput(const size_t step) const  // Bu(k) = Bd(k) * u(k)
	{
		// FIXME [implement] >>
		assert(false);

		// g_y: the y-component of gravity, a_Fy: the y-component of the acceleration exerted by the robot's input force
		// Bu = Bd * (g_y + a_Fy)
		//gsl_vector_set_zero(Bu_);
		const double g_y = 0.0;
		const double a_Fy = 0.0;
		gsl_vector_set(Bu_, 0, Ts_ * (g_y + a_Fy));

		return Bu_;
	}

	// actual measurement
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		// FIXME [implement] >>
		assert(false);

		// y_tilde = a measured y-axis acceleration
		gsl_vector_set(y_tilde_, 0, 0.0);

		return y_tilde_;
	}
};

class ZAccelKalmanFilter: public AccelKalmanFilter
{
public:
	typedef AccelKalmanFilter base_type;

public:
	ZAccelKalmanFilter(const double Ts, const double Qv, const double Qa, const double Qb, const double Ra, const gsl_vector *x0, const gsl_matrix *P0)
	: base_type(Ts, Qv, Qa, Qb, Ra, x0, P0)
	{}

public:
	/*virtual*/ gsl_vector * getControlInput(const size_t step) const  // Bu(k) = Bd(k) * u(k)
	{
		// FIXME [implement] >>
		assert(false);

		// g_z: the z-component of gravity, a_Fz: the z-component of the acceleration exerted by the robot's input force
		// Bu = Bd * (g_z + a_Fz)
		//gsl_vector_set_zero(Bu_);
		const double g_z = 0.0;
		const double a_Fz = 0.0;
		gsl_vector_set(Bu_, 0, Ts_ * (g_z + a_Fz));

		return Bu_;
	}

	// actual measurement
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		// FIXME [implement] >>
		assert(false);

		// y_tilde = a measured z-axis acceleration
		gsl_vector_set(y_tilde_, 0, 0.0);

		return y_tilde_;
	}
};

class GyroKalmanFilter: public ImuKalmanFilter
{
public:
	typedef ImuKalmanFilter base_type;

public:
	static const size_t stateDim = 2;
	static const size_t inputDim = 1;
	static const size_t outputDim = 1;

protected:
	GyroKalmanFilter(const double Ts, const double Qw, const double Rg, const gsl_vector *x0, const gsl_matrix *P0)
	: base_type(x0, P0, stateDim, inputDim, outputDim),
	  Ts_(Ts), Qw_(Qw), Rg_(Rg),
	  Phi_(NULL), C_(NULL), Qd_(NULL), Rd_(NULL), Bu_(NULL), Du_(NULL), y_tilde_(NULL)
	{
		// Phi = exp(A * Ts) = [ 1 0 ; 0 1 ]
		Phi_ = gsl_matrix_alloc(stateDim, stateDim);
		gsl_matrix_set_identity(Phi_);

		// C = [ 1 1 ]
		C_ = gsl_matrix_alloc(outputDim, stateDim);
		gsl_matrix_set_identity(C_);
		gsl_matrix_set(C_, 0, 0, 1.0);  gsl_matrix_set(C_, 0, 1, 1.0);

		// continuous system  -->  discrete system
		// Qd = integrate(Phi(t) * W(t) * Q(t) * W(t)^T * Phi(t)^T, {t, 0, Ts})
		//	Q = [ Qw 0 ; 0 0 ]
		//	W = I
		//	Qd = [ Qw*Ts 0 ; 0 0 ]
		Qd_ = gsl_matrix_alloc(stateDim, stateDim);
		gsl_matrix_set_zero(Qd_);
		gsl_matrix_set(Qd_, 0, 0, Qw*Ts);
		// Rd = V * R * V^T where V = I
		Rd_ = gsl_matrix_alloc(outputDim, outputDim);
		gsl_matrix_set_all(Rd_, Rg_);

		// no control input: Bu = Bd * u where u(t) = 1
		//	Bu = Bd * u where Bd = integrate(exp(A * t), {t, 0, Ts}) * B or A^-1 * (Ad - I) * B if A is nonsingular
		//	Ad = Phi = exp(A * Ts)
		//	B = [ 0 ; 0 ]
		Bu_ = gsl_vector_alloc(stateDim);
		gsl_vector_set_zero(Bu_);

		// no measurement input
		Du_ = gsl_vector_alloc(outputDim);
		gsl_vector_set_zero(Du_);

		//
		y_tilde_ = gsl_vector_alloc(outputDim);
		gsl_vector_set_zero(y_tilde_);
	}
public:
	~GyroKalmanFilter()
	{
		gsl_matrix_free(Phi_);  Phi_ = NULL;
		gsl_matrix_free(C_);  C_ = NULL;
		gsl_matrix_free(Qd_);  Qd_ = NULL;
		gsl_matrix_free(Rd_);  Rd_ = NULL;

		gsl_vector_free(Bu_);  Bu_ = NULL;
		gsl_vector_free(Du_);  Du_ = NULL;

		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;
	}

private:
	GyroKalmanFilter(const GyroKalmanFilter &rhs);
	GyroKalmanFilter & operator=(const GyroKalmanFilter &rhs);

public:
	/*virtual*/ gsl_vector * getControlInput(const size_t step) const  {  return Bu_;  }  // Bu(k) = Bd(k) * u(k)
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step) const  {  return Du_;  }  // Du(k) = D(k) * u(k) (D == Dd)

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
	const double Qw_;  // variance of a process noise related to an angular velocity state
	const double Rg_;  // variance of a measurement noise

	gsl_matrix *Phi_;
	gsl_matrix *C_;  // Cd = C
	gsl_matrix *Qd_;
	gsl_matrix *Rd_;

	// control input: Bu = Bd * u where Bd = integrate(exp(A * t), {t, 0, Ts}) * B or A^-1 * (Ad - I) * B if A is nonsingular
	gsl_vector *Bu_;
	// measurement input: Du = Dd * u where Dd = D
	gsl_vector *Du_;

	// actual measurement
	gsl_vector *y_tilde_;
};

class XGyroKalmanFilter: public GyroKalmanFilter
{
public:
	typedef GyroKalmanFilter base_type;

public:
	XGyroKalmanFilter(const double Ts, const double Qw, const double Rg, const gsl_vector *x0, const gsl_matrix *P0)
	: base_type(Ts, Qw, Rg, x0, P0)
	{}

public:
	// actual measurement
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		// FIXME [implement] >>
		assert(false);

		// y_tilde = a measured x-axis angular velocity
		gsl_vector_set(y_tilde_, 0, 0.0);

		return y_tilde_;
	}
};

class YGyroKalmanFilter: public GyroKalmanFilter
{
public:
	typedef GyroKalmanFilter base_type;

public:
	YGyroKalmanFilter(const double Ts, const double Qw, const double Rg, const gsl_vector *x0, const gsl_matrix *P0)
	: base_type(Ts, Qw, Rg, x0, P0)
	{}

public:
	// actual measurement
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		// FIXME [implement] >>
		assert(false);

		// y_tilde = a measured y-axis angular velocity
		gsl_vector_set(y_tilde_, 0, 0.0);

		return y_tilde_;
	}
};

class ZGyroKalmanFilter: public GyroKalmanFilter
{
public:
	typedef GyroKalmanFilter base_type;

public:
	ZGyroKalmanFilter(const double Ts, const double Qw, const double Rg, const gsl_vector *x0, const gsl_matrix *P0)
	: base_type(Ts, Qw, Rg, x0, P0)
	{}

public:
	// actual measurement
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		// FIXME [implement] >>
		assert(false);

		// y_tilde = a measured z-axis angular velocity
		gsl_vector_set(y_tilde_, 0, 0.0);

		return y_tilde_;
	}
};

bool runImuKalmanFilter(size_t step, ImuKalmanFilter &filter, double &prioriEstimate, double &posterioriEstimate)
{
#if 0
	// method #1
	// 1-based time step. 0-th time step is initial
	//size_t step = 0;
	//while (step < Nstep)
	{
		// 0. initial estimates: x(0) & P(0)

		// 1. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const gsl_vector *Bu = filter.getControlInput(step);
		if (!filter.updateTime(step, Bu))
			return false;

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			//const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			prioriEstimate = gsl_vector_get(x_hat, 0);
		}

		// advance time step
		++step;

		// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const gsl_vector *actualMeasurement = filter.getMeasurement(step);
		const gsl_vector *Du = filter.getMeasurementInput(step);
		if (!filter.updateMeasurement(step, actualMeasurement, Du))
			return false;

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			//const gsl_matrix *K = filter.getKalmanGain();
			//const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			posterioriEstimate = gsl_vector_get(x_hat, 0);
		}
	}
#else
	// method #2
	// 0-based time step. 0-th time step is initial
	//size_t step = 0;
	//while (step < Nstep)
	{
		// 0. initial estimates: x-(0) & P-(0)

		// 1. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const gsl_vector *actualMeasurement = filter.getMeasurement(step);
		const gsl_vector *Du = filter.getMeasurementInput(step);
		if (!filter.updateMeasurement(step, actualMeasurement, Du))
			return false;

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			//const gsl_matrix *K = filter.getKalmanGain();
			//const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			posterioriEstimate = gsl_vector_get(x_hat, 0);
		}

		// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const gsl_vector *Bu = filter.getControlInput(step);
		if (!filter.updateTime(step, Bu))
			return false;

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			//const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			prioriEstimate = gsl_vector_get(x_hat, 0);
		}

		// advance time step
		++step;
	}
#endif

	return true;
}

}  // unnamed namespace

void imu_kalman_filter()
{
	gsl_vector *Xa0 = gsl_vector_alloc(AccelKalmanFilter::stateDim);
	gsl_matrix *Pa0 = gsl_matrix_alloc(AccelKalmanFilter::stateDim, AccelKalmanFilter::stateDim);
	gsl_vector_set_zero(Xa0);
	gsl_matrix_set_zero(Pa0);

	const double Ts = 0.1;  // sampling time

	const double Qv = std::sqrt(0.03);  // variance of a process noise related to a velocity state
	const double Qa = 0.03;  // variance of a process noise related to an acceleration state
	const double Qb = 0.01 * 0.01;  // variance of a process noise related to an acceleration's bias state
	const double Ra = 0.01;  // variance of a measurement noise
	XAccelKalmanFilter xAccelFilter(Ts, Qv, Qa, Qb, Ra, Xa0, Pa0);
	YAccelKalmanFilter yAccelFilter(Ts, Qv, Qa, Qb, Ra, Xa0, Pa0);
	ZAccelKalmanFilter zAccelFilter(Ts, Qv, Qa, Qb, Ra, Xa0, Pa0);

	gsl_vector_free(Xa0);  Xa0 = NULL;
	gsl_matrix_free(Pa0);  Pa0 = NULL;

	//
	gsl_vector *Xg0 = gsl_vector_alloc(GyroKalmanFilter::stateDim);
	gsl_matrix *Pg0 = gsl_matrix_alloc(GyroKalmanFilter::stateDim, GyroKalmanFilter::stateDim);
	gsl_vector_set_zero(Xg0);
	gsl_matrix_set_zero(Pg0);

	const double Qw = 0.2 * 3.141592 / 180.0;  // variance of a process noise related to an angular velocity state
	const double Rg = 0.25;  // variance of a measurement noise
	XGyroKalmanFilter xGyroFilter(Ts, Qw, Rg, Xg0, Pg0);
	YGyroKalmanFilter yGyroFilter(Ts, Qw, Rg, Xg0, Pg0);
	ZGyroKalmanFilter zGyroFilter(Ts, Qw, Rg, Xg0, Pg0);

	gsl_vector_free(Xg0);  Xg0 = NULL;
	gsl_matrix_free(Pg0);  Pg0 = NULL;

	//
	const size_t Nstep = 100;

	double prioriEstimate, posterioriEstimate;

	size_t step = 0;
	while (step < Nstep)
	{
		const bool retval1 = runImuKalmanFilter(step, xAccelFilter, prioriEstimate, posterioriEstimate);
		const bool retval2 = runImuKalmanFilter(step, yAccelFilter, prioriEstimate, posterioriEstimate);
		const bool retval3 = runImuKalmanFilter(step, zAccelFilter, prioriEstimate, posterioriEstimate);
		const bool retval4 = runImuKalmanFilter(step, xGyroFilter, prioriEstimate, posterioriEstimate);
		const bool retval5 = runImuKalmanFilter(step, yGyroFilter, prioriEstimate, posterioriEstimate);
		const bool retval6 = runImuKalmanFilter(step, zGyroFilter, prioriEstimate, posterioriEstimate);
		assert(retval1 && retval2 && retval3 && retval4 && retval5 && retval6);

		// advance time step
		++step;
	}
}
