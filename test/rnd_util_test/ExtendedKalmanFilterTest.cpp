#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/ExtendedKalmanFilter.h"
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

// "Kalman Filtering: Theory and Practice Using MATLAB" Example 4.1 (pp. 123)
class SimpleSystemExtendedKalmanFilter: public swl::ExtendedKalmanFilter
{
public:
	typedef swl::ExtendedKalmanFilter base_type;

public:
	SimpleSystemExtendedKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: base_type(x0, P0, stateDim, inputDim, outputDim),
	  Phi_(NULL), C_(NULL), /*W_(NULL), V_(NULL),*/ Qd_(NULL), Rd_(NULL), Bu_(NULL), Du_(NULL), f_eval_(NULL), h_eval_(NULL), y_tilde_(NULL)
	{
		// Phi = exp(T * A) = [ 1 ]
		Phi_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set(Phi_, 0, 0, 1.0);

		// C = [ 1 ]
		C_ = gsl_matrix_alloc(outputDim_, stateDim_);
		gsl_matrix_set(C_, 0, 0, 1.0);

		// W = [ 1 ]
		//W_ = gsl_matrix_alloc(stateDim_, inputDim_);
		//gsl_matrix_set_identity(W_);

		// V = [ 1 ]
		//V_ = gsl_matrix_alloc(outputDim_, outputDim_);
		//gsl_matrix_set_identity(V_);

		// Qd = W * Q * W^T
		Qd_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set(Qd_, 0, 0, 1.0);

		// Rd = V * R * V^T
		Rd_ = gsl_matrix_alloc(outputDim_, outputDim_);
		gsl_matrix_set(Rd_, 0, 0, 2.0);

		// no control input
		Bu_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(Bu_);

		// no measurement input
		Du_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(Du_);

		//
		f_eval_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(f_eval_);

		//
		h_eval_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(h_eval_);

		//
		y_tilde_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(y_tilde_);
	}
	~SimpleSystemExtendedKalmanFilter()
	{
		gsl_matrix_free(Phi_);  Phi_ = NULL;
		gsl_matrix_free(C_);  C_ = NULL;
		//gsl_matrix_free(W_);  W_ = NULL;
		//gsl_matrix_free(V_);  V_ = NULL;
		gsl_matrix_free(Qd_);  Qd_ = NULL;
		gsl_matrix_free(Rd_);  Rd_ = NULL;

		gsl_vector_free(Bu_);  Bu_ = NULL;
		gsl_vector_free(Du_);  Du_ = NULL;

		gsl_vector_free(f_eval_);  f_eval_ = NULL;
		gsl_vector_free(h_eval_);  h_eval_ = NULL;

		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;
	}

private:
	SimpleSystemExtendedKalmanFilter(const SimpleSystemExtendedKalmanFilter &rhs);
	SimpleSystemExtendedKalmanFilter & operator=(const SimpleSystemExtendedKalmanFilter &rhs);

private:
	// for continuous Kalman filter
	/*virtual*/ gsl_matrix * doGetSystemMatrix(const double time, const gsl_vector *state) const  // A(t) = df(t, x(t), u(t), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	// for discrete Kalman filter
	/*virtual*/ gsl_matrix * doGetStateTransitionMatrix(const size_t step, const gsl_vector *state) const  {  return Phi_;  }  // Phi(k) = exp(A(k) * T) where A(k) = df(k, x(k), u(k), 0)/dx
	/*virtual*/ gsl_matrix * doGetOutputMatrix(const size_t step, const gsl_vector *state) const  {  return C_;  }  // Cd(k) = dh(k, x(k), u(k), 0)/dx

	///*virtual*/ gsl_matrix * doGetProcessNoiseCouplingMatrix(const size_t step) const  {  return W_;  }  // W(k) = df(k, x(k), u(k), 0)/dw
	///*virtual*/ gsl_matrix * doGetMeasurementNoiseCouplingMatrix(const size_t step) const  {  return V_;  }  // V(k) = dh(k, x(k), u(k), 0)/dv
	/*virtual*/ gsl_matrix * doGetProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd = W * Q * W^t, but not Q
	/*virtual*/ gsl_matrix * doGetMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return Rd_;  }  // Rd = V * R * V^T, but not R

	/*virtual*/ gsl_vector * doEvaluatePlantEquation(const size_t step, const gsl_vector *state) const  // f = f(k, x(k), u(k), 0)
	{
		const gsl_matrix *Phi = doGetStateTransitionMatrix(step, state);
		gsl_vector_memcpy(f_eval_, Bu_);
		gsl_blas_dgemv(CblasNoTrans, 1.0, Phi, state, 1.0, f_eval_);
		return f_eval_;
	}
	/*virtual*/ gsl_vector * doEvaluateMeasurementEquation(const size_t step, const gsl_vector *state) const  // h = h(k, x(k), u(k), 0)
	{
		const gsl_matrix *Cd = doGetOutputMatrix(step, state);
		gsl_vector_memcpy(h_eval_, Du_);
		gsl_blas_dgemv(CblasNoTrans, 1.0, Cd, state, 1.0, h_eval_);
		return h_eval_;
	}

	// actual measurement
	/*virtual*/ gsl_vector * doGetMeasurement(const size_t step, const gsl_vector *state) const
	{
#if 1  // 1-based time step
		if (1 == step) gsl_vector_set(y_tilde_, 0, 2.0);
		else if (2 == step) gsl_vector_set(y_tilde_, 0, 3.0);
#else  // 0-based time step
		if (0 == step) gsl_vector_set(y_tilde_, 0, 2.0);
		else if (1 == step) gsl_vector_set(y_tilde_, 0, 3.0);
#endif
		else throw std::runtime_error("undefined measurement");
		return y_tilde_;
	}

protected:
	gsl_matrix *Phi_;
	gsl_matrix *C_;  // Cd = C
	//gsl_matrix *W_;
	//gsl_matrix *V_;
	gsl_matrix *Qd_;
	gsl_matrix *Rd_;

	// control input: Bu = Bd * u where Bd = A^-1 * (Ad - I) * B
	gsl_vector *Bu_;
	// measurement input: Du = Dd * u where Dd = D
	gsl_vector *Du_;

	// evalution of the plant equation: f = f(k, x(k), u(k), 0)
	gsl_vector *f_eval_;
	// evalution of the measurement equation: h = h(k, x(k), u(k), 0)
	gsl_vector *h_eval_;

	// actual measurement
	gsl_vector *y_tilde_;
};

// "Kalman Filtering: Theory and Practice Using MATLAB" Example 5.3 (pp. 184)
//	1) continuous ==> discrete
//	2) driving force exits
class LinearMassStringDamperSystemExtendedKalmanFilter: public swl::ExtendedKalmanFilter
{
public:
	typedef swl::ExtendedKalmanFilter base_type;

private:
	static const double T;
	static const double zeta;
	static const double omega;
	static const double Q;
	static const double R;
	static const double Fd;  // driving force

public:
	LinearMassStringDamperSystemExtendedKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: base_type(x0, P0, stateDim, inputDim, outputDim),
	  Phi_hat_(NULL), Cd_(NULL), /*W_(NULL), V_(NULL),*/ Qd_hat_(NULL), Rd_hat_(NULL), f_eval_(NULL), h_eval_(NULL), y_tilde_(NULL), driving_force_(NULL),
	  Phi_(NULL), x_(NULL)
	{
		//A_ = gsl_matrix_alloc(stateDim_, stateDim_);
		//gsl_matrix_set(A_, 0, 0, 0.0);  gsl_matrix_set(A_, 0, 1, 1.0);
		//gsl_matrix_set(A_, 1, 0, -omega * omega);  gsl_matrix_set(A_, 1, 1, -2.0 * zeta * omega);

		const double lambda = std::exp(-T * omega * zeta);
		const double psi    = 1.0 - zeta * zeta;
		const double xi     = std::sqrt(psi);
		const double theta  = xi * omega * T;
		const double c = std::cos(theta);
		const double s = std::sin(theta);

		// Phi = exp(T * A) -> I + T * A where A = df/dt
		//	the EKF approximation for Phi is I + T * A
		Phi_hat_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set_zero(Phi_hat_);

		// C = [ 1 0 0 ]
		Cd_ = gsl_matrix_alloc(outputDim_, stateDim_);
		gsl_matrix_set(Cd_, 0, 0, 1.0);  gsl_matrix_set(Cd_, 0, 1, 0.0);  gsl_matrix_set(Cd_, 0, 2, 0.0);

		// W = [ 0 1 0 ]^T
		//W_ = gsl_matrix_alloc(stateDim_, inputDim_);
		//gsl_matrix_set(W_, 0, 0, 0.0);  gsl_matrix_set(W_, 1, 0, 1.0);  gsl_matrix_set(W_, 2, 0, 0.0);

		// V = [ 1 ]
		//V_ = gsl_matrix_alloc(outputDim_, outputDim_);
		//gsl_matrix_set_identity(V_);

		// Qd = W * Q * W^T 
		//	the EKF approximation of Qd will be W * [ T * Q ] * W^T
		Qd_hat_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set_zero(Qd_hat_);
		gsl_matrix_set(Qd_hat_, 1, 1, Q * T);

		// Rd = V * R * V^T
		Rd_hat_ = gsl_matrix_alloc(outputDim_, outputDim_);
		gsl_matrix_set_all(Rd_hat_, R);

		//
		f_eval_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(f_eval_);

		//
		h_eval_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(h_eval_);

		//
		y_tilde_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(y_tilde_);

		// driving force: Bu = Bd * u
		//	where Bd = A^-1 * (Ad - I) * B, Ad = Phi = exp(A * T), B = [ 0 Fd 0 ]^T, u(t) = 1
		driving_force_ = gsl_vector_alloc(stateDim_);
#if 0
		gsl_vector_set(driving_force_, 0, Fd * (1.0 - lambda*(c-zeta*s*xi/psi)) / (omega*omega));
		gsl_vector_set(driving_force_, 1, Fd * lambda*s / (omega*xi));
		gsl_vector_set(driving_force_, 2, 0.0);
#else
		gsl_vector_set(driving_force_, 0, Fd * (1.0 - lambda*(c-zeta*s/xi)) / (omega*omega));
		gsl_vector_set(driving_force_, 1, Fd * lambda*s / (omega*xi));
		gsl_vector_set(driving_force_, 2, 0.0);
#endif

		// the exact solution for them matrix exponential, exp(T * A) for the state transition matrix
		Phi_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set_zero(Phi_);
#if 1
		gsl_matrix_set(Phi_, 0, 0, lambda*c + zeta*s/xi);  gsl_matrix_set(Phi_, 0, 1, lambda*s/(omega*xi));
		gsl_matrix_set(Phi_, 1, 0, -omega*lambda*s/xi);  gsl_matrix_set(Phi_, 1, 1, lambda*c - zeta*s/xi);
		gsl_matrix_set(Phi_, 2, 2, 1.0);
#else
		// my calculation result
		gsl_matrix_set(Phi_, 0, 0, lambda*(c+zeta*s/xi));  gsl_matrix_set(Phi_, 0, 1, lambda*s/(omega*xi));
		gsl_matrix_set(Phi_, 1, 0, -omega*lambda*s/xi);  gsl_matrix_set(Phi_, 1, 1, lambda*(c-zeta*s/xi));
		gsl_matrix_set(Phi_, 2, 2, 1.0);
#endif

		//
		x_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(x_);
		gsl_vector_set(x_, 2, zeta);
	}
	~LinearMassStringDamperSystemExtendedKalmanFilter()
	{
		gsl_matrix_free(Phi_hat_);  Phi_hat_ = NULL;
		gsl_matrix_free(Cd_);  Cd_ = NULL;
		//gsl_matrix_free(W_);  W_ = NULL;
		//gsl_matrix_free(V_);  V_ = NULL;
		gsl_matrix_free(Qd_hat_);  Qd_hat_ = NULL;
		gsl_matrix_free(Rd_hat_);  Rd_hat_ = NULL;

		gsl_vector_free(f_eval_);  f_eval_ = NULL;
		gsl_vector_free(h_eval_);  h_eval_ = NULL;

		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;

		gsl_vector_free(driving_force_);  driving_force_ = NULL;

		gsl_matrix_free(Phi_);  Phi_ = NULL;
		gsl_vector_free(x_);  x_ = NULL;
	}

private:
	LinearMassStringDamperSystemExtendedKalmanFilter(const LinearMassStringDamperSystemExtendedKalmanFilter &rhs);
	LinearMassStringDamperSystemExtendedKalmanFilter & operator=(const LinearMassStringDamperSystemExtendedKalmanFilter &rhs);

private:
	// for continuous Kalman filter
	/*virtual*/ gsl_matrix * doGetSystemMatrix(const double time, const gsl_vector *state) const  // A(t) = df(t, x(t), u(t), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	// for discrete Kalman filter
	/*virtual*/ gsl_matrix * doGetStateTransitionMatrix(const size_t step, const gsl_vector *state) const  // Phi(k) = exp(A(k) * T) where A(k) = df(k, x(k), u(k), 0)/dx
	{
		// Phi = exp(T * A) -> I + T * A where A = df/dt
		//	the EKF approximation for Phi is I + T * A
		gsl_matrix_set_zero(Phi_hat_);
		gsl_matrix_set(Phi_hat_, 0, 0, 1.0);  gsl_matrix_set(Phi_hat_, 0, 1, T);
		gsl_matrix_set(Phi_hat_, 1, 0, -T*omega*omega);  gsl_matrix_set(Phi_hat_, 1, 1, 1.0-2.0*T*omega*gsl_vector_get(state, 2));  gsl_matrix_set(Phi_hat_, 1, 2, -T*2.0*omega*gsl_vector_get(state, 1));
		gsl_matrix_set(Phi_hat_, 2, 2, 1.0);
		return Phi_hat_;
	}
	/*virtual*/ gsl_matrix * doGetOutputMatrix(const size_t step, const gsl_vector *state) const  // Cd(k) = dh(k, x(k), u(k), 0)/dx
	{  return Cd_;  }

	///*virtual*/ gsl_matrix * doGetProcessNoiseCouplingMatrix(const size_t step) const  {  return W_;  }  // W(k) = df(k, x(k), u(k), 0)/dw
	///*virtual*/ gsl_matrix * doGetMeasurementNoiseCouplingMatrix(const size_t step) const  {  return V_;  }  // V(k) = dh(k, x(k), u(k), 0)/dv
	/*virtual*/ gsl_matrix * doGetProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_hat_;  }  // Qd = W * Q * W^t, but not Q
	/*virtual*/ gsl_matrix * doGetMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return Rd_hat_;  }  // Rd = V * R * V^T, but not R

	/*virtual*/ gsl_vector * doEvaluatePlantEquation(const size_t step, const gsl_vector *state) const  // f = f(k, x(k), u(k), 0)
	{
		// update of true state w/o noise
		gsl_vector *v = gsl_vector_alloc(stateDim_);
		gsl_vector_memcpy(v, x_);
		gsl_vector_memcpy(x_, driving_force_);
		gsl_blas_dgemv(CblasNoTrans, 1.0, Phi_, v, 1.0, x_);
		gsl_vector_free(v);  v = NULL;

#if 0
		const double x1 = gsl_vector_get(state, 0);
		const double x2 = gsl_vector_get(state, 1);
		const double x3 = gsl_vector_get(state, 2);
		gsl_vector_set(f_eval_, 0, x2);
		gsl_vector_set(f_eval_, 0, -omega*omega*x1 - 2.0*x2*x3*omega);
		gsl_vector_set(f_eval_, 0, 0.0);
#else
		const gsl_matrix *Phi_hat = doGetStateTransitionMatrix(step, state);
		gsl_vector_memcpy(f_eval_, driving_force_);
		gsl_blas_dgemv(CblasNoTrans, 1.0, Phi_hat, state, 1.0, f_eval_);
#endif

		return f_eval_;
	}
	/*virtual*/ gsl_vector * doEvaluateMeasurementEquation(const size_t step, const gsl_vector *state) const  // h = h(k, x(k), u(k), 0)
	{
		gsl_vector_set(h_eval_, 0, gsl_vector_get(state, 0));
		return h_eval_;
	}

	// actual measurement
	/*virtual*/ gsl_vector * doGetMeasurement(const size_t step, const gsl_vector *state) const
	{
		gsl_vector_set(y_tilde_, 0, gsl_vector_get(x_, 0));  // measurement (no noise)
		return y_tilde_;
	}

protected:
	gsl_matrix *Phi_hat_;
	gsl_matrix *Cd_;  // Cd = C
	//gsl_matrix *W_;
	//gsl_matrix *V_;
	gsl_matrix *Qd_hat_;
	gsl_matrix *Rd_hat_;

	// evalution of the plant equation: f = f(k, x(k), u(k), 0)
	gsl_vector *f_eval_;
	// evalution of the measurement equation: h = h(k, x(k), u(k), 0)
	gsl_vector *h_eval_;

	// actual measurement
	gsl_vector *y_tilde_;

	// driving force input
	gsl_vector *driving_force_;

	gsl_matrix *Phi_;
	gsl_vector *x_;
};

/*static*/ const double LinearMassStringDamperSystemExtendedKalmanFilter::T = 0.01;
/*static*/ const double LinearMassStringDamperSystemExtendedKalmanFilter::zeta = 0.2;
/*static*/ const double LinearMassStringDamperSystemExtendedKalmanFilter::omega = 5.0;
/*static*/ const double LinearMassStringDamperSystemExtendedKalmanFilter::Q = 4.47;
/*static*/ const double LinearMassStringDamperSystemExtendedKalmanFilter::R = 0.01;
/*static*/ const double LinearMassStringDamperSystemExtendedKalmanFilter::Fd = 12.0;

void simple_system_kalman_filter()
{
	const size_t stateDim = 1;
	const size_t inputDim = 1;
	const size_t outputDim = 1;

	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_vector_set(x0, 0, 1.0);
	gsl_matrix_set(P0, 0, 0, 10.0);

	SimpleSystemExtendedKalmanFilter filter(x0, P0, stateDim, inputDim, outputDim);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	const size_t Nstep = 2;
	std::vector<double> state, gain, errVar;
	state.reserve(Nstep * 2);
	gain.reserve(Nstep);
	errVar.reserve(Nstep * 2);

#if 1
	// method #1
	// 1-based time step. 0-th time step is initial
	size_t step = 0;
	while (step < Nstep)
	{
		// 0. initial estimates: x(0) & P(0)

		// 1. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const bool retval1 = filter.updateTime(step);
		assert(retval1);

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			state.push_back(gsl_vector_get(x_hat, 0));
			errVar.push_back(gsl_matrix_get(P, 0, 0));
		}

		// advance time step
		++step;

		// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const bool retval2 = filter.updateMeasurement(step);
		assert(retval2);

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *K = filter.getKalmanGain();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			state.push_back(gsl_vector_get(x_hat, 0));
			gain.push_back(gsl_matrix_get(K, 0, 0));
			errVar.push_back(gsl_matrix_get(P, 0, 0));
		}
	}
#else
	// method #2
	// 0-based time step. 0-th time step is initial
	size_t step = 0;
	while (step < Nstep)
	{
		// 0. initial estimates: x-(0) & P-(0)

		// 1. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const bool retval1 = filter.updateMeasurement(step);
		assert(retval1);

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *K = filter.getKalmanGain();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			state.push_back(gsl_vector_get(x_hat, 0));
			gain.push_back(gsl_matrix_get(K, 0, 0));
			errVar.push_back(gsl_matrix_get(P, 0, 0));
		}

		// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const bool retval2 = filter.updateTime(step);
		assert(retval2);

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			state.push_back(gsl_vector_get(x_hat, 0));
			errVar.push_back(gsl_matrix_get(P, 0, 0));
		}

		// advance time step
		++step;
	}
#endif
}

void linear_mass_spring_damper_system_extended_kalman_filter()
{
	const size_t stateDim = 3;
	const size_t inputDim = 1;
	const size_t outputDim = 1;

	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_vector_set_zero(x0);
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(P0);
	gsl_matrix_scale(P0, 2.0);

	LinearMassStringDamperSystemExtendedKalmanFilter filter(x0, P0, stateDim, inputDim, outputDim);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	const size_t Nstep = 100;
	std::vector<double> pos, vel, damp;
	std::vector<double> posGain, velGain, dampGain;
	std::vector<double> posErrVar, velErrVar, dampErrVar;
	pos.reserve(Nstep * 2);
	vel.reserve(Nstep * 2);
	damp.reserve(Nstep * 2);
	posGain.reserve(Nstep);
	velGain.reserve(Nstep);
	dampGain.reserve(Nstep);
	posErrVar.reserve(Nstep * 2);
	velErrVar.reserve(Nstep * 2);
	dampErrVar.reserve(Nstep * 2);

#if 1
	// method #1
	// 1-based time step. 0-th time step is initial
	size_t step = 0;
	while (step < Nstep)
	{
		// 0. initial estimates: x(0) & P(0)

		// 1. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const bool retval1 = filter.updateTime(step);
		assert(retval1);

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			pos.push_back(gsl_vector_get(x_hat, 0));
			vel.push_back(gsl_vector_get(x_hat, 1));
			damp.push_back(gsl_vector_get(x_hat, 2));
			posErrVar.push_back(gsl_matrix_get(P, 0, 0));
			velErrVar.push_back(gsl_matrix_get(P, 1, 1));
			dampErrVar.push_back(gsl_matrix_get(P, 2, 2));
		}

		// advance time step
		++step;

		// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const bool retval2 = filter.updateMeasurement(step);
		assert(retval2);

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *K = filter.getKalmanGain();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			pos.push_back(gsl_vector_get(x_hat, 0));
			vel.push_back(gsl_vector_get(x_hat, 1));
			damp.push_back(gsl_vector_get(x_hat, 2));
			posGain.push_back(gsl_matrix_get(K, 0, 0));
			velGain.push_back(gsl_matrix_get(K, 1, 0));
			dampGain.push_back(gsl_matrix_get(K, 2, 0));
			posErrVar.push_back(gsl_matrix_get(P, 0, 0));
			velErrVar.push_back(gsl_matrix_get(P, 1, 1));
			dampErrVar.push_back(gsl_matrix_get(P, 2, 2));
		}
	}
#else
	// method #2
	// 0-based time step. 0-th time step is initial
	size_t step = 0;
	while (step < Nstep)
	{
		// 0. initial estimates: x-(0) & P-(0)

		// 1. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const bool retval1 = filter.updateMeasurement(step);
		assert(retval1);

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *K = filter.getKalmanGain();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			pos.push_back(gsl_vector_get(x_hat, 0));
			vel.push_back(gsl_vector_get(x_hat, 1));
			damp.push_back(gsl_vector_get(x_hat, 2));
			posGain.push_back(gsl_matrix_get(K, 0, 0));
			velGain.push_back(gsl_matrix_get(K, 1, 0));
			dampGain.push_back(gsl_matrix_get(K, 2, 0));
			posErrVar.push_back(gsl_matrix_get(P, 0, 0));
			velErrVar.push_back(gsl_matrix_get(P, 1, 1));
			dampErrVar.push_back(gsl_matrix_get(P, 2, 2));
		}

		// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const bool retval2 = filter.updateTime(step);
		assert(retval2);

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			pos.push_back(gsl_vector_get(x_hat, 0));
			vel.push_back(gsl_vector_get(x_hat, 1));
			damp.push_back(gsl_vector_get(x_hat, 2));
			posErrVar.push_back(gsl_matrix_get(P, 0, 0));
			velErrVar.push_back(gsl_matrix_get(P, 1, 1));
			dampErrVar.push_back(gsl_matrix_get(P, 2, 2));
		}

		// advance time step
		++step;
	}
#endif
}

}  // unnamed namespace

void extended_kalman_filter()
{
	simple_system_kalman_filter();
	linear_mass_spring_damper_system_extended_kalman_filter();
}
