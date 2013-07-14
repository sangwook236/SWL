//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/DiscreteNonlinearStochasticSystem.h"
#include "swl/rnd_util/UnscentedKalmanFilterWithAdditiveNoise.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <fstream>
#include <cmath>
#include <ctime>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


void output_data_to_file(std::ostream &stream, const std::string &variable_name, const std::vector<double> &data);

namespace {
namespace local {

// "Kalman Filtering: Theory and Practice Using MATLAB" Example 4.1 (pp. 123)
class SimpleLinearSystem: public swl::DiscreteNonlinearStochasticSystem
{
public:
	typedef swl::DiscreteNonlinearStochasticSystem base_type;

public:
	static const double Ts;
	static const double Q;
	static const double R;

public:
	SimpleLinearSystem(const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim)
	: base_type(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim),
	  Phi_(NULL), C_(NULL), /*W_(NULL), V_(NULL),*/ Qd_(NULL), Rd_(NULL), Bu_(NULL), Du_(NULL), y_tilde_(NULL)
	{
		// Phi = exp(A * Ts) = [ 1 ]
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
		gsl_matrix_set(Qd_, 0, 0, Q);

		// Rd = V * R * V^T
		Rd_ = gsl_matrix_alloc(outputDim_, outputDim_);
		gsl_matrix_set(Rd_, 0, 0, R);

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
	~SimpleLinearSystem()
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
	SimpleLinearSystem(const SimpleLinearSystem &rhs);
	SimpleLinearSystem & operator=(const SimpleLinearSystem &rhs);

public:
	// the stochastic differential equation: f = f(k, x(k), u(k), w(k)) = Phi(k) * x(k) + Bd(k) * u(k) + W(k) * w(k)
	/*virtual*/ gsl_vector * evaluatePlantEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		gsl_vector_memcpy(f_eval_, Bu_);  // Bd(k) = 0
		if (noise) gsl_vector_add(f_eval_, noise);  // W(k) = I
		//gsl_blas_dgemv(CblasNoTrans, 1.0, Bd_, input, 1.0, f_eval_);
		gsl_blas_dgemv(CblasNoTrans, 1.0, Phi_, state, 1.0, f_eval_);
		return f_eval_;
	}
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const  // Phi(k) = exp(A(k) * Ts) where A(k) = df(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const  // Bu(k) = Bd(k) * u(k)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const  // W(k) = df(k, x(k), u(k), 0)/dw
	//{  throw std::runtime_error("this function doesn't have to be called");  }

	// the observation equation: h = h(k, x(k), u(k), v(k)) = Cd(k) * x(k) + Dd(k) * u(k) + V(k) * v(k)
	/*virtual*/ gsl_vector * evaluateMeasurementEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		gsl_vector_memcpy(h_eval_, Du_);  // Dd(k) = 0
		if (noise) gsl_vector_add(h_eval_, noise);  // V(k) = I
		//gsl_blas_dgemv(CblasNoTrans, 1.0, Dd_, input, 1.0, h_eval_);
		gsl_blas_dgemv(CblasNoTrans, 1.0, C_, state, 1.0, h_eval_);
		return h_eval_;
	}
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const  // Cd(k) = dh(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const  // Du(k) = D(k) * u(k) (D == Dd)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const  // V(k) = dh(k, x(k), u(k), 0)/dv
	//{  throw std::runtime_error("this function doesn't have to be called");  }

	// noise covariance matrices
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  // Qd = W * Q * W^T, but not Q
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  // Rd = V * R * V^T, but not R
	{  throw std::runtime_error("this function doesn't have to be called");  }

	// actual measurement
	gsl_vector * simulateMeasurement(const size_t step, const gsl_vector *state) const
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

private:
	gsl_matrix *Phi_;
	gsl_matrix *C_;  // Cd = C
	//gsl_matrix *W_;
	//gsl_matrix *V_;
	gsl_matrix *Qd_;
	gsl_matrix *Rd_;

	// control input: Bu = Bd * u where Bd = integrate(exp(A * t), {t, 0, Ts}) * B or A^-1 * (Ad - I) * B if A is nonsingular
	gsl_vector *Bu_;
	// measurement input: Du = Dd * u where Dd = D
	gsl_vector *Du_;

	// evalution of the plant equation: f = f(k, x(k), u(k), w(k))
	gsl_vector *f_eval_;
	// evalution of the measurement equation: h = h(k, x(k), u(k), v(k))
	gsl_vector *h_eval_;

	// actual measurement
	gsl_vector *y_tilde_;
};

/*static*/ const double SimpleLinearSystem::Ts = 1.0;
/*static*/ const double SimpleLinearSystem::Q = 1.0;
/*static*/ const double SimpleLinearSystem::R = 2.0;

// "Kalman Filtering: Theory and Practice Using MATLAB" Example 5.3 (pp. 184)
//	1) continuous ==> discrete
//	2) driving force exits
class LinearMassStringDamperSystem: public swl::DiscreteNonlinearStochasticSystem
{
public:
	typedef swl::DiscreteNonlinearStochasticSystem base_type;

public:
	static const double Ts;
	static const double zeta;
	static const double omega;
	static const double Q;
	static const double R;
	static const double Fd;  // driving force

public:
	LinearMassStringDamperSystem(const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim)
	: base_type(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim),
	  f_eval_(NULL), h_eval_(NULL), y_tilde_(NULL), driving_force_(NULL),
	  Phi_true_(NULL), x_true_(NULL)
	{
		const double lambda = std::exp(-Ts * omega * zeta);
		const double psi    = 1.0 - zeta * zeta;
		const double xi     = std::sqrt(psi);
		const double theta  = xi * omega * Ts;
		const double c = std::cos(theta);
		const double s = std::sin(theta);

		//
		f_eval_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(f_eval_);

		//
		h_eval_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(h_eval_);

		//
		y_tilde_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(y_tilde_);

		// driving force: Bu = Bd * u where u(t) = 1
		//	Bd = integrate(exp(A * t), {t, 0, Ts}) * B or A^-1 * (Ad - I) * B if A is nonsingular
		//	Ad = Phi = exp(A * Ts)
		//	B = [ 0 ; Fd ; 0 ]
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

		// the exact solution for them matrix exponential, exp(A * Ts) for the state transition matrix
		Phi_true_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set_zero(Phi_true_);
#if 1
		gsl_matrix_set(Phi_true_, 0, 0, lambda*c + zeta*s/xi);  gsl_matrix_set(Phi_true_, 0, 1, lambda*s/(omega*xi));
		gsl_matrix_set(Phi_true_, 1, 0, -omega*lambda*s/xi);  gsl_matrix_set(Phi_true_, 1, 1, lambda*c - zeta*s/xi);
		gsl_matrix_set(Phi_true_, 2, 2, 1.0);
#else
		// my calculation result
		gsl_matrix_set(Phi_true_, 0, 0, lambda*(c+zeta*s/xi));  gsl_matrix_set(Phi_true_, 0, 1, lambda*s/(omega*xi));
		gsl_matrix_set(Phi_true_, 1, 0, -omega*lambda*s/xi);  gsl_matrix_set(Phi_true_, 1, 1, lambda*(c-zeta*s/xi));
		gsl_matrix_set(Phi_true_, 2, 2, 1.0);
#endif

		//
		x_true_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(x_true_);
		gsl_vector_set(x_true_, 2, zeta);
	}
	~LinearMassStringDamperSystem()
	{
		gsl_vector_free(f_eval_);  f_eval_ = NULL;
		gsl_vector_free(h_eval_);  h_eval_ = NULL;

		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;

		gsl_vector_free(driving_force_);  driving_force_ = NULL;

		gsl_matrix_free(Phi_true_);  Phi_true_ = NULL;
		gsl_vector_free(x_true_);  x_true_ = NULL;
	}

private:
	LinearMassStringDamperSystem(const LinearMassStringDamperSystem &rhs);
	LinearMassStringDamperSystem & operator=(const LinearMassStringDamperSystem &rhs);

public:
	// the stochastic differential equation: f = f(k, x(k), u(k), w(k))
	/*virtual*/ gsl_vector * evaluatePlantEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		// update of true state w/o noise
		gsl_vector *v = gsl_vector_alloc(stateDim_);
		gsl_vector_memcpy(v, x_true_);
		gsl_vector_memcpy(x_true_, driving_force_);
		gsl_blas_dgemv(CblasNoTrans, 1.0, Phi_true_, v, 1.0, x_true_);

		//
		const double &x1 = gsl_vector_get(state, 0);
		const double &x2 = gsl_vector_get(state, 1);
		const double &x3 = gsl_vector_get(state, 2);
		const double &w2 = noise ? gsl_vector_get(noise, 1) : 0.0;

		// Phi = exp(A * Ts) -> I + A * Ts where A = df/dx
		//	the EKF approximation for Phi is I + A * Ts
		//	A = [ 0 1 0 ; -omega^2 -2*omega*x3 -2*omega*x2 ; 0 0 0 ]  <==  A = df/dx

		const double f1 = x1 + Ts * x2;
		const double f2 = -omega*omega * Ts * x1 + (1.0 - 2.0 * omega * Ts * x3) * x2 - 2.0 * omega * Ts * x2 * x3;
		const double f3 = x3;

		gsl_vector_set(f_eval_, 0, f1);
		gsl_vector_set(f_eval_, 1, f2);
		gsl_vector_set(f_eval_, 2, f3);

		// input driven by driving force
		gsl_vector_add(f_eval_, driving_force_);

		// input driven by noise
		//const double lambda = std::exp(-Ts * omega * zeta);
		//const double psi    = 1.0 - zeta * zeta;
		//const double xi     = std::sqrt(psi);
		//const double theta  = xi * omega * Ts;
		//const double c = std::cos(theta);
		//const double s = std::sin(theta);

		//gsl_vector_set(v, 0, w2 * (1.0 - lambda*(c-zeta*s/xi)) / (omega*omega));
		//gsl_vector_set(v, 1, w2 * lambda*s / (omega*xi));
		//gsl_vector_set(v, 2, 0.0);

		//gsl_vector_add(f_eval_, v);

		gsl_vector_free(v);  v = NULL;

		return f_eval_;
	}
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const  // Phi(k) = exp(A(k) * Ts) where A(k) = df(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const  // Bu(k) = Bd(k) * u(k)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const  // W(k) = df(k, x(k), u(k), 0)/dw
	//{  throw std::runtime_error("this function doesn't have to be called");  }

	// the observation equation: h = h(k, x(k), u(k), v(k))
	/*virtual*/ gsl_vector * evaluateMeasurementEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		const double &x1 = gsl_vector_get(state, 0);
		const double &v1 = noise ? gsl_vector_get(noise, 0) : 0.0;

		gsl_vector_set(h_eval_, 0, x1 + v1);
		return h_eval_;
	}
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const  // Cd(k) = dh(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const  // Du(k) = D(k) * u(k) (D == Dd)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const  // V(k) = dh(k, x(k), u(k), 0)/dv
	//{  throw std::runtime_error("this function doesn't have to be called");  }

	// noise covariance matrices
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  // Qd = W * Q * W^T, but not Q
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  // Rd = V * R * V^T, but not R
	{  throw std::runtime_error("this function doesn't have to be called");  }

	// actual measurement
	gsl_vector * simulateMeasurement(const size_t step, const gsl_vector *state) const
	{
		// true state, but not estimated state
		gsl_vector_set(y_tilde_, 0, gsl_vector_get(x_true_, 0));  // measurement (no noise)
		return y_tilde_;
	}

private:
	// evalution of the plant equation: f = f(k, x(k), u(k), w(k))
	gsl_vector *f_eval_;
	// evalution of the measurement equation: h = h(k, x(k), u(k), v(k))
	gsl_vector *h_eval_;

	// actual measurement
	gsl_vector *y_tilde_;

	// driving force input
	gsl_vector *driving_force_;

	gsl_matrix *Phi_true_;
	gsl_vector *x_true_;
};

/*static*/ const double LinearMassStringDamperSystem::Ts = 0.01;
/*static*/ const double LinearMassStringDamperSystem::zeta = 0.2;
/*static*/ const double LinearMassStringDamperSystem::omega = 5.0;
/*static*/ const double LinearMassStringDamperSystem::Q = 4.47;
/*static*/ const double LinearMassStringDamperSystem::R = 0.01;
/*static*/ const double LinearMassStringDamperSystem::Fd = 12.0;

class SimpleNonlinearSystem: public swl::DiscreteNonlinearStochasticSystem
{
public:
	typedef swl::DiscreteNonlinearStochasticSystem base_type;

public:
	static const double Ts;

public:
	SimpleNonlinearSystem(const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim)
	: base_type(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim),
	  f_eval_(NULL), h_eval_(NULL)
	{
		f_eval_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(f_eval_);

		//
		h_eval_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(h_eval_);
	}
	~SimpleNonlinearSystem()
	{
		gsl_vector_free(f_eval_);  f_eval_ = NULL;
		gsl_vector_free(h_eval_);  h_eval_ = NULL;
	}

private:
	SimpleNonlinearSystem(const SimpleNonlinearSystem &rhs);
	SimpleNonlinearSystem & operator=(const SimpleNonlinearSystem &rhs);

public:
	// the stochastic differential equation: f = f(k, x(k), u(k), w(k)) = f(k, x(k), u(k)) + w(k)
	/*virtual*/ gsl_vector * evaluatePlantEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		// update of true state w/o noise
		const double &x1 = gsl_vector_get(state, 0);
		const double &x2 = gsl_vector_get(state, 1);
		const double &x3 = gsl_vector_get(state, 2);

		const double f1 = x2;
		const double f2 = x3;
		const double f3 = 0.05 * x1 * (x2 + x3);

		gsl_vector_set(f_eval_, 0, f1);
		gsl_vector_set(f_eval_, 1, f2);
		gsl_vector_set(f_eval_, 2, f3);
		return f_eval_;
	}
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const  // Phi(k) = exp(A(k) * Ts) where A(k) = df(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const  // Bu(k) = Bd(k) * u(k)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const  // W(k) = df(k, x(k), u(k), 0)/dw
	//{  throw std::runtime_error("this function doesn't have to be called");  }

	// the observation equation: h = h(k, x(k), u(k), v(k)) = h(k, x(k), u(k)) + v(k)
	/*virtual*/ gsl_vector * evaluateMeasurementEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		// update of true state w/o noise
		const double &x1 = gsl_vector_get(state, 0);

		gsl_vector_set(h_eval_, 0, x1);
		return h_eval_;
	}
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const  // Cd(k) = dh(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const  // Du(k) = D(k) * u(k) (D == Dd)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const  // V(k) = dh(k, x(k), u(k), 0)/dv
	//{  throw std::runtime_error("this function doesn't have to be called");  }

	// noise covariance matrices
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  // Qd = W * Rw * W^T, but not Rw
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  // Rd = V * Rv * V^T, but not Rv
	{  throw std::runtime_error("this function doesn't have to be called");  }

private:
	// evalution of the plant equation: f = f(k, x(k), u(k), w(k))
	gsl_vector *f_eval_;
	// evalution of the measurement equation: h = h(k, x(k), u(k), v(k))
	gsl_vector *h_eval_;
};

/*static*/ const double SimpleNonlinearSystem::Ts = 1.0;

// "On sequential Monte Carlo sampling methods for Bayesian filtering", Arnaud Doucet, Simon Godsill, and Christophe Andrieu,
//	Statistics and Computing, 10, pp. 197-208, 2000  ==>  pp. 206
// "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", N. J. Gordon, D. J. Salmond, and A. F. M. Smith,
//	IEE Proceedings, vol. 140, no. 2, pp. 107-113, 1993  ==>  pp. 109
// "A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking", M. S. Arulampalam, Simon Maskell, Neil Gordon, and Time Clapp,
//	Trans. on Signal Processing, vol. 50, no. 2, pp. 174-188, 2002  ==>  pp. 183
// "An Introduction to Sequential Monte Carlo Methods", Arnaud Doucet, Nando de Freitas, and Neil Gordon, 2001  ==>  pp. 12
class NonstationaryGrowthSystem: public swl::DiscreteNonlinearStochasticSystem
{
public:
	typedef swl::DiscreteNonlinearStochasticSystem base_type;

public:
	static const double Ts;
	static const double Rw;
	static const double Rv;

public:
	NonstationaryGrowthSystem(const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim)
	: base_type(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim),
	  f_eval_(NULL), h_eval_(NULL), y_tilde_(NULL),
	  baseGenerator_(static_cast<unsigned int>(std::time(NULL))), generator_(baseGenerator_, boost::normal_distribution<>(0.0, Rw))
	{
		f_eval_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(f_eval_);

		//
		h_eval_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(h_eval_);

		//
		y_tilde_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(y_tilde_);
	}
	~NonstationaryGrowthSystem()
	{
		gsl_vector_free(f_eval_);  f_eval_ = NULL;
		gsl_vector_free(h_eval_);  h_eval_ = NULL;

		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;
	}

private:
	NonstationaryGrowthSystem(const NonstationaryGrowthSystem &rhs);
	NonstationaryGrowthSystem & operator=(const NonstationaryGrowthSystem &rhs);

public:
	// the stochastic differential equation: f = f(k, x(k), u(k), w(k))
	/*virtual*/ gsl_vector * evaluatePlantEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		// update of true state w/o noise
		const double &x = gsl_vector_get(state, 0);
		const double &w = noise ? gsl_vector_get(noise, 0) : 0.0;

		const double f = 0.5 * x + 25.0 * x / (1 + x*x) + 8.0 * std::cos(1.2 * Ts * step) + w;

		gsl_vector_set(f_eval_, 0, f);
		return f_eval_;
	}
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const  // Phi(k) = exp(A(k) * Ts) where A(k) = df(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const  // Bu(k) = Bd(k) * u(k)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const  // W(k) = df(k, x(k), u(k), 0)/dw
	//{  throw std::runtime_error("this function doesn't have to be called");  }

	// the observation equation: h = h(k, x(k), u(k), v(k))
	/*virtual*/ gsl_vector * evaluateMeasurementEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		const double &x = gsl_vector_get(state, 0);
		const double &v = noise ? gsl_vector_get(noise, 0) : 0.0;

		gsl_vector_set(h_eval_, 0, (x * x / 20.0) + v);
		return h_eval_;
	}
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const  // Cd(k) = dh(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const  // Du(k) = D(k) * u(k) (D == Dd)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const  // V(k) = dh(k, x(k), u(k), 0)/dv
	//{  throw std::runtime_error("this function doesn't have to be called");  }

	// noise covariance matrices
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  // Qd = W * Rw * W^T, but not Rw
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  // Rd = V * Rv * V^T, but not Rv
	{  throw std::runtime_error("this function doesn't have to be called");  }

	// actual measurement
	gsl_vector * simulateMeasurement(const size_t step, const gsl_vector *state) const
	{
		const double &x = gsl_vector_get(state, 0);
		const double f = 0.5 * x + 25.0 * x / (1 + x*x) + 8.0 * std::cos(1.2 * Ts * step) + generator_();

		gsl_vector_set(y_tilde_, 0, f);  // measurement (no noise)
		return y_tilde_;
	}

private:
	typedef boost::minstd_rand base_generator_type;
	typedef boost::variate_generator<base_generator_type &, boost::normal_distribution<> > generator_type;

private:
	// evalution of the plant equation: f = f(k, x(k), u(k), w(k))
	gsl_vector *f_eval_;
	// evalution of the measurement equation: h = h(k, x(k), u(k), v(k))
	gsl_vector *h_eval_;

	// actual measurement
	gsl_vector *y_tilde_;

	base_generator_type baseGenerator_;
	mutable generator_type generator_;
};

/*static*/ const double NonstationaryGrowthSystem::Ts = 1.0;
/*static*/ const double NonstationaryGrowthSystem::Rw = std::sqrt(10.0);
/*static*/ const double NonstationaryGrowthSystem::Rv = 0.1;

void simple_system_unscented_kalman_filter_with_additive_noise()
{
	const size_t stateDim = 1;
	const size_t inputDim = 1;
	const size_t outputDim = 1;
	const size_t processNoiseDim = stateDim;
	const size_t observationNoiseDim = outputDim;

#if 0
	// for unscented Kalman filter
	const size_t L = stateDim + processNoiseDim + observationNoiseDim;
#else
	// for unscented Kalman filter with additive noise
	const size_t L = stateDim;
#endif
	const double alpha = 1.0e-3;
	const double beta = 2.0;  // for normal distribution
	const double kappa = 0.0;  // 3.0 - L;

	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_vector_set(x0, 0, 1.0);
	gsl_matrix_set(P0, 0, 0, 10.0);

	const SimpleLinearSystem system(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim);
	swl::UnscentedKalmanFilterWithAdditiveNoise filter(system, alpha, beta, kappa, x0, P0);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	gsl_vector *w = gsl_vector_alloc(processNoiseDim);
	gsl_vector_set_zero(w);
	gsl_vector *v = gsl_vector_alloc(observationNoiseDim);
	gsl_vector_set_zero(v);
	gsl_matrix *Q = gsl_matrix_alloc(processNoiseDim, processNoiseDim);
	gsl_matrix_set_zero(Q);
	gsl_matrix_set(Q, 0, 0, SimpleLinearSystem::Q);
	gsl_matrix *R = gsl_matrix_alloc(observationNoiseDim, observationNoiseDim);
	gsl_matrix_set_zero(R);
	gsl_matrix_set(R, 0, 0, SimpleLinearSystem::R);

	//
	const size_t Nstep = 2;
	std::vector<double> state, gain, errVar;
	state.reserve(Nstep * 2);
	gain.reserve(Nstep);
	errVar.reserve(Nstep * 2);

	// method #1
	// 1-based time step. 0-th time step is initial
	size_t step = 0;
	while (step < Nstep)
	{
		// 0. initial estimates: x(0) & P(0)

		// 1. unscented transformation
		filter.performUnscentedTransformation();

		// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const bool retval1 = filter.updateTime(step, NULL, Q);
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

		// 3. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const gsl_vector *actualMeasurement = system.simulateMeasurement(step, filter.getEstimatedState());
		const bool retval2 = filter.updateMeasurement(step, actualMeasurement, NULL, R);
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

	gsl_vector_free(w);  w = NULL;
	gsl_vector_free(v);  v = NULL;
	gsl_matrix_free(Q);  Q = NULL;
	gsl_matrix_free(R);  R = NULL;

	//
	std::ofstream stream("../data/unscented_kalman_filter_with_additive_noise.dat", std::ios::out | std::ios::trunc);
	if (stream)
	{
		output_data_to_file(stream, "state", state);
		output_data_to_file(stream, "gain", gain);
		output_data_to_file(stream, "errVar", errVar);

		stream.close();
	}
}

void linear_mass_spring_damper_system_unscented_kalman_filter_with_additive_noise()
{
	const size_t stateDim = 3;
	const size_t inputDim = 1;
	const size_t outputDim = 1;
	const size_t processNoiseDim = stateDim;
	const size_t observationNoiseDim = outputDim;

#if 0
	// for unscented Kalman filter
	const size_t L = stateDim + processNoiseDim + observationNoiseDim;
#else
	// for unscented Kalman filter with additive noise
	const size_t L = stateDim;
#endif
	const double alpha = 1.0e-3;
	const double beta = 2.0;  // for normal distribution
	const double kappa = 0.0;  // 3.0 - L;

	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_vector_set_zero(x0);
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(P0);
	gsl_matrix_scale(P0, 2.0);

	const LinearMassStringDamperSystem system(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim);
	swl::UnscentedKalmanFilterWithAdditiveNoise filter(system, alpha, beta, kappa, x0, P0);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	gsl_vector *w = gsl_vector_alloc(processNoiseDim);
	gsl_vector_set_zero(w);
	gsl_vector *v = gsl_vector_alloc(observationNoiseDim);
	gsl_vector_set_zero(v);
	gsl_matrix *Q = gsl_matrix_alloc(processNoiseDim, processNoiseDim);
	gsl_matrix_set_zero(Q);
	gsl_matrix_set(Q, 1, 1, LinearMassStringDamperSystem::Q);
	gsl_matrix *R = gsl_matrix_alloc(observationNoiseDim, observationNoiseDim);
	gsl_matrix_set_zero(R);
	gsl_matrix_set(R, 0, 0, LinearMassStringDamperSystem::R);

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

	// method #1
	// 1-based time step. 0-th time step is initial
	size_t step = 0;
	while (step < Nstep)
	{
		// 0. initial estimates: x(0) & P(0)

		// 1. unscented transformation
		filter.performUnscentedTransformation();

		// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const bool retval1 = filter.updateTime(step, NULL, Q);
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

		// 3. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const gsl_vector *actualMeasurement = system.simulateMeasurement(step, filter.getEstimatedState());
		const bool retval2 = filter.updateMeasurement(step, actualMeasurement, NULL, R);
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

	gsl_vector_free(w);  w = NULL;
	gsl_vector_free(v);  v = NULL;
	gsl_matrix_free(Q);  Q = NULL;
	gsl_matrix_free(R);  R = NULL;

	//
	std::ofstream stream("../data/unscented_kalman_filter_with_additive_noise.dat", std::ios::out | std::ios::trunc);
	if (stream)
	{
		output_data_to_file(stream, "pos", pos);
		output_data_to_file(stream, "vel", vel);
		output_data_to_file(stream, "damp", damp);
		output_data_to_file(stream, "posGain", posGain);
		output_data_to_file(stream, "velGain", velGain);
		output_data_to_file(stream, "dampGain", dampGain);
		output_data_to_file(stream, "posErrVar", posErrVar);
		output_data_to_file(stream, "velErrVar", velErrVar);
		output_data_to_file(stream, "dampErrVar", dampErrVar);

		stream.close();
	}
}

void simple_nonlinear_system_unscented_kalman_filter_with_additive_noise()
{
	const size_t stateDim = 3;
	const size_t inputDim = 1;
	const size_t outputDim = 1;
	const size_t processNoiseDim = stateDim;
	const size_t observationNoiseDim = outputDim;

	const double sigma_w = 0.1;
	const double sigma_v = 0.1;

	//
#if 0
	// for unscented Kalman filter
	const size_t L = stateDim + processNoiseDim + observationNoiseDim;
#else
	// for unscented Kalman filter with additive noise
	const size_t L = stateDim;
#endif
	const double alpha = 1.0e-3;
	const double beta = 2.0;  // for normal distribution
	const double kappa = 0.0;  // 3.0 - L;
	const double sigma_init = 1.0;

	//
	typedef boost::minstd_rand base_generator_type;
	typedef boost::variate_generator<base_generator_type &, boost::normal_distribution<> > generator_type;
	base_generator_type baseGenerator(static_cast<unsigned int>(std::time(NULL)));
	generator_type generator1(baseGenerator, boost::normal_distribution<>(0.0, sigma_w));
	generator_type generator2(baseGenerator, boost::normal_distribution<>(0.0, sigma_v));

	gsl_vector *actualState = gsl_vector_alloc(stateDim);
	gsl_vector_set(actualState, 0, 0.0);
	gsl_vector_set(actualState, 1, 0.0);
	gsl_vector_set(actualState, 2, 1.0);

	//
	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_vector_set(x0, 0, gsl_vector_get(actualState, 0) + generator1());
	gsl_vector_set(x0, 1, gsl_vector_get(actualState, 1) + generator1());
	gsl_vector_set(x0, 2, gsl_vector_get(actualState, 2) + generator1());

	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(P0);
	gsl_matrix_scale(P0, sigma_init);

	const SimpleNonlinearSystem system(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim);
	swl::UnscentedKalmanFilterWithAdditiveNoise filter(system, alpha, beta, kappa, x0, P0);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	gsl_vector *w = gsl_vector_alloc(processNoiseDim);
	gsl_vector_set_zero(w);
	gsl_vector *v = gsl_vector_alloc(observationNoiseDim);
	gsl_vector_set_zero(v);
	gsl_matrix *Q = gsl_matrix_alloc(processNoiseDim, processNoiseDim);
	gsl_matrix_set_identity(Q);
	gsl_matrix_scale(Q, sigma_w*sigma_w);
	gsl_matrix *R = gsl_matrix_alloc(observationNoiseDim, observationNoiseDim);
	gsl_matrix_set_identity(R);
	gsl_matrix_scale(R, sigma_v*sigma_v);

	gsl_vector *actualMeasurement = gsl_vector_alloc(outputDim);

	//
	const size_t Nstep = 20;
	std::vector<double> actualPos, actualVel, actualAccel;
	std::vector<double> estPos_m, estVel_m, estAccel_m;
	std::vector<double> estPos, estVel, estAccel;
	std::vector<double> posGain, velGain, accelGain;
	std::vector<double> posErrVar, velErrVar, accelErrVar;
	actualPos.reserve(Nstep);
	actualVel.reserve(Nstep);
	actualAccel.reserve(Nstep);
	estPos_m.reserve(Nstep);
	estVel_m.reserve(Nstep);
	estAccel_m.reserve(Nstep);
	estPos.reserve(Nstep);
	estVel.reserve(Nstep);
	estAccel.reserve(Nstep);
	posGain.reserve(Nstep);
	velGain.reserve(Nstep);
	accelGain.reserve(Nstep);
	posErrVar.reserve(Nstep * 2);
	velErrVar.reserve(Nstep * 2);
	accelErrVar.reserve(Nstep * 2);

	// method #1
	// 1-based time step. 0-th time step is initial
	size_t step = 0;
	while (step < Nstep)
	{
		actualPos.push_back(gsl_vector_get(actualState, 0));
		actualVel.push_back(gsl_vector_get(actualState, 1));
		actualAccel.push_back(gsl_vector_get(actualState, 2));

		// 0. initial estimates: x(0) & P(0)

		// 1. unscented transformation
		filter.performUnscentedTransformation();

		// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const bool retval1 = filter.updateTime(step, NULL, Q);
		assert(retval1);

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			estPos_m.push_back(gsl_vector_get(x_hat, 0));
			estVel_m.push_back(gsl_vector_get(x_hat, 1));
			estAccel_m.push_back(gsl_vector_get(x_hat, 2));
			posErrVar.push_back(gsl_matrix_get(P, 0, 0));
			velErrVar.push_back(gsl_matrix_get(P, 1, 1));
			accelErrVar.push_back(gsl_matrix_get(P, 2, 2));
		}

		// advance time step
		++step;

		// 3. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const gsl_vector *y_tilde = system.evaluateMeasurementEquation(step, actualState, NULL, NULL);
		gsl_vector_set(actualMeasurement, 0, gsl_vector_get(y_tilde, 0) + generator2());
		const bool retval2 = filter.updateMeasurement(step, actualMeasurement, NULL, R);
		assert(retval2);

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *K = filter.getKalmanGain();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			estPos.push_back(gsl_vector_get(x_hat, 0));
			estVel.push_back(gsl_vector_get(x_hat, 1));
			estAccel.push_back(gsl_vector_get(x_hat, 2));
			posGain.push_back(gsl_matrix_get(K, 0, 0));
			velGain.push_back(gsl_matrix_get(K, 1, 1));
			accelGain.push_back(gsl_matrix_get(K, 2, 2));
			posErrVar.push_back(gsl_matrix_get(P, 0, 0));
			velErrVar.push_back(gsl_matrix_get(P, 1, 1));
			accelErrVar.push_back(gsl_matrix_get(P, 2, 2));
		}

		//
		const gsl_vector *ff = system.evaluatePlantEquation(step, actualState, NULL, NULL);
		gsl_vector_set(actualState, 0, gsl_vector_get(ff, 0) + generator1());
		gsl_vector_set(actualState, 1, gsl_vector_get(ff, 1) + generator1());
		gsl_vector_set(actualState, 2, gsl_vector_get(ff, 2) + generator1());
	}

	gsl_vector_free(actualState);  actualState = NULL;
	gsl_vector_free(actualMeasurement);  actualMeasurement = NULL;
	gsl_vector_free(w);  w = NULL;
	gsl_vector_free(v);  v = NULL;
	gsl_matrix_free(Q);  Q = NULL;
	gsl_matrix_free(R);  R = NULL;

	//
	std::ofstream stream("../data/unscented_kalman_filter_with_additive_noise.dat", std::ios::out | std::ios::trunc);
	if (stream)
	{
		output_data_to_file(stream, "actualPos", actualPos);
		output_data_to_file(stream, "actualVel", actualVel);
		output_data_to_file(stream, "actualAccel", actualAccel);
		output_data_to_file(stream, "estPos", estPos);
		output_data_to_file(stream, "estVel", estVel);
		output_data_to_file(stream, "estAccel", estAccel);

		stream.close();
	}
}

void nonstationary_growth_system_unscented_kalman_filter_with_additive_noise()
{
	const size_t stateDim = 1;
	const size_t inputDim = 1;
	const size_t outputDim = 1;
	const size_t processNoiseDim = stateDim;
	const size_t observationNoiseDim = outputDim;

#if 0
	// for unscented Kalman filter
	const size_t L = stateDim + processNoiseDim + observationNoiseDim;
#else
	// for unscented Kalman filter with additive noise
	const size_t L = stateDim;
#endif
	const double alpha = 1.0e-3;
	const double beta = 2.0;  // for normal distribution
	const double kappa = 0.0;  // 3.0 - L;
	const double x_init = 0.1;
	const double sigma_init = std::sqrt(2.0);

	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_vector_set_all(x0, x_init);
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(P0);
	gsl_matrix_scale(P0, sigma_init);

	const NonstationaryGrowthSystem system(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim);
	swl::UnscentedKalmanFilterWithAdditiveNoise filter(system, alpha, beta, kappa, x0, P0);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	gsl_vector *w = gsl_vector_alloc(processNoiseDim);
	gsl_vector_set_zero(w);
	gsl_vector *v = gsl_vector_alloc(observationNoiseDim);
	gsl_vector_set_zero(v);
	gsl_matrix *Q = gsl_matrix_alloc(processNoiseDim, processNoiseDim);
	gsl_matrix_set_identity(Q);
	gsl_matrix_scale(Q, NonstationaryGrowthSystem::Rw);
	gsl_matrix *R = gsl_matrix_alloc(observationNoiseDim, observationNoiseDim);
	gsl_matrix_set_identity(R);
	gsl_matrix_scale(R, NonstationaryGrowthSystem::Rv);

	//
	const size_t Nstep = 100;
	std::vector<double> state;
	std::vector<double> gain;
	std::vector<double> errVar;
	state.reserve(Nstep * 2);
	gain.reserve(Nstep);
	errVar.reserve(Nstep * 2);

	// method #1
	// 1-based time step. 0-th time step is initial
	size_t step = 0;
	while (step < Nstep)
	{
		// 0. initial estimates: x(0) & P(0)

		// 1. unscented transformation
		filter.performUnscentedTransformation();

		// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const bool retval1 = filter.updateTime(step, NULL, Q);
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

		// 3. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const gsl_vector *actualMeasurement = system.simulateMeasurement(step, filter.getEstimatedState());
		const bool retval2 = filter.updateMeasurement(step, actualMeasurement, NULL, R);
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

	gsl_vector_free(w);  w = NULL;
	gsl_vector_free(v);  v = NULL;
	gsl_matrix_free(Q);  Q = NULL;
	gsl_matrix_free(R);  R = NULL;

	//
	std::ofstream stream("../data/unscented_kalman_filter_with_additive_noise.dat", std::ios::out | std::ios::trunc);
	if (stream)
	{
		output_data_to_file(stream, "state", state);
		output_data_to_file(stream, "gain", gain);
		output_data_to_file(stream, "errVar", errVar);

		stream.close();
	}
}

}  // namespace local
}  // unnamed namespace

void unscented_kalman_filter_with_additive_noise()
{
	//local::simple_system_unscented_kalman_filter_with_additive_noise();
	local::linear_mass_spring_damper_system_unscented_kalman_filter_with_additive_noise();
	//local::simple_nonlinear_system_unscented_kalman_filter_with_additive_noise();
	//nonstationary_growth_system_unscented_kalman_filter_with_additive_noise();
}
