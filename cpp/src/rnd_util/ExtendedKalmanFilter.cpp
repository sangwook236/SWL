#include "swl/Config.h"
#include "swl/rnd_util/ExtendedKalmanFilter.h"
#include "swl/rnd_util/DiscreteNonlinearStochasticSystem.h"
#include "swl/rnd_util/ContinuousNonlinearStochasticSystem.h"
#include <gsl/gsl_linalg.h>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-------------------------------------------------------------------------
// the extended Kalman filter for the discrete nonlinear stochastic system

// x(k+1) = f(k, x(k), u(k), w(k))
// y(k) = h(k, x(k), u(k), v(k))
// where E[w(k)] = E[v(k)] = 0, Q(k) = E[w(k) * w(k)^T], R(k) = E[v(k) * v(k)^T], N(k) = E[w(k) * v(k)^T]
//
// currently, this code is implemented only for N(k) = 0
// without loss of generality, N(k) = E[w(k) * v(k)^T] can be transformed into N(k) = E[w'(k) * v(k)^T] = 0
//	[ref] "Kalman Filtering and Neural Networks", Simon Haykin, ch. 6, pp. 206

// ***** method #1
// 0. initial estimates: x(0) & P(0)
// 1. time update (prediction): x(k-1) & P(k-1)  ==>  x-(k) & P-(k)
// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
//	==> result: posterior estimates, x(k) & P(k), conditioned on all available measurements at time k
//	==> 1-based time step. 0-th time step is initial

// ***** method #2
// 0. initial estimates: x-(0) & P-(0)
// 1. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
//	==> result: prior estimates, x-(k+1) & P-(k+1), conditioned on all prior measurements except the one at time k+1
//	==> 0-based time step. 0-th time step is initial

DiscreteExtendedKalmanFilter::DiscreteExtendedKalmanFilter(const DiscreteNonlinearStochasticSystem &system, const gsl_vector *x0, const gsl_matrix *P0)
: system_(system), x_hat_(NULL), /*y_hat_(NULL),*/ P_(NULL), K_(NULL),
  residual_(NULL), RR_(NULL), invRR_(NULL), PCt_(NULL), permutation_(NULL), v_(NULL), M_(NULL)//, M2_(NULL)
{
	const size_t &stateDim = system_.getStateDim();
	const size_t &inputDim = system_.getInputDim();
	const size_t &outputDim = system_.getOutputDim();

	if (x0 && P0 && stateDim && inputDim && outputDim &&
		stateDim == x0->size && stateDim == P0->size1 && stateDim == P0->size2)
	{
		x_hat_ = gsl_vector_alloc(stateDim);
		//y_hat_ = gsl_vector_alloc(outputDim);
		P_ = gsl_matrix_alloc(stateDim, stateDim);
		K_ = gsl_matrix_alloc(stateDim, outputDim);

		residual_ = gsl_vector_alloc(outputDim);
		RR_ = gsl_matrix_alloc(outputDim, outputDim);
		invRR_ = gsl_matrix_alloc(outputDim, outputDim);
		PCt_ = gsl_matrix_alloc(stateDim, outputDim);
		permutation_ = gsl_permutation_alloc(outputDim);

		v_ = gsl_vector_alloc(stateDim);
		M_ = gsl_matrix_alloc(stateDim, stateDim);
		M2_ = gsl_matrix_alloc(stateDim, stateDim);

		gsl_vector_memcpy(x_hat_, x0);
		//gsl_vector_set_zero(y_hat_);
		gsl_matrix_memcpy(P_, P0);
		//gsl_matrix_set_identity(K_);
	}
}

DiscreteExtendedKalmanFilter::~DiscreteExtendedKalmanFilter()
{
	gsl_vector_free(x_hat_);  x_hat_ = NULL;
	//gsl_vector_free(y_hat_);  y_hat_ = NULL;
	gsl_matrix_free(P_);  P_ = NULL;
	gsl_matrix_free(K_);  K_ = NULL;

	gsl_vector_free(residual_);  residual_ = NULL;
	gsl_matrix_free(RR_);  RR_ = NULL;
	gsl_matrix_free(invRR_);  invRR_ = NULL;
	gsl_matrix_free(PCt_);  PCt_ = NULL;
	gsl_permutation_free(permutation_);  permutation_ = NULL;

	gsl_vector_free(v_);  v_ = NULL;
	gsl_matrix_free(M_);  M_ = NULL;
	gsl_matrix_free(M2_);  M2_ = NULL;
}

// time update (prediction)
bool DiscreteExtendedKalmanFilter::updateTime(const size_t step, const gsl_vector *input)
{
	if (!x_hat_ || /*!y_hat_ ||*/ !P_ || !K_) return false;

	const gsl_vector *f_eval = system_.evaluatePlantEquation(step, x_hat_, input, NULL);  // f = f(k, x(k), u(k), 0)

	const gsl_matrix *Phi = system_.getStateTransitionMatrix(step, x_hat_);  // Phi(k) = exp(A(k) * T) where A(k) = df(k, x(k), u(k), 0)/dx
#if 0
	const gsl_matrix *W = system_.getProcessNoiseCouplingMatrix(step);  // W(k) = df(k, x(k), u(k), 0)/dw
	const gsl_matrix *Q = system_.getProcessNoiseCovarianceMatrix(step);  // Q(k)
#else
	const gsl_matrix *Qd = system_.getProcessNoiseCovarianceMatrix(step);  // Qd(k) = W * Q(k) * W(k)^T
#endif
	if (!Phi || !Qd || !f_eval) return false;

	// 1. propagate time
	// x-(k+1) = f(k, x(k), u(k), 0)
	gsl_vector_memcpy(x_hat_, f_eval);

	// P-(k+1) = Phi(k) * P(k) * Phi(k)^T + Qd(k) where Phi(k) = exp(A * T), A = df(k, x(k), u(k), 0)/dx, Qd(k) = W(k) * Q(k) * W(k)^T, W(k) = df(k, x(k), u(k), 0)/dw
#if 0
	// using Q
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Phi, P_, 0.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, M_, Phi, 0.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, W, Qd, 0.0, M2_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, M2_, W, 0.0, P_) ||
		GSL_SUCCESS != gsl_matrix_add(P_, M_))
		return false;
#else
	// using Qd
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Phi, P_, 0.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, M_, Phi, 0.0, P_) ||
		GSL_SUCCESS != gsl_matrix_add(P_, Qd))
		return false;
#endif

	// preserve symmetry of P
	gsl_matrix_transpose_memcpy(M_, P_);
	gsl_matrix_add(P_, M_);
	gsl_matrix_scale(P_, 0.5);

	return true;
}

// measurement update (correction)
bool DiscreteExtendedKalmanFilter::updateMeasurement(const size_t step, const gsl_vector *actualMeasurement, const gsl_vector *input)
{
	if (!x_hat_ || /*!y_hat_ ||*/ !P_ || !K_) return false;

	const gsl_vector *h_eval = system_.evaluateMeasurementEquation(step, x_hat_, input, NULL);  // h = h(k, x(k), u(k), 0)

	const gsl_matrix *Cd = system_.getOutputMatrix(step, x_hat_);  // Cd(k) = dh(k, x-(k), u(k), 0)/dx
#if 0
	const gsl_matrix *V = system_.getMeasurementNoiseCouplingMatrix(step);  // V(k) = dh(k, x-(k), u(k), 0)/dv
	const gsl_matrix *R = system_.getMeasurementNoiseCovarianceMatrix(step);  // R(k)
#else
	const gsl_matrix *Rd = system_.getMeasurementNoiseCovarianceMatrix(step);  // Rd(k) = V(k) * R(k) * V(k)^T
#endif
	if (!Cd || !Rd || !h_eval || !actualMeasurement) return false;

	// 1. calculate Kalman gain: K(k) = P-(k) * Cd(k)^T * (Cd(k) * P-(k) * Cd(k)^T + Rd(k))^-1 where Cd(k) = dh(k, x-(k), u(k), 0)/dx, Rd(k) = V(k) * R(k) * V(k)^T, V(k) = dh(k, x-(k), u(k), 0)/dv
	// inverse of matrix using LU decomposition
	gsl_matrix_memcpy(RR_, Rd);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, P_, Cd, 0.0, PCt_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Cd, PCt_, 1.0, RR_))
		return false;

	int signum;
	if (GSL_SUCCESS != gsl_linalg_LU_decomp(RR_, permutation_, &signum) ||
		GSL_SUCCESS != gsl_linalg_LU_invert(RR_, permutation_, invRR_))
		return false;

	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, PCt_, invRR_, 0.0, K_))  // calculate Kalman gain
		return false;

	// 2. update measurement: x(k) = x-(k) + K(k) * (y_tilde(k) - y_hat(k)) where y_hat(k) = h(k, x-(k), u(k), 0)
#if 0
	// save an estimated measurement, y_hat
	gsl_vector_memcpy(y_hat_, h_eval);
	gsl_vector_memcpy(residual_, y_hat_);
	if (GSL_SUCCESS != gsl_vector_sub(residual_, actualMeasurement) ||  // calculate residual = y_tilde(k) - y_hat(k)
		GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, -1.0, K_, residual_, 1.0, x_hat_))  // calculate x_hat(k)
		return false;
#else
	gsl_vector_memcpy(residual_, h_eval);
	if (GSL_SUCCESS != gsl_vector_sub(residual_, actualMeasurement) ||  // calculate residual = y_tilde(k) - y_hat(k)
		GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, -1.0, K_, residual_, 1.0, x_hat_))  // calculate x_hat(k)
		return false;
#endif

	// 3. update covariance: P(k) = (I - K(k) * Cd(k)) * P-(k)
#if 0
	// not working
	gsl_matrix_set_identity(M_);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, K_, Cd, 1.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, M_, P_, 0.0, P_))
		return false;
#else
	gsl_matrix_set_identity(M_);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, K_, Cd, 1.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, M_, P_, 0.0, M2_))
		return false;
	gsl_matrix_memcpy(P_, M2_);
#endif

	// preserve symmetry of P
	gsl_matrix_transpose_memcpy(M_, P_);
	gsl_matrix_add(P_, M_);
	gsl_matrix_scale(P_, 0.5);

	return true;
}

//-------------------------------------------------------------------------
// the extended Kalman filter for the continuous nonlinear stochastic system

// dx(t)/dt = f(t, x(t), u(t), w(t))
// y(t) = h(t, x(t), u(t), v(t))
// where E[w(t)] = E[v(t)] = 0, Q(t) = E[w(t) * w(t)^T], R(t) = E[v(t) * v(t)^T], N(t) = E[w(t) * v(t)^T]
//
// currently, this code is implemented only for N(t) = 0
// without loss of generality, N(t) = E[w(t) * v(t)^T] can be transformed into N(t) = E[w'(t) * v(t)^T] = 0
//	[ref] "Kalman Filtering and Neural Networks", Simon Haykin, ch. 6, pp. 206

// ***** method #1
// 0. initial estimates: x(0) & P(0)
// 1. time update (prediction): x(k-1) & P(k-1)  ==>  x-(k) & P-(k)
// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
//	==> result: posterior estimates, x(k) & P(k), conditioned on all available measurements at time k
//	==> 1-based time step. 0-th time step is initial

// ***** method #2
// 0. initial estimates: x-(0) & P-(0)
// 1. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
//	==> result: prior estimates, x-(k+1) & P-(k+1), conditioned on all prior measurements except the one at time k+1
//	==> 0-based time step. 0-th time step is initial

ContinuousExtendedKalmanFilter::ContinuousExtendedKalmanFilter(const ContinuousNonlinearStochasticSystem &system, const gsl_vector *x0, const gsl_matrix *P0)
: system_(system), x_hat_(NULL), /*y_hat_(NULL),*/ P_(NULL), K_(NULL)
{
	const size_t &stateDim = system_.getStateDim();
	const size_t &inputDim = system_.getInputDim();
	const size_t &outputDim = system_.getOutputDim();

	if (x0 && P0 && stateDim && inputDim && outputDim &&
		stateDim == x0->size && stateDim == P0->size1 && stateDim == P0->size2)
	{
		x_hat_ = gsl_vector_alloc(stateDim);
		//y_hat_ = gsl_vector_alloc(outputDim);
		P_ = gsl_matrix_alloc(stateDim, stateDim);
		K_ = gsl_matrix_alloc(stateDim, outputDim);

		gsl_vector_memcpy(x_hat_, x0);
		//gsl_vector_set_zero(y_hat_);
		gsl_matrix_memcpy(P_, P0);
		//gsl_matrix_set_identity(K_);
	}
}

ContinuousExtendedKalmanFilter::~ContinuousExtendedKalmanFilter()
{
	gsl_vector_free(x_hat_);  x_hat_ = NULL;
	//gsl_vector_free(y_hat_);  y_hat_ = NULL;
	gsl_matrix_free(P_);  P_ = NULL;
	gsl_matrix_free(K_);  K_ = NULL;
}

// time update (prediction)
bool ContinuousExtendedKalmanFilter::updateTime(const double time, const gsl_vector *input)
{
#if 0
	if (!x_hat_ || /*!y_hat_ ||*/ !P_ || !K_) return false;

	const gsl_vector *f_eval = system_.evaluatePlantEquation(time, x_hat_, input, NULL);  // f = f(t, x(t), u(t), w(t))

	const gsl_matrix *A = doGetStateTransitionMatrix(time, x_hat_);  // A(t) = df(t, x(t), u(t), 0)/dx
#if 0
	const gsl_matrix *W = doGetProcessNoiseCouplingMatrix(time);  // W(t) = df(t, x(t), u(t), 0)/dw
	const gsl_matrix *Q = doGetProcessNoiseCovarianceMatrix(time);  // Q(t)
#else
	const gsl_matrix *Qd = doGetProcessNoiseCovarianceMatrix(time);  // Qd(t) = W * Q(t) * W(t)^T
#endif
	if (!A || !Qd || !f_eval) return false;

	// 1. Propagate time.
	// dx(t)/dt = f(t, x(t), u(t), 0)
	// dP(t)/dt = A(t) * P(t) + P(t) * A(t)^T + Qd(t) where A(t) = df(t, x(t), u(t), 0)/dx, Qd(t) = W(t) * Q(t) * W(t)^T, W(t) = df(t, x(t), u(t), 0)/dw

	// Preserve symmetry of P.
	gsl_matrix_transpose_memcpy(M_, P_);
	gsl_matrix_add(P_, M_);
	gsl_matrix_scale(P_, 0.5);

	return true;
#else
	throw std::runtime_error("Not yet implemented");
#endif
}

// Measurement update (correction).
bool ContinuousExtendedKalmanFilter::updateMeasurement(const double time, const gsl_vector *actualMeasurement, const gsl_vector *input)
{
#if 0
	if (!x_hat_ || /*!y_hat_ ||*/ !P_ || !K_) return false;

	const gsl_vector *h_eval = system_.evaluateMeasurementEquation(step, x_hat_, input, NULL);  // h = h(t, x(t), u(t), v(t))

	const gsl_matrix *C = doGetOutputMatrix(time, x_hat_);  // C(t) = dh(t, x(t), u(t), 0)/dx
#if 0
	const gsl_matrix *V = doGetMeasurementNoiseCouplingMatrix(time);  // V(t) = dh(t, x(t), u(t), 0)/dv
	const gsl_matrix *R = doGetMeasurementNoiseCovarianceMatrix(time);  // R(t)
#else
	const gsl_matrix *Rd = doGetMeasurementNoiseCovarianceMatrix(time);  // Rd(t) = V(t) * R(t) * V(t)^T
#endif
	if (!C || !Rd || !h_eval || !actualMeasurement) return false;

	// 1. calculate Kalman gain: K(t) = P(t) * C(t)^T * Rd(t)^-1 where C(t) = dh(t, x(t), u(t), 0)/dx, Rd(t) = V(t) * R(t) * V(t)^T, V(t) = dh(t, x(t), u(t), 0)/dv
	// 2. update measurement: dx(t)/dt = f(t, x(t), u(t), 0) + K(t) * (y_tilde(t) - y_hat(t)) where y_hat(t) = h(t, x(t), u(t), 0) ???
	// 3. update covariance:
	//	dP(t)/dt = A(t) * P(t) + P(t) * A(t)^T + Qd(t) - P(t) * C(t)^T * Rd(t)^-1 * C(t) * P(t) : the matrix Riccati differential equation
	//	         = A(t) * P(t) + P(t) * A(t)^T + Qd(t) - K(t) * Rd(t) * K(t)^T

	// preserve symmetry of P
	gsl_matrix_transpose_memcpy(M_, P_);
	gsl_matrix_add(P_, M_);
	gsl_matrix_scale(P_, 0.5);

	return true;
#else
	throw std::runtime_error("Not yet implemented");
#endif
}

}  // namespace swl
