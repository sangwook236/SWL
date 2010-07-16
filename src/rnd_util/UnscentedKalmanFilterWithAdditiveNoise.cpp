#include "swl/Config.h"
#include "swl/rnd_util/UnscentedKalmanFilterWithAdditiveNoise.h"
#include "swl/rnd_util/DiscreteNonlinearStochasticSystem.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <iostream>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-------------------------------------------------------------------------
// the unscented Kalman filter for the discrete nonlinear stochastic system

// x(k+1) = f(k, x(k), u(k)) + w(k)
// y(k) = h(k, x(k), u(k)) + v(k)
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

UnscentedKalmanFilterWithAdditiveNoise::UnscentedKalmanFilterWithAdditiveNoise(const DiscreteNonlinearStochasticSystem &system, const double alpha, const double beta, const double kappa, const gsl_vector *x0, const gsl_matrix *P0)
: system_(system), alpha_(alpha), beta_(beta), kappa_(kappa),
  L_(system_.getStateDim()),
  lambda_(alpha * alpha * (L_ + kappa) - L_), sigmaDim_(2 * L_ + 1), gamma_(std::sqrt(L_ + lambda_)),
  x_hat_(NULL), y_hat_(NULL), P_(NULL), K_(NULL),
  Chi_star_(NULL), Chi_(NULL), Upsilon_(NULL), Pyy_(NULL), Pxy_(NULL),
  Wm0_(lambda_ / (L_ + lambda_)), Wc0_(1.0 - alpha*alpha + beta + lambda_ / (L_ + lambda_)), Wi_(0.5 / (L_ + lambda_)),
  x_tmp_(NULL), y_tmp_(NULL), P_tmp_(NULL), Pyy_tmp_(NULL), invPyy_(NULL), KPyy_tmp_(NULL), permutation_(NULL)
{
	const size_t &stateDim = system_.getStateDim();
	const size_t &inputDim = system_.getInputDim();
	const size_t &outputDim = system_.getOutputDim();

	if (x0 && P0 && stateDim && inputDim && outputDim &&
		stateDim == x0->size && stateDim == P0->size1 && stateDim == P0->size2)
	{
		x_hat_ = gsl_vector_alloc(stateDim);
		y_hat_ = gsl_vector_alloc(outputDim);
		P_ = gsl_matrix_alloc(stateDim, stateDim);
		K_ = gsl_matrix_alloc(stateDim, outputDim);

		Chi_star_ = gsl_matrix_alloc(L_, sigmaDim_);
		Chi_ = gsl_matrix_alloc(L_, sigmaDim_);
		Upsilon_ = gsl_matrix_alloc(outputDim, sigmaDim_);

		Pyy_ = gsl_matrix_alloc(outputDim, outputDim);
		Pxy_ = gsl_matrix_alloc(stateDim, outputDim);

		//
		x_tmp_ = gsl_vector_alloc(stateDim);
		y_tmp_ = gsl_vector_alloc(outputDim);
		P_tmp_ = gsl_matrix_alloc(stateDim, stateDim);
		Pyy_tmp_ = gsl_matrix_alloc(outputDim, outputDim);
		invPyy_ = gsl_matrix_alloc(outputDim, outputDim);
		KPyy_tmp_ = gsl_matrix_alloc(stateDim, outputDim);

		permutation_ = gsl_permutation_alloc(outputDim);

		gsl_vector_memcpy(x_hat_, x0);
		//gsl_vector_set_zero(y_hat_);
		gsl_matrix_memcpy(P_, P0);
		//gsl_matrix_set_identity(K_);

		gsl_matrix_set_zero(Chi_);
	}
}

UnscentedKalmanFilterWithAdditiveNoise::~UnscentedKalmanFilterWithAdditiveNoise()
{
	gsl_vector_free(x_hat_);  x_hat_ = NULL;
	gsl_vector_free(y_hat_);  y_hat_ = NULL;
	gsl_matrix_free(P_);  P_ = NULL;
	gsl_matrix_free(K_);  K_ = NULL;

	gsl_matrix_free(Chi_star_);  Chi_star_ = NULL;
	gsl_matrix_free(Chi_);  Chi_ = NULL;
	gsl_matrix_free(Upsilon_);  Upsilon_ = NULL;

	gsl_matrix_free(Pyy_);  Pyy_ = NULL;
	gsl_matrix_free(Pxy_);  Pxy_ = NULL;

	//
	gsl_vector_free(x_tmp_);  x_tmp_ = NULL;
	gsl_vector_free(y_tmp_);  y_tmp_ = NULL;
	gsl_matrix_free(P_tmp_);  P_tmp_ = NULL;
	gsl_matrix_free(Pyy_tmp_);  Pyy_tmp_ = NULL;
	gsl_matrix_free(invPyy_);  invPyy_ = NULL;
	gsl_matrix_free(KPyy_tmp_);  KPyy_tmp_ = NULL;

	gsl_permutation_free(permutation_);  permutation_ = NULL;
}

//
bool UnscentedKalmanFilterWithAdditiveNoise::performUnscentedTransformation()
{
	if (!x_hat_ || !P_ || !Chi_) return false;

#if 0
	// Choleskcy decomposition: A = L * L^T
	gsl_linalg_cholesky_decomp(P_);
	for (size_t k = 1; k < L_; ++k)
		gsl_vector_set_zero(&gsl_matrix_superdiagonal(P_, k).vector);
#else
	// A^1/2 = V * D^1/2 * V^-1  <==  f(A) = V * f(J) * V^-1
	// if symmetric matrix, A^1/2 = V * D^1/2 * V^T

	// FIXME [modify] >> efficiency
	gsl_vector* eval = gsl_vector_alloc(L_);
	gsl_matrix* evec = gsl_matrix_alloc(L_, L_);
	gsl_eigen_symmv_workspace* ew = gsl_eigen_symmv_alloc(L_);

	gsl_eigen_symmv(P_, eval, evec, ew);

	for (size_t i = 0; i < eval->size; ++i)
		gsl_vector_set(eval, i, std::sqrt(gsl_vector_get(eval, i)));

	gsl_matrix_set_zero(P_);
	gsl_vector_memcpy(&gsl_matrix_diagonal(P_).vector, eval);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, evec, P_, 0.0, P_tmp_))
		return false;

	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, P_tmp_, evec, 0.0, P_))
		return false;

	gsl_eigen_symmv_free(ew);  ew = NULL;
	gsl_vector_free(eval);  eval = NULL;
	gsl_matrix_free(evec);  evec = NULL;
#endif

	gsl_matrix_scale(P_, gamma_);

	gsl_vector_memcpy(&gsl_matrix_column(Chi_, 0).vector, x_hat_);
	for (size_t i = 0; i < L_; ++i)
	{
		const gsl_vector *pp = &gsl_matrix_column(P_, i).vector;

		gsl_vector_memcpy(x_tmp_, x_hat_);
		gsl_vector_add(x_tmp_, pp);
		gsl_vector_memcpy(&gsl_matrix_column(Chi_, i + 1).vector, x_tmp_);

		gsl_vector_memcpy(x_tmp_, x_hat_);
		gsl_vector_sub(x_tmp_, pp);
		gsl_vector_memcpy(&gsl_matrix_column(Chi_, L_ + i + 1).vector, x_tmp_);
	}

	return true;
}

// time update (prediction)
bool UnscentedKalmanFilterWithAdditiveNoise::updateTime(const size_t step, const gsl_vector *input, const gsl_matrix *Q)
{
	if (!x_hat_ || !P_ || !Chi_star_ || !Chi_) return false;

	// propagate time
	// x-(k)
	gsl_vector_set_zero(x_hat_);
	for (size_t i = 0; i < sigmaDim_; ++i)
	{
		const gsl_vector *xx = &gsl_matrix_column(Chi_, i).vector;
		const gsl_vector *f_eval = system_.evaluatePlantEquation(step, xx, input, NULL);  // f = f(k, x(k), u(k), 0)
		gsl_vector_memcpy(&gsl_matrix_column(Chi_star_, i).vector, f_eval);

		// y = a x + y
		gsl_blas_daxpy((0 == i ? Wm0_ : Wi_), f_eval, x_hat_);
	}

	// P-(k)
	gsl_matrix_set_zero(P_);
	for (size_t i = 0; i < sigmaDim_; ++i)
	{
		const gsl_vector *xx = &gsl_matrix_column(Chi_star_, i).vector;

		gsl_vector_memcpy(x_tmp_, xx);
		gsl_vector_sub(x_tmp_, x_hat_);

		// C = a op(A) op(B) + C
		const gsl_matrix *XX = &gsl_matrix_view_vector(x_tmp_, x_tmp_->size, 1).matrix;
		if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, (0 == i ? Wc0_ : Wi_), XX, XX, 1.0, P_))
			return false;
	}

	gsl_matrix_add(P_, Q);

	//
	gsl_matrix_memcpy(P_tmp_, Q);

#if 0
	// Choleskcy decomposition: A = L * L^T
	gsl_linalg_cholesky_decomp(P_tmp_);
	for (size_t k = 1; k < L_; ++k)
		gsl_vector_set_zero(&gsl_matrix_superdiagonal(P_tmp_, k).vector);
#else
	// A^1/2 = V * D^1/2 * V^-1  <==  f(A) = V * f(J) * V^-1
	// if symmetric matrix, A^1/2 = V * D^1/2 * V^T

	// FIXME [modify] >> efficiency
	gsl_matrix* PP_tmp = gsl_matrix_alloc(L_, L_);
	gsl_vector* eval = gsl_vector_alloc(L_);
	gsl_matrix* evec = gsl_matrix_alloc(L_, L_);
	gsl_eigen_symmv_workspace* ew = gsl_eigen_symmv_alloc(L_);

	gsl_eigen_symmv(P_tmp_, eval, evec, ew);

	for (size_t i = 0; i < eval->size; ++i)
		gsl_vector_set(eval, i, std::sqrt(gsl_vector_get(eval, i)));

	gsl_matrix_set_zero(P_tmp_);
	gsl_vector_memcpy(&gsl_matrix_diagonal(P_tmp_).vector, eval);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, evec, P_tmp_, 0.0, PP_tmp))
		return false;

	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, PP_tmp, evec, 0.0, P_tmp_))
		return false;

	gsl_eigen_symmv_free(ew);  ew = NULL;
	gsl_vector_free(eval);  eval = NULL;
	gsl_matrix_free(evec);  evec = NULL;
	gsl_matrix_free(PP_tmp);  PP_tmp = NULL;
#endif

	gsl_matrix_scale(P_tmp_, gamma_);  // gamma * sqrt(Q)

	// FIXME [check] >>
	const gsl_vector *xs0 = &gsl_matrix_column(Chi_star_, 0).vector;

	gsl_vector_memcpy(&gsl_matrix_column(Chi_, 0).vector, xs0);
	for (size_t i = 0; i < L_; ++i)
	{
		const gsl_vector *pp = &gsl_matrix_column(P_tmp_, i).vector;

		gsl_vector_memcpy(x_tmp_, xs0);
		gsl_vector_add(x_tmp_, pp);
		gsl_vector_memcpy(&gsl_matrix_column(Chi_, i + 1).vector, x_tmp_);

		gsl_vector_memcpy(x_tmp_, xs0);
		gsl_vector_sub(x_tmp_, pp);
		gsl_vector_memcpy(&gsl_matrix_column(Chi_, L_ + i + 1).vector, x_tmp_);
	}

	// preserve symmetry of P
	gsl_matrix_transpose_memcpy(P_tmp_, P_);
	gsl_matrix_add(P_, P_tmp_);
	gsl_matrix_scale(P_, 0.5);

	return true;
}

// measurement update (correction)
bool UnscentedKalmanFilterWithAdditiveNoise::updateMeasurement(const size_t step, const gsl_vector *actualMeasurement, const gsl_vector *input, const gsl_matrix *R)
{
	if (!x_hat_ || !y_hat_ || !P_ || !K_ || !Chi_ || !Upsilon_ || !actualMeasurement) return false;

	// y-(k)
	gsl_vector_set_zero(y_hat_);
	for (size_t i = 0; i < sigmaDim_; ++i)
	{
		const gsl_vector *xx = &gsl_matrix_column(Chi_, i).vector;

		const gsl_vector *h_eval = system_.evaluateMeasurementEquation(step, xx, input, NULL);  // h = h(k, x(k), u(k), 0)
		gsl_vector_memcpy(&gsl_matrix_column(Upsilon_, i).vector, h_eval);

		// y = a x + y
		gsl_blas_daxpy((0 == i ? Wm0_ : Wi_), h_eval, y_hat_);
	}

	// Pyy & Pxy
	gsl_matrix_set_zero(Pyy_);
	gsl_matrix_set_zero(Pxy_);
	for (size_t i = 0; i < sigmaDim_; ++i)
	{
		const gsl_vector *yy = &gsl_matrix_column(Upsilon_, i).vector;
		const gsl_vector *xx = &gsl_matrix_column(Chi_, i).vector;

		gsl_vector_memcpy(y_tmp_, yy);
		gsl_vector_sub(y_tmp_, y_hat_);
		gsl_vector_memcpy(x_tmp_, xx);
		gsl_vector_sub(x_tmp_, x_hat_);

		// C = a op(A) op(B) + b C
		const gsl_matrix *YY = &gsl_matrix_view_vector(y_tmp_, y_tmp_->size, 1).matrix;
		const gsl_matrix *XX = &gsl_matrix_view_vector(x_tmp_, x_tmp_->size, 1).matrix;
		if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, (0 == i ? Wc0_ : Wi_), YY, YY, 1.0, Pyy_) ||
			GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, (0 == i ? Wc0_ : Wi_), XX, YY, 1.0, Pxy_))
			return false;
	}

	gsl_matrix_add(Pyy_, R);

	// Kalman gain: K(k) = Pxy * Pyy^-1
	// inverse of matrix using LU decomposition
	int signum;
	gsl_matrix_memcpy(Pyy_tmp_, Pyy_);
	if (GSL_SUCCESS != gsl_linalg_LU_decomp(Pyy_tmp_, permutation_, &signum) ||
		GSL_SUCCESS != gsl_linalg_LU_invert(Pyy_tmp_, permutation_, invPyy_))
		return false;

	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Pxy_, invPyy_, 0.0, K_))  // calculate Kalman gain
		return false;

	// update measurement: x(k) = x-(k) + K(k) * (y_tilde(k) - y_hat(k))
	gsl_vector_memcpy(y_tmp_, y_hat_);
	if (GSL_SUCCESS != gsl_vector_sub(y_tmp_, actualMeasurement) ||  // calculate residual = y_tilde(k) - y_hat(k)
		GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, -1.0, K_, y_tmp_, 1.0, x_hat_))  // calculate x_hat(k)
		return false;

	// update covariance: P(k) = P-(k) - K(k) * Pyy * K(k)^T
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, K_, Pyy_, 0.0, KPyy_tmp_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, KPyy_tmp_, K_, 0.0, P_tmp_))
		return false;
	gsl_matrix_sub(P_, P_tmp_);

	// preserve symmetry of P
	gsl_matrix_transpose_memcpy(P_tmp_, P_);
	gsl_matrix_add(P_, P_tmp_);
	gsl_matrix_scale(P_, 0.5);

	return true;
}

}  // namespace swl
