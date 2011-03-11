#include "swl/Config.h"
#include "swl/rnd_util/UnscentedKalmanFilter.h"
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

UnscentedKalmanFilter::UnscentedKalmanFilter(const DiscreteNonlinearStochasticSystem &system, const double alpha, const double beta, const double kappa, const gsl_vector *x0, const gsl_matrix *P0)
: system_(system), alpha_(alpha), beta_(beta), kappa_(kappa),
  L_(system_.getStateDim() + system_.getProcessNoiseDim() + system_.getObservationNoiseDim()),
  lambda_(alpha * alpha * (L_ + kappa) - L_), sigmaDim_(2 * L_ + 1), gamma_(std::sqrt(L_ + lambda_)),
  x_hat_(NULL), y_hat_(NULL), P_(NULL), K_(NULL),
  xa_(NULL), Chi_a_(NULL), P_a_(NULL), Chi_(NULL), Upsilon_(NULL), Pyy_(NULL), Pxy_(NULL),
  Wm0_(lambda_ / (L_ + lambda_)), Wc0_(1.0 - alpha*alpha + beta + lambda_ / (L_ + lambda_)), Wi_(0.5 / (L_ + lambda_)),
  xa_tmp_(NULL), x_tmp_(NULL), y_tmp_(NULL), P_tmp_(NULL), P_a_tmp_(NULL), Pyy_tmp_(NULL), invPyy_(NULL), KPyy_tmp_(NULL), permutation_(NULL),
  eigenVal_(NULL), eigenVec_(NULL), eigenWS_(NULL)
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

		xa_ = gsl_vector_alloc(L_);

		Chi_a_ = gsl_matrix_alloc(L_, sigmaDim_);
		P_a_ = gsl_matrix_alloc(L_, L_);
		Chi_ = gsl_matrix_alloc(stateDim, sigmaDim_);
		Upsilon_ = gsl_matrix_alloc(outputDim, sigmaDim_);

		Pyy_ = gsl_matrix_alloc(outputDim, outputDim);
		Pxy_ = gsl_matrix_alloc(stateDim, outputDim);

		//
		xa_tmp_ = gsl_vector_alloc(L_);
		x_tmp_ = gsl_vector_alloc(stateDim);
		y_tmp_ = gsl_vector_alloc(outputDim);
		P_tmp_ = gsl_matrix_alloc(stateDim, stateDim);
		P_a_tmp_ = gsl_matrix_alloc(L_, L_);
		Pyy_tmp_ = gsl_matrix_alloc(outputDim, outputDim);
		invPyy_ = gsl_matrix_alloc(outputDim, outputDim);
		KPyy_tmp_ = gsl_matrix_alloc(stateDim, outputDim);

		permutation_ = gsl_permutation_alloc(outputDim);

		eigenVal_ = gsl_vector_alloc(L_);
		eigenVec_ = gsl_matrix_alloc(L_, L_);
		eigenWS_ = gsl_eigen_symmv_alloc(L_);

		//
		gsl_vector_memcpy(x_hat_, x0);
		//gsl_vector_set_zero(y_hat_);
		gsl_matrix_memcpy(P_, P0);
		//gsl_matrix_set_identity(K_);

		gsl_matrix_set_zero(Chi_a_);
		gsl_matrix_set_zero(P_a_);
	}
}

UnscentedKalmanFilter::~UnscentedKalmanFilter()
{
	gsl_vector_free(x_hat_);  x_hat_ = NULL;
	gsl_vector_free(y_hat_);  y_hat_ = NULL;
	gsl_matrix_free(P_);  P_ = NULL;
	gsl_matrix_free(K_);  K_ = NULL;

	gsl_vector_free(xa_);  xa_ = NULL;

	gsl_matrix_free(Chi_a_);  Chi_a_ = NULL;
	gsl_matrix_free(P_a_);  P_a_ = NULL;
	gsl_matrix_free(Chi_);  Chi_ = NULL;
	gsl_matrix_free(Upsilon_);  Upsilon_ = NULL;

	gsl_matrix_free(Pyy_);  Pyy_ = NULL;
	gsl_matrix_free(Pxy_);  Pxy_ = NULL;

	//
	gsl_vector_free(xa_tmp_);  xa_tmp_ = NULL;
	gsl_vector_free(x_tmp_);  x_tmp_ = NULL;
	gsl_vector_free(y_tmp_);  y_tmp_ = NULL;
	gsl_matrix_free(P_tmp_);  P_tmp_ = NULL;
	gsl_matrix_free(P_a_tmp_);  P_a_tmp_ = NULL;
	gsl_matrix_free(Pyy_tmp_);  Pyy_tmp_ = NULL;
	gsl_matrix_free(invPyy_);  invPyy_ = NULL;
	gsl_matrix_free(KPyy_tmp_);  KPyy_tmp_ = NULL;

	gsl_permutation_free(permutation_);  permutation_ = NULL;

	gsl_vector_free(eigenVal_);  eigenVal_ = NULL;
	gsl_matrix_free(eigenVec_);  eigenVec_ = NULL;
	gsl_eigen_symmv_free(eigenWS_);  eigenWS_ = NULL;
}

//
bool UnscentedKalmanFilter::performUnscentedTransformation(const gsl_vector *w, const gsl_vector *v, const gsl_matrix *Q, const gsl_matrix *R)
{
	if (!x_hat_ || !P_ || !P_a_ || !Chi_a_ || !xa_) return false;

	const size_t &stateDim = system_.getStateDim();
	const size_t &processNoiseDim = system_.getProcessNoiseDim();
	const size_t &observationNoiseDim = system_.getObservationNoiseDim();

	gsl_vector *pp = NULL, *pa = NULL;

	gsl_matrix_set_zero(P_a_);
	gsl_matrix_memcpy(&gsl_matrix_submatrix(P_a_, 0, 0, stateDim, stateDim).matrix, P_);
	gsl_matrix_memcpy(&gsl_matrix_submatrix(P_a_, stateDim, stateDim, processNoiseDim, processNoiseDim).matrix, Q);
	gsl_matrix_memcpy(&gsl_matrix_submatrix(P_a_, stateDim + processNoiseDim, stateDim + processNoiseDim, observationNoiseDim, observationNoiseDim).matrix, R);

	// sqrt(P_a(k-1))
#if 0
	// use Cholesky decomposition: A = L * L^T
	gsl_linalg_cholesky_decomp(P_a_);
	// make lower triangular matrix
	for (size_t k = 1; k < L_; ++k)
		gsl_vector_set_zero(&gsl_matrix_superdiagonal(P_a_, k).vector);
#else
	// use singular value decomposition: A = U * S * V^T
	// A^1/2 = V * D^1/2 * V^-1  <==  f(A) = V * f(J) * V^-1
	// if symmetric matrix, A^1/2 = V * D^1/2 * V^T

	gsl_eigen_symmv(P_a_, eigenVal_, eigenVec_, eigenWS_);

	gsl_matrix_set_zero(P_a_);
	pp = &gsl_matrix_diagonal(P_a_).vector;
	for (size_t i = 0; i < eigenVal_->size; ++i)
		gsl_vector_set(pp, i, std::sqrt(gsl_vector_get(eigenVal_, i)));

	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, eigenVec_, P_a_, 0.0, P_a_tmp_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, P_a_tmp_, eigenVec_, 0.0, P_a_))
		return false;
#endif

	gsl_matrix_scale(P_a_, gamma_);  // gamma * sqrt(P_a(k-1))

	// Chi_a(k-1)
	// TODO [check] >> check if it is correct to use w & v or not
	gsl_vector_memcpy(&gsl_vector_subvector(xa_, 0, stateDim).vector, x_hat_);
	gsl_vector_memcpy(&gsl_vector_subvector(xa_, stateDim, processNoiseDim).vector, w);
	gsl_vector_memcpy(&gsl_vector_subvector(xa_, stateDim + processNoiseDim, observationNoiseDim).vector, v);

	gsl_vector_memcpy(&gsl_matrix_column(Chi_a_, 0).vector, xa_);
	for (size_t i = 1; i <= L_; ++i)
	{
		pa = &gsl_matrix_column(P_a_, i - 1).vector;

		gsl_vector_memcpy(xa_tmp_, xa_);
		gsl_vector_add(xa_tmp_, pa);
		gsl_vector_memcpy(&gsl_matrix_column(Chi_a_, i).vector, xa_tmp_);

		gsl_vector_memcpy(xa_tmp_, xa_);
		gsl_vector_sub(xa_tmp_, pa);
		gsl_vector_memcpy(&gsl_matrix_column(Chi_a_, L_ + i).vector, xa_tmp_);
	}

	return true;
}

// time update (prediction)
bool UnscentedKalmanFilter::updateTime(const size_t step, const gsl_vector *input)
{
	if (!x_hat_ || !P_ || !Chi_a_ || !Chi_) return false;

	const size_t &stateDim = system_.getStateDim();
	const size_t &processNoiseDim = system_.getProcessNoiseDim();

	gsl_vector *xa = NULL, *xx = NULL, *ww = NULL, *f_eval = NULL;
	gsl_matrix *XX = NULL;

	// propagate time
	// x-(k)
	gsl_vector_set_zero(x_hat_);
	for (size_t i = 0; i < sigmaDim_; ++i)
	{
		xa = &gsl_matrix_column(Chi_a_, i).vector;
		xx = &gsl_vector_subvector(xa, 0, stateDim).vector;
		ww = &gsl_vector_subvector(xa, stateDim, processNoiseDim).vector;

		f_eval = system_.evaluatePlantEquation(step, xx, input, ww);  // f = f(k, x(k), u(k), w(k))
		gsl_vector_memcpy(&gsl_matrix_column(Chi_, i).vector, f_eval);

		// y = a x + y
		gsl_blas_daxpy((0 == i ? Wm0_ : Wi_), f_eval, x_hat_);
	}

	// P-(k)
	gsl_matrix_set_zero(P_);
	for (size_t i = 0; i < sigmaDim_; ++i)
	{
		xx = &gsl_matrix_column(Chi_, i).vector;

		gsl_vector_memcpy(x_tmp_, xx);
		gsl_vector_sub(x_tmp_, x_hat_);

		// C = a op(A) op(B) + C
		XX = &gsl_matrix_view_vector(x_tmp_, x_tmp_->size, 1).matrix;
		if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, (0 == i ? Wc0_ : Wi_), XX, XX, 1.0, P_))
			return false;
	}

	// preserve symmetry of P
	gsl_matrix_transpose_memcpy(P_tmp_, P_);
	gsl_matrix_add(P_, P_tmp_);
	gsl_matrix_scale(P_, 0.5);

	return true;
}

// measurement update (correction)
bool UnscentedKalmanFilter::updateMeasurement(const size_t step, const gsl_vector *actualMeasurement, const gsl_vector *input)
{
	if (!x_hat_ || !y_hat_ || !P_ || !K_ || !Chi_a_ || !Chi_ || !Upsilon_ || !actualMeasurement) return false;

	const size_t &stateDim = system_.getStateDim();
	const size_t &processNoiseDim = system_.getProcessNoiseDim();
	const size_t &observationNoiseDim = system_.getObservationNoiseDim();

	gsl_vector *xa = NULL, *xx = NULL, *yy = NULL, *vv = NULL, *h_eval = NULL;
	gsl_matrix *XX = NULL, *YY = NULL;

	// y-(k)
	gsl_vector_set_zero(y_hat_);
	for (size_t i = 0; i < sigmaDim_; ++i)
	{
		xx = &gsl_matrix_column(Chi_, i).vector;
		xa = &gsl_matrix_column(Chi_a_, i).vector;
		vv = &gsl_vector_subvector(xa, stateDim + processNoiseDim, observationNoiseDim).vector;

		h_eval = system_.evaluateMeasurementEquation(step, xx, input, vv);  // h = h(k, x(k), u(k), v(k))
		gsl_vector_memcpy(&gsl_matrix_column(Upsilon_, i).vector, h_eval);

		// y = a x + y
		gsl_blas_daxpy((0 == i ? Wm0_ : Wi_), h_eval, y_hat_);
	}

	// Pyy & Pxy
	gsl_matrix_set_zero(Pyy_);
	gsl_matrix_set_zero(Pxy_);
	for (size_t i = 0; i < sigmaDim_; ++i)
	{
		yy = &gsl_matrix_column(Upsilon_, i).vector;
		xx = &gsl_matrix_column(Chi_, i).vector;

		gsl_vector_memcpy(y_tmp_, yy);
		gsl_vector_sub(y_tmp_, y_hat_);
		gsl_vector_memcpy(x_tmp_, xx);
		gsl_vector_sub(x_tmp_, x_hat_);

		// C = a op(A) op(B) + b C
		YY = &gsl_matrix_view_vector(y_tmp_, y_tmp_->size, 1).matrix;
		XX = &gsl_matrix_view_vector(x_tmp_, x_tmp_->size, 1).matrix;
		if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, (0 == i ? Wc0_ : Wi_), YY, YY, 1.0, Pyy_) ||
			GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, (0 == i ? Wc0_ : Wi_), XX, YY, 1.0, Pxy_))
			return false;
	}

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
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, -1.0, KPyy_tmp_, K_, 1.0, P_))
		return false;

	// preserve symmetry of P
	gsl_matrix_transpose_memcpy(P_tmp_, P_);
	gsl_matrix_add(P_, P_tmp_);
	gsl_matrix_scale(P_, 0.5);

	return true;
}

}  // namespace swl
