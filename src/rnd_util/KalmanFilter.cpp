#include "swl/Config.h"
#include "swl/rnd_util/KalmanFilter.h"
#include <gsl/gsl_linalg.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

KalmanFilter::KalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
: x_hat_(NULL), y_hat_(NULL), P_(NULL), K_(NULL), stateDim_(stateDim), inputDim_(inputDim), outputDim_(outputDim),
  r_(NULL), RR_(NULL), invRR_(NULL), PHt_(NULL), permutation_(NULL), v_(NULL), M_(NULL)//, M2_(NULL)
{
	if (x0 && P0 && stateDim_ && inputDim_ && outputDim_ &&
		stateDim_ == x0->size && stateDim_ == P0->size1 && stateDim_ == P0->size2)
	{
		x_hat_ = gsl_vector_alloc(stateDim_);
		y_hat_ = gsl_vector_alloc(outputDim_);
		P_ = gsl_matrix_alloc(stateDim_, stateDim_);
		K_ = gsl_matrix_alloc(stateDim_, outputDim_);

		r_ = gsl_vector_alloc(outputDim_);
		RR_ = gsl_matrix_alloc(outputDim_, outputDim_);
		invRR_ = gsl_matrix_alloc(outputDim_, outputDim_);
		PHt_ = gsl_matrix_alloc(stateDim_, outputDim_);
		permutation_ = gsl_permutation_alloc(outputDim_);

		v_ = gsl_vector_alloc(stateDim_);
		M_ = gsl_matrix_alloc(stateDim_, stateDim_);
		M2_ = gsl_matrix_alloc(stateDim_, stateDim_);

		gsl_vector_memcpy(x_hat_, x0);
		//gsl_vector_set_zero(y_hat_);
		gsl_matrix_memcpy(P_, P0);
		//gsl_matrix_set_identity(K_);
	}
}

KalmanFilter::~KalmanFilter()
{
	gsl_vector_free(x_hat_);  x_hat_ = NULL;
	gsl_vector_free(y_hat_);  y_hat_ = NULL;
	gsl_matrix_free(P_);  P_ = NULL;
	gsl_matrix_free(K_);  K_ = NULL;

	gsl_vector_free(r_);  r_ = NULL;
	gsl_matrix_free(RR_);  RR_ = NULL;
	gsl_matrix_free(invRR_);  invRR_ = NULL;
	gsl_matrix_free(PHt_);  PHt_ = NULL;
	gsl_permutation_free(permutation_);  permutation_ = NULL;

	gsl_vector_free(v_);  v_ = NULL;
	gsl_matrix_free(M_);  M_ = NULL;
	gsl_matrix_free(M2_);  M2_ = NULL;
}

bool KalmanFilter::propagate(const double time)
{
#if 0
	if (!x_hat_ || !y_hat_ || !P_ || !K_) return false;

	const gsl_matrix *A = getSystemMatrix(step);
	const gsl_matrix *G = getInputMatrix(step);
	const gsl_matrix *C = getOutputMatrix(step);
	const gsl_matrix *Q = getProcessNoiseCovarianceMatrix(step);  // Q, but not Qd
	const gsl_matrix *R = getMeasurementNoiseCovarianceMatrix(step);

	const gsl_vector *Bu = getControlInput(step);
	const gsl_vector *y_tilde = getMeasurement(step);
	if (!A || !G || !C || !Q || !R || !Bu || !y_tilde) return false;

	// gsl_blas_dgemv: y = a op(A) x + b y
	// gsl_blas_dgemm: C = a op(A) op(B) + b C
	//	CBLAS_TRANSPOSE_t: CblasNoTrans, CblasTrans, CblasConjTrans

	// calculate Kalman gain

	// update measurement

	// update covariance

	// propagate time

	// preserve symmetry of P
	gsl_matrix_transpose_memcpy(M_, P_);
	gsl_matrix_add(P_, M_);
	gsl_matrix_scale(P_, 0.5);

	return true;
#else
	throw std::runtime_error("not yet implemented");
#endif
}

bool KalmanFilter::propagate(const size_t step)  // 1-based time step. 0-th time step is initial
{
	if (!x_hat_ || !y_hat_ || !P_ || !K_) return false;

	const gsl_matrix *Phi = getStateTransitionMatrix(step);
	const gsl_matrix *H = getOutputMatrix(step);
	const gsl_matrix *Qd = getProcessNoiseCovarianceMatrix(step);  // Qd, but not Q
	const gsl_matrix *R = getMeasurementNoiseCovarianceMatrix(step);

	const gsl_vector *Bu = getControlInput(step);
	const gsl_vector *y_tilde = getMeasurement(step);
	if (!Phi || !H || !Qd || !R || !Bu || !y_tilde) return false;

	// gsl_blas_dgemv: y = a op(A) x + b y
	// gsl_blas_dgemm: H = a op(A) op(B) + b H
	//	CBLAS_TRANSPOSE_t: CblasNoTrans, CblasTrans, CblasConjTrans

	// 1. propagate time
#if 0
	// using Q
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Phi, P_, 0.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, M_, Phi, 0.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, G, Qd, 0.0, M2_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, M2_, G, 0.0, P_) ||
		GSL_SUCCESS != gsl_matrix_add(P_, M_))
		return false;
#else
	// using Qd = Gamma * Q * Gamma^T
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Phi, P_, 0.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, M_, Phi, 0.0, P_) ||
		GSL_SUCCESS != gsl_matrix_add(P_, Qd))
		return false;
#endif

#if 0
	// not working
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, Phi, x_hat_, 0.0, x_hat_) ||
		GSL_SUCCESS != gsl_vector_add(x_hat_, Bu))
		return false;
#else
	gsl_vector_memcpy(v_, x_hat_);
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, Phi, v_, 0.0, x_hat_) ||
		GSL_SUCCESS != gsl_vector_add(x_hat_, Bu))
		return false;
#endif

	// 2. calculate Kalman gain
	// inverse of matrix using LU decomposition
	gsl_matrix_memcpy(RR_, R);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, P_, H, 0.0, PHt_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, H, PHt_, 1.0, RR_))
		return false;

	int signum;
	gsl_linalg_LU_decomp(RR_, permutation_, &signum);
	gsl_linalg_LU_invert(RR_, permutation_, invRR_);

	//
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, PHt_, invRR_, 0.0, K_))  // calculate Kalman gain
		return false;

	// 3. update measurement
#if 1
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, H, x_hat_, 0.0, y_hat_))  // calcuate y_hat
		return false;
	gsl_vector_memcpy(r_, y_hat_);
	if (GSL_SUCCESS != gsl_vector_sub(r_, y_tilde) ||  // calculate residual = y_tilde - y_hat
		GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, -1.0, K_, r_, 1.0, x_hat_))  // calculate x_hat
		return false;
#else
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, H, x_hat_, 0.0, r_) ||  // calcuate y_hat
		GSL_SUCCESS != gsl_vector_sub(r_, y_tilde) ||  // calculate residual = y_tilde - y_hat
		GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, -1.0, K_, r_, 1.0, x_hat_))  // calculate x_hat
		return false;
#endif

	// 4. update covariance
#if 0
	// not working
	gsl_matrix_set_identity(M_);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, K_, H, 1.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, M_, P_, 0.0, P_))
		return false;
#else
	gsl_matrix_set_identity(M_);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, K_, H, 1.0, M_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, M_, P_, 0.0, M2_))
		return false;
	gsl_matrix_memcpy(P_, M2_);
#endif

	// 5. preserve symmetry of P
	gsl_matrix_transpose_memcpy(M_, P_);
	gsl_matrix_add(P_, M_);
	gsl_matrix_scale(P_, 0.5);

	return true;
}

}  // namespace swl
