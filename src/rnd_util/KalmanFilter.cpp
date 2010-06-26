#include "swl/Config.h"
#include "swl/rnd_util/KalmanFilter.h"
#include <gsl/gsl_linalg.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

KalmanFilter::KalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
: x_hat_(NULL), /*y_hat_(NULL),*/ P_(NULL), K_(NULL), stateDim_(stateDim), inputDim_(inputDim), outputDim_(outputDim),
  residual_(NULL), RR_(NULL), invRR_(NULL), PCt_(NULL), permutation_(NULL), v_(NULL), M_(NULL)//, M2_(NULL)
{
	if (x0 && P0 && stateDim_ && inputDim_ && outputDim_ &&
		stateDim_ == x0->size && stateDim_ == P0->size1 && stateDim_ == P0->size2)
	{
		x_hat_ = gsl_vector_alloc(stateDim_);
		//y_hat_ = gsl_vector_alloc(outputDim_);
		P_ = gsl_matrix_alloc(stateDim_, stateDim_);
		K_ = gsl_matrix_alloc(stateDim_, outputDim_);

		residual_ = gsl_vector_alloc(outputDim_);
		RR_ = gsl_matrix_alloc(outputDim_, outputDim_);
		invRR_ = gsl_matrix_alloc(outputDim_, outputDim_);
		PCt_ = gsl_matrix_alloc(stateDim_, outputDim_);
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

// Kalman-Bucy filter
bool KalmanFilter::propagate(const double time)
{
	//-------------------------------------------------------------------------
	// the continuous plant:
	//
	// dx(t)/dt = A(t) * x(t) + B(t) * u(t) + W(t) * w(t)
	// y(t) = C(t) * x(t) + D(t) * u(t) + H(t) * w(t) + V(t) * v(t)
	// where E[w(t)] = E[v(t)] = 0, Q(t) = E[w(t) * w(t)^T], R(t) = E[v(t) * v(t)^T], N(t) = E[w(t) * v(t)^T]
	//
	// currently, this code is implemented only for H(t) = N(t) = 0
	// in case of H(t) != 0 || N(t) != 0, refer to Kalman filter in Matlab's help

#if 0
	if (!x_hat_ || /*!y_hat_ ||*/ !P_ || !K_) return false;

	const gsl_matrix *A = doGetSystemMatrix(time, x_hat_);
	const gsl_matrix *C = doGetOutputMatrix(time, x_hat_);

#if 0
	const gsl_matrix *W = doGetProcessNoiseCouplingMatrix(time);
	const gsl_matrix *V = doGetMeasurementNoiseCouplingMatrix(time);
	const gsl_matrix *Q = doGetProcessNoiseCovarianceMatrix(time);  // Q(t)
	const gsl_matrix *R = doGetMeasurementNoiseCovarianceMatrix(time);  // R(t)
#else
	const gsl_matrix *Qd = doGetProcessNoiseCovarianceMatrix(time);  // Qd(t) = W(t) * Q(t) * W(t)^T
	const gsl_matrix *Rd = doGetMeasurementNoiseCovarianceMatrix(time);  // Rd(t) = V(t) * R(t) * V(t)^T
#endif

	const gsl_vector *Bu = doGetControlInput(time, x_hat_);  // Bu(t) = B(t) * u(t)
	const gsl_vector *Du = doGetMeasurementInput(time, x_hat_);  // Du(t) = D(t) * u(t)

	const gsl_vector *y_tilde = doGetMeasurement(time, x_hat_);  // actual measurement

	if (!A || !C || !W || !V || !Qd || !Rd || !Bu || !Du || !y_tilde) return false;

	//-------------------------------------------------------------------------
	// continuous Kalman filter time update equations (prediction)

	// 1. propagate time
	// dx(t)/dt = A(t) * x(t) + B(t) * u(t)
	// dP(t)/dt = A(t) * P(t) + P(t) * A(t)^T + Qd where Qd(t) = W(t) * Q(t) * W(t)^T

	// preserve symmetry of P
	//gsl_matrix_transpose_memcpy(M_, P_);
	//gsl_matrix_add(P_, M_);
	//gsl_matrix_scale(P_, 0.5);

	//-------------------------------------------------------------------------
	// continuous Kalman filter measurement update equations (correction)

	// 1. calculate Kalman gain: K(t) = P(t) * C(t)^T * Rd(t)^-1 where Rd(t) = V(t) * R(t) * V(t)^T
	// 2. update measurement: dx(t)/dt = A(t) * x(t) + B(t) * u(t) + K(t) * (y_tilde(t) - y_hat(t)) where y_hat(t) = C(t) * x(t) + D(t) * u(t)
	// 3. update covariance:
	//	dP(t)/dt = A(t) * P(t) + P(t) * A(t)^T + Qd(t) - P(t) * C(t)^T * Rd(t)^-1 * C(t) * P(t) : the matrix Riccati differential equation
	//	         = A(t) * P(t) + P(t) * A(t)^T + Qd(t) - K(t) * Rd(t) * K(t)^T

	// preserve symmetry of P
	gsl_matrix_transpose_memcpy(M_, P_);
	gsl_matrix_add(P_, M_);
	gsl_matrix_scale(P_, 0.5);

	return true;
#else
	throw std::runtime_error("not yet implemented");
#endif
}

#if 0
// 0. initial estimates: x(0) & P(0)
// 1. time update (prediction): x(k-1) & P(k-1)  ==>  x-(k) & P-(k)
// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
//	==> result: posterior estimates, x(k) & P(k), conditioned on all available measurements at time k
bool KalmanFilter::propagate(const size_t step)  // 1-based time step. 0-th time step is initial
{
	//-------------------------------------------------------------------------
	// the discrete plant:
	//
	// x(k+1) = Phi(k) * x(k) + Bd(k) * u(k) + W(k) * w(k)
	// y(k) = Cd(k) * x(k) + Dd(k) * u(k) + H(k) * w(k) + V(k) * v(k)
	// where E[w(k)] = E[v(k)] = 0, Q(k) = E[w(k) * w(k)^T], R(k) = E[v(k) * v(k)^T], N(k) = E[w(k) * v(k)^T], Qd(k) = W(k) * Q(k) * W(k)^T, Rd(k) = V(k) * R(k) * V(k)^T
	//
	// currently, this code is implemented only for H(k) = N(k) = 0
	// in case of H(k) != 0 || N(k) != 0, refer to Kalman filter in Matlab's help

	if (!x_hat_ || /*!y_hat_ ||*/ !P_ || !K_) return false;

	const gsl_matrix *Phi = doGetStateTransitionMatrix(step, x_hat_);
	const gsl_matrix *Cd = doGetOutputMatrix(step, x_hat_);

#if 0
	const gsl_matrix *W = doGetProcessNoiseCouplingMatrix(step);
	const gsl_matrix *V = doGetMeasurementNoiseCouplingMatrix(step);
	const gsl_matrix *Q = doGetProcessNoiseCovarianceMatrix(step);  // Q(k)
	const gsl_matrix *R = doGetMeasurementNoiseCovarianceMatrix(step);  // R(k)
#else
	const gsl_matrix *Qd = doGetProcessNoiseCovarianceMatrix(step);  // Qd(k) = W(k) * Q(k) * W(k)^T
	const gsl_matrix *Rd = doGetMeasurementNoiseCovarianceMatrix(step);  // Rd(k) = V(k) * R(k) * V(k)^T
#endif

	const gsl_vector *Bu = doGetControlInput(step, x_hat_);  // Bu(k) = Bd(k) * u(k)
	const gsl_vector *Du = doGetMeasurementInput(step, x_hat_);  // Du(k) = Dd(k) * u(k)

	const gsl_vector *y_tilde = doGetMeasurement(step, x_hat_);  // actual measurement

	if (!Phi || !Cd || !Qd || !Rd || !Bu || !Du || !y_tilde) return false;

	//-------------------------------------------------------------------------
	// 1. time update (prediction)

	// FIXME [modify] >> some quantities at the time step, k-1 are required
	//	Phi(k-1), Bd(k-1), Q(k-1) & u(k-1) are used to predict x-(k)

	// (1) propagate time
	// x-(k) = Phi(k-1) * x(k-1) + Bd(k-1) * u(k-1)
	gsl_vector_memcpy(v_, x_hat_);
	gsl_vector_memcpy(x_hat_, Bu);
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, Phi, v_, 1.0, x_hat_))
		return false;

	// P-(k) = Phi(k-1) * P(k-1) * Phi(k-1)^T + Qd(k-1) where Qd(k-1) = W(k-1) * Q(k-1) * W(k-1)^T
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
	//gsl_matrix_transpose_memcpy(M_, P_);
	//gsl_matrix_add(P_, M_);
	//gsl_matrix_scale(P_, 0.5);

	//-------------------------------------------------------------------------
	// 2. measurement update (correction)

	// (1) calculate Kalman gain: K(k) = P-(k) * Cd(k)^T * (Cd(k) * P-(k) * Cd(k)^T + Rd(k))^-1 where Rd(k) = V(k) * R(k) * V(k)^T
	// inverse of matrix using LU decomposition
	gsl_matrix_memcpy(RR_, Rd);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, P_, Cd, 0.0, PCt_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Cd, PCt_, 1.0, RR_))
		return false;

	int signum;
	if (GSL_SUCCESS != gsl_linalg_LU_decomp(RR_, permutation_, &signum) ||
		GSL_SUCCESS != gsl_linalg_LU_invert(RR_, permutation_, invRR_))
		return false;

	//
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, PCt_, invRR_, 0.0, K_))  // calculate Kalman gain
		return false;

	// (2) update measurement: x(k) = x-(k) + K(k) * (y_tilde(k) - y_hat(k)) where y_hat(k) = Cd(k) * x-(k) + Dd(k) * u(k)
#if 0
	// save an estimated measurement, y_hat
	gsl_vector_memcpy(y_hat_, Du);
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, Cd, x_hat_, 1.0, y_hat_))  // calcuate y_hat(k)
		return false;
	gsl_vector_memcpy(residual_, y_hat_);
	if (GSL_SUCCESS != gsl_vector_sub(residual_, y_tilde) ||  // calculate residual = y_tilde(k) - y_hat(k)
		GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, -1.0, K_, residual_, 1.0, x_hat_))  // calculate x_hat(k)
		return false;
#else
	gsl_vector_memcpy(residual_, Du);
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, Cd, x_hat_, 1.0, residual_) ||  // calcuate y_hat(k)
		GSL_SUCCESS != gsl_vector_sub(residual_, y_tilde) ||  // calculate residual = y_tilde(k) - y_hat(k)
		GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, -1.0, K_, residual_, 1.0, x_hat_))  // calculate x_hat(k)
		return false;
#endif

	// (3) update covariance: P(k) = (I - K(k) * Cd(k)) * P-(k)
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
#else
// 0. initial estimates: x-(0) & P-(0)
// 1. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
//	==> result: prior estimates, x-(k+1) & P-(k+1), conditioned on all prior measurements except the one at time k+1
bool KalmanFilter::propagate(const size_t step)  // 0-based time step. 0-th time step is initial
{
	//-------------------------------------------------------------------------
	// the discrete plant:
	//
	// x(k+1) = Phi(k) * x(k) + Bd(k) * u(k) + W(k) * w(k)
	// y(k) = Cd(k) * x(k) + Dd(k) * u(k) + H(k) * w(k) + V(k) * v(k)
	// where E[w(k)] = E[v(k)] = 0, Q(k) = E[w(k) * w(k)^T], R(k) = E[v(k) * v(k)^T], N(k) = E[w(k) * v(k)^T], Qd(k) = W(k) * Q(k) * W(k)^T, Rd(k) = V(k) * R(k) * V(k)^T
	//
	// currently, this code is implemented only for H(k) = N(k) = 0
	// in case of H(k) != 0 || N(k) != 0, refer to Kalman filter in Matlab's help

	if (!x_hat_ || /*!y_hat_ ||*/ !P_ || !K_) return false;

	const gsl_matrix *Phi = doGetStateTransitionMatrix(step, x_hat_);
	const gsl_matrix *Cd = doGetOutputMatrix(step, x_hat_);

#if 0
	const gsl_matrix *W = doGetProcessNoiseCouplingMatrix(step);
	const gsl_matrix *V = doGetMeasurementNoiseCouplingMatrix(step);
	const gsl_matrix *Q = doGetProcessNoiseCovarianceMatrix(step);  // Q(k)
	const gsl_matrix *R = doGetMeasurementNoiseCovarianceMatrix(step);  // R(k)
#else
	const gsl_matrix *Qd = doGetProcessNoiseCovarianceMatrix(step);  // Qd(k) = W(k) * Q(k) * W(k)^T
	const gsl_matrix *Rd = doGetMeasurementNoiseCovarianceMatrix(step);  // Rd(k) = V(k) * R(k) * V(k)^T
#endif

	const gsl_vector *Bu = doGetControlInput(step, x_hat_);  // Bu(k) = Bd(k) * u(k)
	const gsl_vector *Du = doGetMeasurementInput(step, x_hat_);  // Du(k) = Dd(k) * u(k)

	const gsl_vector *y_tilde = doGetMeasurement(step, x_hat_);  // actual measurement

	if (!Phi || !Cd || !Qd || !Rd || !Bu || !Du || !y_tilde) return false;

	//-------------------------------------------------------------------------
	// 1. measurement update (correction)

	// (1) calculate Kalman gain: K(k) = P-(k) * Cd(k)^T * (Cd(k) * P-(k) * Cd(k)^T + Rd(k))^-1 where Rd(k) = V(k) * R(k) * V(k)^T
	// inverse of matrix using LU decomposition
	gsl_matrix_memcpy(RR_, Rd);
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, P_, Cd, 0.0, PCt_) ||
		GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Cd, PCt_, 1.0, RR_))
		return false;

	int signum;
	if (GSL_SUCCESS != gsl_linalg_LU_decomp(RR_, permutation_, &signum) ||
		GSL_SUCCESS != gsl_linalg_LU_invert(RR_, permutation_, invRR_))
		return false;

	//
	if (GSL_SUCCESS != gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, PCt_, invRR_, 0.0, K_))  // calculate Kalman gain
		return false;

	// (2) update measurement: x(k) = x-(k) + K(k) * (y_tilde(k) - y_hat(k)) where y_hat(k) = Cd(k) * x-(k) + Dd(k) * u(k)
#if 0
	// save an estimated measurement, y_hat
	gsl_vector_memcpy(y_hat_, Du);
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, Cd, x_hat_, 1.0, y_hat_))  // calcuate y_hat(k)
		return false;
	gsl_vector_memcpy(residual_, y_hat_);
	if (GSL_SUCCESS != gsl_vector_sub(residual_, y_tilde) ||  // calculate residual = y_tilde(k) - y_hat(k)
		GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, -1.0, K_, residual_, 1.0, x_hat_))  // calculate x_hat(k)
		return false;
#else
	gsl_vector_memcpy(residual_, Du);
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, Cd, x_hat_, 1.0, residual_) ||  // calcuate y_hat(k)
		GSL_SUCCESS != gsl_vector_sub(residual_, y_tilde) ||  // calculate residual = y_tilde(k) - y_hat(k)
		GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, -1.0, K_, residual_, 1.0, x_hat_))  // calculate x_hat(k)
		return false;
#endif

	// (3) update covariance: P(k) = (I - K(k) * Cd(k)) * P-(k)
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
	//gsl_matrix_transpose_memcpy(M_, P_);
	//gsl_matrix_add(P_, M_);
	//gsl_matrix_scale(P_, 0.5);

	//-------------------------------------------------------------------------
	// 2. time update (prediction)

	// (1) propagate time
	// x-(k+1) = Phi(k) * x(k) + Bd(k) * u(k)
	gsl_vector_memcpy(v_, x_hat_);
	gsl_vector_memcpy(x_hat_, Bu);
	if (GSL_SUCCESS != gsl_blas_dgemv(CblasNoTrans, 1.0, Phi, v_, 1.0, x_hat_))
		return false;

	// P-(k+1) = Phi(k) * P(k) * Phi(k)^T + Qd(k) where Qd(k) = W(k) * Q(k) * W(k)^T
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
#endif

}  // namespace swl
