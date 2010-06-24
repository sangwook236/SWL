#include "swl/Config.h"
#include "swl/rnd_util/ExtendedKalmanFilter.h"
#include <gsl/gsl_linalg.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

ExtendedKalmanFilter::ExtendedKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
: x_(NULL), y_(NULL), P_(NULL), K_(NULL), stateDim_(stateDim), inputDim_(inputDim), outputDim_(outputDim)
{
	if (x0 && P0 && stateDim_ && inputDim_ && outputDim_ &&
		stateDim_ == x0->size && stateDim_ == P0->size1 && stateDim_ == P0->size2)
	{
		x_ = gsl_vector_alloc(stateDim_);
		y_ = gsl_vector_alloc(outputDim_);
		P_ = gsl_matrix_alloc(stateDim_, stateDim_);
		K_ = gsl_matrix_alloc(stateDim_, outputDim_);
/*
		r_ = gsl_vector_alloc(outputDim_);
		RR_ = gsl_matrix_alloc(outputDim_, outputDim_);
		invRR_ = gsl_matrix_alloc(outputDim_, outputDim_);
		PHt_ = gsl_matrix_alloc(stateDim_, outputDim_);
		permutation_ = gsl_permutation_alloc(outputDim_);

		M1_ = gsl_matrix_alloc(stateDim_, stateDim_);
		M2_ = gsl_matrix_alloc(stateDim_, inputDim_);
*/
		gsl_vector_memcpy(x_, x0);
		//gsl_vector_set_zero(y_);
		gsl_matrix_memcpy(P_, P0);
		//gsl_matrix_set_identity(K_);
	}
}

ExtendedKalmanFilter::~ExtendedKalmanFilter()
{
	gsl_vector_free(x_);  x_ = NULL;
	gsl_vector_free(y_);  y_ = NULL;
	gsl_matrix_free(P_);  P_ = NULL;
	gsl_matrix_free(K_);  K_ = NULL;
/*
	gsl_vector_free(r_);  r_ = NULL;
	gsl_matrix_free(RR_);  RR_ = NULL;
	gsl_matrix_free(invRR_);  invRR_ = NULL;
	gsl_matrix_free(PHt_);  PHt_ = NULL;
	gsl_permutation_free(permutation_);  permutation_ = NULL;

	gsl_matrix_free(M1_);  M1_ = NULL;
	gsl_matrix_free(M2_);  M2_ = NULL;
*/
}

bool ExtendedKalmanFilter::propagate(const double time)
{
#if 0
	if (!x_ || !y_ || !P_) return false;

	gsl_matrix *A = getSystemMatrix(step);
	gsl_matrix *B = getInputMatrix(step);
	gsl_matrix *C = getOutputMatrix(step);
	gsl_matrix *Q = getProcessNoiseCovarianceMatrix(step);
	gsl_matrix *R = getMeasurementNoiseCovarianceMatrix(step);
	gsl_vector *y_tilde = getMeasurement(step);
	if (!A || !B || !C || !Q || !R || !y_tilde) return false;

	// gsl_blas_dgemv: y = a op(A) x + b y
	// gsl_blas_dgemm: C = a op(A) op(B) + b C
	//	CBLAS_TRANSPOSE_t: CblasNoTrans, CblasTrans, CblasConjTrans

	return true;
#else
	throw std::runtime_error("not yet implemented");
#endif
}

bool ExtendedKalmanFilter::propagate(const size_t step)  // 1-based time step. 0-th time step is initial
{
#if 0
	if (!x_ || !y_ || !P_) return false;

	gsl_matrix *Phi = getStateTransitionMatrix(step);
	gsl_matrix *B = getInputMatrix(step);
	gsl_matrix *C = getOutputMatrix(step);
	gsl_matrix *Q = getProcessNoiseCovarianceMatrix(step);
	gsl_matrix *R = getMeasurementNoiseCovarianceMatrix(step);
	gsl_vector *y_tilde = getMeasurement(step);
	if (!Phi || !B || !C || !Q || !R || !y_tilde) return false;

	// gsl_blas_dgemv: y = a op(A) x + b y
	// gsl_blas_dgemm: C = a op(A) op(B) + b C
	//	CBLAS_TRANSPOSE_t: CblasNoTrans, CblasTrans, CblasConjTrans

	return true;
#else
	throw std::runtime_error("not yet implemented");
#endif
}

}  // namespace swl
