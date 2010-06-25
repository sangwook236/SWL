#if !defined(__SWL_RND_UTIL__KALMAN_FILTER__H_)
#define __SWL_RND_UTIL__KALMAN_FILTER__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <gsl/gsl_blas.h>
#include <vector>


#ifdef __cplusplus
extern "C" {
#endif

//typedef struct gsl_vector;
//typedef struct gsl_matrix;
typedef struct gsl_permutation_struct gsl_permutation;

#ifdef __cplusplus
}
#endif

namespace swl {

//--------------------------------------------------------------------------
//

class SWL_RND_UTIL_API KalmanFilter
{
public:
	//typedef KalmanFilter base_type;

protected:
	KalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim);
public:
	virtual ~KalmanFilter();

private:
	KalmanFilter(const KalmanFilter &rhs);
	KalmanFilter & operator=(const KalmanFilter &rhs);

public:
	// for continuous Kalman filter
	bool propagate(const double time);
	// for discrete Kalman filter
	bool propagate(const size_t step);  // 1-based time step. 0-th time step is initial

	const gsl_vector * getState() const  {  return x_hat_;  }
	const gsl_vector * getOutput() const  {  return y_hat_;  }
	const gsl_matrix * getStateErrorCovarianceMatrix() const  {  return P_;  }
	const gsl_matrix * getKalmanGain() const  {  return K_;  }

	size_t getStateDim() const  {  return stateDim_;  }
	size_t getInputDim() const  {  return inputDim_;  }
	size_t getOutputDim() const  {  return outputDim_;  }

private:
	virtual gsl_vector * getMeasurement(const size_t step) const = 0;

	// for continuous Kalman filter
	virtual gsl_matrix * getSystemMatrix(const size_t step) const = 0;  // A
	// for discrete Kalman filter
	virtual gsl_matrix * getStateTransitionMatrix(const size_t step) const = 0;  // Phi

	virtual gsl_matrix * getOutputMatrix(const size_t step) const = 0;  // C or Cd
	virtual gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const = 0;  // G
	virtual gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const = 0;  // H
	virtual gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const = 0;  // Q or Qd
	virtual gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const = 0;  // R

	virtual gsl_vector * getControlInput(const size_t step) const = 0;  // Bu = B * u or Bd * u

protected:
	// estimated state vector
	gsl_vector *x_hat_;
	// estimated output(measurement) vector
	gsl_vector *y_hat_;
	// state error covariance matrix
	gsl_matrix *P_;
	// Kalman gain
	gsl_matrix *K_;

	const size_t stateDim_;
	const size_t inputDim_;
	const size_t outputDim_;

private:
	// r = y_tilde - y_hat
	gsl_vector *r_;

	gsl_matrix *RR_;
	gsl_matrix *invRR_;
	gsl_matrix *PHt_;
	gsl_permutation *permutation_;

	// for temporary computation
	gsl_vector *v_;
	gsl_matrix *M_;
	gsl_matrix *M2_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__KALMAN_FILTER__H_
