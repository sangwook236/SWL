#if !defined(__SWL_RND_UTIL__EXTENDED_KALMAN_FILTER__H_)
#define __SWL_RND_UTIL__EXTENDED_KALMAN_FILTER__H_ 1


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

class SWL_RND_UTIL_API ExtendedKalmanFilter
{
public:
	//typedef ExtendedKalmanFilter base_type;

protected:
	ExtendedKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim);
public:
	virtual ~ExtendedKalmanFilter();

private:
	ExtendedKalmanFilter(const ExtendedKalmanFilter &rhs);
	ExtendedKalmanFilter & operator=(const ExtendedKalmanFilter &rhs);

public:
	// for continuous Kalman filter
	bool propagate(const double time);
	// for discrete Kalman filter
	bool propagate(const size_t step);  // 1-based time step. 0-th time step is initial

	const gsl_vector * getState() const  {  return x_;  }
	const gsl_vector * getOutput() const  {  return y_;  }
	const gsl_matrix * getStateErrorCovarianceMatrix() const  {  return P_;  }
	const gsl_matrix * getKalmanGain() const  {  return K_;  }

	size_t getStateDim() const  {  return stateDim_;  }
	size_t getInputDim() const  {  return inputDim_;  }
	size_t getOutputDim() const  {  return outputDim_;  }

private:
	virtual gsl_vector * getMeasurement(const size_t step) const = 0;

	// for continuous Kalman filter
	virtual gsl_matrix * getSystemMatrix(const size_t step) const = 0;
	// for discrete Kalman filter
	virtual gsl_matrix * getStateTransitionMatrix(const size_t step) const = 0;
	virtual gsl_matrix * getInputMatrix(const size_t step) const = 0;
	virtual gsl_matrix * getOutputMatrix(const size_t step) const = 0;
	virtual gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const = 0;
	virtual gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const = 0;

	virtual gsl_vector * getControlInput(const size_t step) const = 0;

protected:
	// state vector
	gsl_vector *x_;
	// computed output vector
	gsl_vector *y_;
	// state error covariance matrix
	gsl_matrix *P_;
	// Kalman gain
	gsl_matrix *K_;

	const size_t stateDim_;
	const size_t inputDim_;
	const size_t outputDim_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__EXTENDED_KALMAN_FILTER__H_
