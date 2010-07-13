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

class DiscreteLinearStochasticSystem;

//--------------------------------------------------------------------------
//

class SWL_RND_UTIL_API KalmanFilter
{
public:
	//typedef KalmanFilter base_type;

public:
	KalmanFilter(const DiscreteLinearStochasticSystem &system, const gsl_vector *x0, const gsl_matrix *P0);
	virtual ~KalmanFilter();

private:
	KalmanFilter(const KalmanFilter &rhs);
	KalmanFilter & operator=(const KalmanFilter &rhs);

public:
	// for continuous Kalman filter
	bool updateTime(const double time, const gsl_vector *Bu);  // Bu(t) = B(t) * u(t)
	bool updateMeasurement(const double time, const gsl_vector *actualMeasurement, const gsl_vector *Du);  // Du(t) = D(t) * u(t)
	// for discrete Kalman filter
	bool updateTime(const size_t step, const gsl_vector *Bu);  // Bu(k) = Bd(k) * u(k)
	bool updateMeasurement(const size_t step, const gsl_vector *actualMeasurement, const gsl_vector *Du);  // Du(k) = Dd(k) * u(k)

	const gsl_vector * getEstimatedState() const  {  return x_hat_;  }
	//const gsl_vector * getEstimatedMeasurement() const  {  return y_hat_;  }
	const gsl_matrix * getStateErrorCovarianceMatrix() const  {  return P_;  }
	const gsl_matrix * getKalmanGain() const  {  return K_;  }

protected:
	const DiscreteLinearStochasticSystem &system_;

	// estimated state vector
	gsl_vector *x_hat_;
	// estimated measurement vector
	//gsl_vector *y_hat_;
	// state error covariance matrix
	gsl_matrix *P_;
	// Kalman gain
	gsl_matrix *K_;

private:
	// residual = y_tilde - y_hat
	gsl_vector *residual_;

	gsl_matrix *RR_;
	gsl_matrix *invRR_;
	gsl_matrix *PCt_;
	gsl_permutation *permutation_;

	// for temporary computation
	gsl_vector *v_;
	gsl_matrix *M_;
	gsl_matrix *M2_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__KALMAN_FILTER__H_
