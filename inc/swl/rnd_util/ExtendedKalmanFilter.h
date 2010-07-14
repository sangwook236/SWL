#if !defined(__SWL_RND_UTIL__EXTENDED_KALMAN_FILTER__H_)
#define __SWL_RND_UTIL__EXTENDED_KALMAN_FILTER__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <gsl/gsl_blas.h>


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

class DiscreteNonlinearStochasticSystem;
class ContinuousNonlinearStochasticSystem;

//--------------------------------------------------------------------------
// discrete extended Kalman filter

class SWL_RND_UTIL_API DiscreteExtendedKalmanFilter
{
public:
	//typedef DiscreteExtendedKalmanFilter base_type;

public:
	DiscreteExtendedKalmanFilter(const DiscreteNonlinearStochasticSystem &system, const gsl_vector *x0, const gsl_matrix *P0);
	virtual ~DiscreteExtendedKalmanFilter();

private:
	DiscreteExtendedKalmanFilter(const DiscreteExtendedKalmanFilter &rhs);
	DiscreteExtendedKalmanFilter & operator=(const DiscreteExtendedKalmanFilter &rhs);

public:
	bool updateTime(const size_t step, const gsl_vector *f_eval);  // f = f(k, x(k), u(k), 0)
	bool updateMeasurement(const size_t step, const gsl_vector *actualMeasurement, const gsl_vector *h_eval);  // h = h(k, x(k), u(k), 0)

	const gsl_vector * getEstimatedState() const  {  return x_hat_;  }
	//const gsl_vector * getEstimatedMeasurement() const  {  return y_hat_;  }
	const gsl_matrix * getStateErrorCovarianceMatrix() const  {  return P_;  }
	const gsl_matrix * getKalmanGain() const  {  return K_;  }

protected:
	const DiscreteNonlinearStochasticSystem &system_;

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

//--------------------------------------------------------------------------
// continuous extended Kalman filter

class SWL_RND_UTIL_API ContinuousExtendedKalmanFilter
{
public:
	//typedef ContinuousExtendedKalmanFilter base_type;

public:
	ContinuousExtendedKalmanFilter(const ContinuousNonlinearStochasticSystem &system, const gsl_vector *x0, const gsl_matrix *P0);
	virtual ~ContinuousExtendedKalmanFilter();

private:
	ContinuousExtendedKalmanFilter(const ContinuousExtendedKalmanFilter &rhs);
	ContinuousExtendedKalmanFilter & operator=(const ContinuousExtendedKalmanFilter &rhs);

public:
	bool updateTime(const double time, const gsl_vector *f_eval);  // f = f(t, x(t), u(t), 0)
	bool updateMeasurement(const double time, const gsl_vector *actualMeasurement, const gsl_vector *h_eval);  // h = h(t, x(t), u(t), 0)

	const gsl_vector * getEstimatedState() const  {  return x_hat_;  }
	//const gsl_vector * getEstimatedMeasurement() const  {  return y_hat_;  }
	const gsl_matrix * getStateErrorCovarianceMatrix() const  {  return P_;  }
	const gsl_matrix * getKalmanGain() const  {  return K_;  }

protected:
	const ContinuousNonlinearStochasticSystem &system_;

	// estimated state vector
	gsl_vector *x_hat_;
	// estimated measurement vector
	//gsl_vector *y_hat_;
	// state error covariance matrix
	gsl_matrix *P_;
	// Kalman gain
	gsl_matrix *K_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__EXTENDED_KALMAN_FILTER__H_
