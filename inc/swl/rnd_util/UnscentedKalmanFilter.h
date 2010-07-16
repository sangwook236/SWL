#if !defined(__SWL_RND_UTIL__UNSCENTED_KALMAN_FILTER__H_)
#define __SWL_RND_UTIL__UNSCENTED_KALMAN_FILTER__H_ 1


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

//--------------------------------------------------------------------------
// the unscented Kalman filter

class SWL_RND_UTIL_API UnscentedKalmanFilter
{
public:
	//typedef UnscentedKalmanFilter base_type;

public:
	UnscentedKalmanFilter(const DiscreteNonlinearStochasticSystem &system, const double alpha, const double beta, const double kappa, const gsl_vector *x0, const gsl_matrix *P0);
	virtual ~UnscentedKalmanFilter();

private:
	UnscentedKalmanFilter(const UnscentedKalmanFilter &rhs);
	UnscentedKalmanFilter & operator=(const UnscentedKalmanFilter &rhs);

public:
	bool performUnscentedTransformation(const gsl_vector *w, const gsl_vector *v, const gsl_matrix *Q, const gsl_matrix *R);
	bool updateTime(const size_t step, const gsl_vector *input);
	bool updateMeasurement(const size_t step, const gsl_vector *actualMeasurement, const gsl_vector *input);

	const gsl_vector * getEstimatedState() const  {  return x_hat_;  }
	//const gsl_vector * getEstimatedMeasurement() const  {  return y_hat_;  }
	const gsl_matrix * getStateErrorCovarianceMatrix() const  {  return P_;  }
	const gsl_matrix * getKalmanGain() const  {  return K_;  }

protected:
	const DiscreteNonlinearStochasticSystem &system_;

	const size_t L_;
	const double alpha_;
	const double beta_;
	const double kappa_;

	// estimated state vector
	gsl_vector *x_hat_;
	// estimated measurement vector
	gsl_vector *y_hat_;
	// state error covariance matrix
	gsl_matrix *P_;
	// Kalman gain
	gsl_matrix *K_;

private:
	const double lambda_;
	const double gamma_;  // sqrt(L + lambda)
	const size_t sigmaDim_;  // 2L + 1

	gsl_vector *xa_;

	// a matrix of 2L + 1 sigma vectors
	gsl_matrix *Chi_a_;
	gsl_matrix *P_a_;
	gsl_matrix *Chi_;
	gsl_matrix *Upsilon_;

	const double Wm0_;
	const double Wc0_;
	const double Wi_;

	gsl_matrix *Pyy_;
	gsl_matrix *Pxy_;

	//
	gsl_vector *xa_tmp_;
	gsl_vector *x_tmp_;
	gsl_vector *y_tmp_;
	gsl_matrix *P_tmp_;
	gsl_matrix *Pyy_tmp_;
	gsl_matrix *invPyy_;
	gsl_matrix *KPyy_tmp_;

	gsl_permutation *permutation_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__UNSCENTED_KALMAN_FILTER__H_
