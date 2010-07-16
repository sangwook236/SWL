#if !defined(__SWL_RND_UTIL__CONTINUOUS_NONLINEAR_STOCHASTIC_SYSTEM__H_)
#define __SWL_RND_UTIL__CONTINUOUS_NONLINEAR_STOCHASTIC_SYSTEM__H_ 1


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

// the stochastic differential equation: dx/dt(t) = f(t, x(t), u(t), * w(t))
// the observation equation: y(t) = h(t, x(t), u(t), v(t))

class ContinuousNonlinearStochasticSystem
{
public:
	//typedef ContinuousNonlinearStochasticSystem base_type;

protected:
	ContinuousNonlinearStochasticSystem(const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim)
	: stateDim_(stateDim), inputDim_(inputDim), outputDim_(outputDim), processNoiseDim_(processNoiseDim), observationNoiseDim_(observationNoiseDim)
	{}
public:
	virtual ~ContinuousNonlinearStochasticSystem()  {}

private:
	ContinuousNonlinearStochasticSystem(const ContinuousNonlinearStochasticSystem &rhs);
	ContinuousNonlinearStochasticSystem & operator=(const ContinuousNonlinearStochasticSystem &rhs);

public:
	// the stochastic differential equation: f = f(t, x(t), u(t), w(t))
	virtual gsl_vector * evaluatePlantEquation(const double time, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const = 0;

	// for a linearized system
	virtual gsl_matrix * getSystemMatrix(const double time, const gsl_vector *state) const = 0;  // A(t) = df(t, x(t), u(t), 0)/dx
	virtual gsl_vector * getControlInput(const double time, const gsl_vector *state) const = 0;  // Bu(t) = B(t) * u(t)
	//virtual gsl_matrix * getProcessNoiseCouplingMatrix(const double time) const = 0;  // W(t) = df(t, x(t), u(t), 0)/dw

	// the observation equation: h = h(t, x(t), u(t), v(t))
	virtual gsl_vector * evaluateMeasurementEquation(const double time, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const = 0;

	// for a linearized system
	virtual gsl_matrix * getOutputMatrix(const double time, const gsl_vector *state) const = 0;  // C(t) = dh(t, x(t), u(t), 0)/dx
	virtual gsl_vector * getMeasurementInput(const double time, const gsl_vector *state) const = 0;  // Du(t) = D(t) * u(t)
	//virtual gsl_matrix * getMeasurementNoiseCouplingMatrix(const double time) const = 0;  // V(t) = dh(t, x(t), u(t), 0)/dv

	// noise covariance matrices
	virtual gsl_matrix * getProcessNoiseCovarianceMatrix(const double time) const = 0;  // Q or Qd = W * Q * W^T
	virtual gsl_matrix * getMeasurementNoiseCovarianceMatrix(const double time) const = 0;  // R or Rd = V * R * V^T

	// actual measurement
	//virtual gsl_vector * doGetMeasurement(const double time, const gsl_vector *state) const = 0;

	size_t getStateDim() const  {  return stateDim_;  }
	size_t getInputDim() const  {  return inputDim_;  }
	size_t getOutputDim() const  {  return outputDim_;  }
	size_t getProcessNoiseDim() const  {  return processNoiseDim_;  }
	size_t getObservationNoiseDim() const  {  return observationNoiseDim_;  }

protected:
	const size_t stateDim_;
	const size_t inputDim_;
	const size_t outputDim_;
	const size_t processNoiseDim_;
	const size_t observationNoiseDim_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__CONTINUOUS_NONLINEAR_STOCHASTIC_SYSTEM__H_
