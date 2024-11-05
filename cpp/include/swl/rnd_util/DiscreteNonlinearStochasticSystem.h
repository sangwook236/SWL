#if !defined(__SWL_RND_UTIL__DISCRETE_NONLINEAR_STOCHASTIC_SYSTEM__H_)
#define __SWL_RND_UTIL__DISCRETE_NONLINEAR_STOCHASTIC_SYSTEM__H_ 1


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

// the stochastic differential equation: x(k+1) = f(k, x(k), u(k), w(k))
// the observation equation: y(k) = h(k, x(k), u(k), v(k))

class DiscreteNonlinearStochasticSystem
{
public:
	//typedef DiscreteNonlinearStochasticSystem base_type;

protected:
	DiscreteNonlinearStochasticSystem(const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim)
	: stateDim_(stateDim), inputDim_(inputDim), outputDim_(outputDim), processNoiseDim_(processNoiseDim), observationNoiseDim_(observationNoiseDim)
	{}
public:
	virtual ~DiscreteNonlinearStochasticSystem()  {}

private:
	DiscreteNonlinearStochasticSystem(const DiscreteNonlinearStochasticSystem &rhs);  // not implemented
	DiscreteNonlinearStochasticSystem & operator=(const DiscreteNonlinearStochasticSystem &rhs);  // not implemented

public:
	// the stochastic differential equation: f = f(k, x(k), u(k), w(k))
	virtual gsl_vector * evaluatePlantEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const = 0;

	// for a linearized system
	virtual gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const = 0;  // Phi(k) = exp(A(k) * Ts) where A(k) = df(k, x(k), u(k), 0)/dx
	virtual gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const = 0;  // Bu(k) = Bd(k) * u(k)
	//virtual gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const = 0;  // W(k) = df(k, x(k), u(k), 0)/dw

	// the observation equation: h = h(k, x(k), u(k), v(k))
	virtual gsl_vector * evaluateMeasurementEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const = 0;

	// for a linearized system
	virtual gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const = 0;  // Cd(k) = dh(k, x(k), u(k), 0)/dx (C == Cd)
	virtual gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const = 0;  // Du(k) = D(k) * u(k) (D == Dd)
	//virtual gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const = 0;  // V(k) = dh(k, x(k), u(k), 0)/dv

	// noise covariance matrices
	virtual gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const = 0;  // Q or Qd = W * Q * W^T
	virtual gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const = 0;  // R or Rd = V * R * V^T

	// actual measurement
	//virtual gsl_vector * doGetMeasurement(const size_t step, const gsl_vector *state) const = 0;

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


#endif  // __SWL_RND_UTIL__DISCRETE_NONLINEAR_STOCHASTIC_SYSTEM__H_
