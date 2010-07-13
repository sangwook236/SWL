#if !defined(__SWL_RND_UTIL__DISCRETE_LINEAR_STOCHASTIC_SYSTEM__H_)
#define __SWL_RND_UTIL__DISCRETE_LINEAR_STOCHASTIC_SYSTEM__H_ 1


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

// the stochastic differential equation: x(k+1) = Phi(k) * x(k) + Bd(k) * u(k) + W(k) * w(k)
// the observation equation: y(k) = Cd(k) * x(k) + Dd(k) * u(k) + V(k) * v(k)
class DiscreteLinearStochasticSystem
{
public:
	//typedef DiscreteLinearStochasticSystem base_type;

protected:
	DiscreteLinearStochasticSystem(const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: stateDim_(stateDim), inputDim_(inputDim), outputDim_(outputDim)
	{}
public:
	virtual ~DiscreteLinearStochasticSystem()  {}

private:
	DiscreteLinearStochasticSystem(const DiscreteLinearStochasticSystem &rhs);
	DiscreteLinearStochasticSystem & operator=(const DiscreteLinearStochasticSystem &rhs);

public:
	// the stochastic differential equation
	virtual gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const = 0;  // Phi(k) = exp(A(k) * Ts)
	virtual gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const = 0;  // Bu(k) = Bd(k) * u(k)
	//virtual gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const = 0;  // W(k)

	// the observation equation
	virtual gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const = 0;  // Cd(k) (C == Cd)
	virtual gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const = 0;  // Du(k) = D(k) * u(k) (D == Dd)
	//virtual gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const = 0;  // V(k)

	virtual gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const = 0;  // Q or Qd = W * Q * W^T
	virtual gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const = 0;  // R or Rd = V * R * V^T

	// actual measurement
	//virtual gsl_vector * doGetMeasurement(const size_t step, const gsl_vector *state) const = 0;

	size_t getStateDim() const  {  return stateDim_;  }
	size_t getInputDim() const  {  return inputDim_;  }
	size_t getOutputDim() const  {  return outputDim_;  }

protected:
	const size_t stateDim_;
	const size_t inputDim_;
	const size_t outputDim_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__DISCRETE_LINEAR_STOCHASTIC_SYSTEM__H_
