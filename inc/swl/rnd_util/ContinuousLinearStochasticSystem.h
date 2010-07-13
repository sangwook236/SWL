#if !defined(__SWL_RND_UTIL__CONTINUOUS_LINEAR_STOCHASTIC_SYSTEM__H_)
#define __SWL_RND_UTIL__CONTINUOUS_LINEAR_STOCHASTIC_SYSTEM__H_ 1


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

// the stochastic differential equation: dx/dt(t) = A(t) * x(t) + B(t) * u(t) + W(t) * w(t)
// the observation equation: y(t) = C(t) * x(t) + D(t) * u(t) + V(t) * v(t)
class ContinuousLinearStochasticSystem
{
public:
	//typedef ContinuousLinearStochasticSystem base_type;

protected:
	ContinuousLinearStochasticSystem(const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: stateDim_(stateDim), inputDim_(inputDim), outputDim_(outputDim)
	{}
public:
	virtual ~ContinuousLinearStochasticSystem()  {}

private:
	ContinuousLinearStochasticSystem(const ContinuousLinearStochasticSystem &rhs);
	ContinuousLinearStochasticSystem & operator=(const ContinuousLinearStochasticSystem &rhs);

public:
	// the stochastic differential equation
	virtual gsl_matrix * getSystemMatrix(const double time, const gsl_vector *state) const = 0;  // A(t)
	virtual gsl_vector * getControlInput(const double time, const gsl_vector *state) const = 0;  // Bu(t) = B(t) * u(t)
	//virtual gsl_matrix * getProcessNoiseCouplingMatrix(const double time) const = 0;  // W(t)

	// the observation equation
	virtual gsl_matrix * getOutputMatrix(const double time, const gsl_vector *state) const = 0;  // C(t)
	virtual gsl_vector * getMeasurementInput(const double time, const gsl_vector *state) const = 0;  // Du(t) = D(t) * u(t)
	//virtual gsl_matrix * getMeasurementNoiseCouplingMatrix(const double time) const = 0;  // V(t)

	virtual gsl_matrix * getProcessNoiseCovarianceMatrix(const double time) const = 0;  // Q or Qd = W * Q * W^T
	virtual gsl_matrix * getMeasurementNoiseCovarianceMatrix(const double time) const = 0;  // R or Rd = V * R * V^T

	// actual measurement
	//virtual gsl_vector * doGetMeasurement(const double time, const gsl_vector *state) const = 0;

	size_t getStateDim() const  {  return stateDim_;  }
	size_t getInputDim() const  {  return inputDim_;  }
	size_t getOutputDim() const  {  return outputDim_;  }

protected:
	const size_t stateDim_;
	const size_t inputDim_;
	const size_t outputDim_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__CONTINUOUS_LINEAR_STOCHASTIC_SYSTEM__H_
