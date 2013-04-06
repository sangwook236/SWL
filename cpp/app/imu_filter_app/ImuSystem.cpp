//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/KalmanFilter.h"
#include "ImuSystem.h"
#include <cmath>
#include <ctime>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
//

bool ImuSystem::runStep(DiscreteKalmanFilter &filter, size_t step, const gsl_vector *Bu, const gsl_vector *Du, const gsl_vector *actualMeasurement, double &prioriEstimate, double &posterioriEstimate) const
{
#if 0
	// method #1
	// 1-based time step. 0-th time step is initial
	//size_t step = 0;
	//while (step < Nstep)
	{
		// 0. initial estimates: x(0) & P(0)

		// 1. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		if (!filter.updateTime(step, Bu))
			return false;

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			//const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			prioriEstimate = gsl_vector_get(x_hat, 0);
		}

		// advance time step
		++step;

		// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		if (!filter.updateMeasurement(step, actualMeasurement, Du))
			return false;

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			//const gsl_matrix *K = filter.getKalmanGain();
			//const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			posterioriEstimate = gsl_vector_get(x_hat, 0);
		}
	}
#else
	// method #2
	// 0-based time step. 0-th time step is initial
	//size_t step = 0;
	//while (step < Nstep)
	{
		// 0. initial estimates: x-(0) & P-(0)

		// 1. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		if (!filter.updateMeasurement(step, actualMeasurement, Du))
			return false;

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			//const gsl_matrix *K = filter.getKalmanGain();
			//const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			posterioriEstimate = gsl_vector_get(x_hat, 0);
		}

		// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		if (!filter.updateTime(step, Bu))
			return false;

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			//const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			prioriEstimate = gsl_vector_get(x_hat, 0);
		}

		// advance time step
		++step;
	}
#endif

	return true;
}

//--------------------------------------------------------------------------
//

AccelSystem::AccelSystem(const double Ts, const double beta, const double Qv, const double Qa, const double Qb, const double Ra)
: base_type(stateDim, inputDim, outputDim),
  Ts_(Ts), beta_(beta), Qv_(Qv), Qa_(Qa), Qb_(Qb), Ra_(Ra),
  Phi_(NULL), C_(NULL), Qd_(NULL), Rd_(NULL), Bd_(NULL)
{
#if 0
	// Phi = exp(A * Ts) = [ 1 Ts 0 ; 0 1 0 ; 0 0 1 ]
	//	A = [ 0 1 0 ; 0 0 0 ; 0 0 0 ]
	Phi_ = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(Phi_);
	gsl_matrix_set(Phi_, 0, 1, Ts_);

	// C = [ 0 1 1 ]
	C_ = gsl_matrix_alloc(outputDim, stateDim);
	gsl_matrix_set(C_, 0, 0, 0.0);  gsl_matrix_set(C_, 0, 1, 1.0);  gsl_matrix_set(C_, 0, 2, 1.0);

	// continuous system  -->  discrete system
	// Qd = integrate(Phi(t) * W(t) * Q(t) * W(t)^T * Phi(t)^T, {t, 0, Ts})
	//	Q = [ Qv 0 0 ; 0 Qa 0 ; 0 0 Qb ]
	//	W = I
	//	Qd = [ (Qa*Ts^3)/3 + Qv*Ts, (Qa*Ts^2)/2,     0 ]
	//	     [         (Qa*Ts^2)/2,       Qa*Ts,     0 ]
	//	     [                   0,           0, Qb*Ts ]
	Qd_ = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set(Qd_, 0, 0, (Qa_*Ts_*Ts_*Ts_)/3 + Qv_*Ts_);  gsl_matrix_set(Qd_, 0, 1, (Qa_*Ts_*Ts_)/2);  gsl_matrix_set(Qd_, 0, 2, 0);
	gsl_matrix_set(Qd_, 1, 0, (Qa_*Ts_*Ts_)/2);  gsl_matrix_set(Qd_, 1, 1, Qa_*Ts_);  gsl_matrix_set(Qd_, 1, 2, 0);
	gsl_matrix_set(Qd_, 2, 0, 0);  gsl_matrix_set(Qd_, 2, 1, 0);  gsl_matrix_set(Qd_, 2, 2, Qb_*Ts_);

	// Rd = V * R * V^T since V = I
	Rd_ = gsl_matrix_alloc(outputDim, outputDim);
	gsl_matrix_set_all(Rd_, Ra_);

	// input matrix: Bd = integrate(exp(A * t), {t, 0, Ts}) * B
	//	B = [ 1 ; 0 ; 0 ]
	//	integrate(exp(A * t), {t, 0, Ts}) = [ Ts Ts^2/2 0 ; 0 Ts 0 ; 0 0 Ts ]
	//	Bd = [ Ts ; 0 ; 0 ]
	Bd_ = gsl_matrix_alloc(stateDim, inputDim);
	gsl_matrix_set(Bd_, 0, 0, Ts_);  gsl_matrix_set(Bd_, 1, 0, 0.0);  gsl_matrix_set(Bd_, 1, 0, 0.0);
#else
	// Phi = exp(A * Ts) = [ 1 Ts 0 ; 0 1 0 ; 0 0 exp(-beta * Ts) ]
	//	A = [ 0 1 0 ; 0 0 0 ; 0 0 -beta ]
	Phi_ = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(Phi_);
	gsl_matrix_set(Phi_, 0, 1, Ts_);
	gsl_matrix_set(Phi_, 2, 2, std::exp(-beta_ * Ts_));

	// C = [ 0 1 1 ]
	C_ = gsl_matrix_alloc(outputDim, stateDim);
	gsl_matrix_set(C_, 0, 0, 0.0);  gsl_matrix_set(C_, 0, 1, 1.0);  gsl_matrix_set(C_, 0, 2, 1.0);

	// continuous system  -->  discrete system
	// Qd = integrate(Phi(t) * W(t) * Q(t) * W(t)^T * Phi(t)^T, {t, 0, Ts})
	//	Q = [ Qv 0 0 ; 0 Qa 0 ; 0 0 Qb ]
	//	W = I
	//	Qd = [ (Qa*Ts^3)/3 + Qv*Ts, (Qa*Ts^2)/2,     0 ]
	//	     [         (Qa*Ts^2)/2,       Qa*Ts,     0 ]
	//	     [                   0,           0, Qb*Ts ]
	Qd_ = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set(Qd_, 0, 0, (Qa_*Ts_*Ts_*Ts_)/3 + Qv_*Ts_);  gsl_matrix_set(Qd_, 0, 1, (Qa_*Ts_*Ts_)/2);  gsl_matrix_set(Qd_, 0, 2, 0);
	gsl_matrix_set(Qd_, 1, 0, (Qa_*Ts_*Ts_)/2);  gsl_matrix_set(Qd_, 1, 1, Qa_*Ts_);  gsl_matrix_set(Qd_, 1, 2, 0);
	gsl_matrix_set(Qd_, 2, 0, 0);  gsl_matrix_set(Qd_, 2, 1, 0);  gsl_matrix_set(Qd_, 2, 2, Qb_*Ts_);

	// Rd = V * R * V^T since V = I
	Rd_ = gsl_matrix_alloc(outputDim, outputDim);
	gsl_matrix_set_all(Rd_, Ra_);

	// input matrix: Bd = integrate(exp(A * t), {t, 0, Ts}) * B
	//	B = [ 1 ; 0 ; 0 ]
	//	integrate(exp(A * t), {t, 0, Ts}) = [ Ts Ts^2/2 0 ; 0 Ts 0 ; 0 0 (exp(beta * Ts) - 1)/beta ]
	//	Bd = [ Ts ; 0 ; 0 ]
	Bd_ = gsl_matrix_alloc(stateDim, inputDim);
	gsl_matrix_set(Bd_, 0, 0, Ts_);  gsl_matrix_set(Bd_, 1, 0, 0.0);  gsl_matrix_set(Bd_, 1, 0, 0.0);
#endif
}

AccelSystem::~AccelSystem()
{
	gsl_matrix_free(Phi_);  Phi_ = NULL;
	gsl_matrix_free(C_);  C_ = NULL;
	gsl_matrix_free(Qd_);  Qd_ = NULL;
	gsl_matrix_free(Rd_);  Rd_ = NULL;

	gsl_matrix_free(Bd_);  Bd_ = NULL;
}

//--------------------------------------------------------------------------
//

GyroSystem::GyroSystem(const double Ts, const double beta, const double Qw, const double Qb, const double Rg)
: base_type(stateDim, inputDim, outputDim),
  Ts_(Ts), beta_(beta), Qw_(Qw), Qb_(Qb), Rg_(Rg),
  Phi_(NULL), C_(NULL), Qd_(NULL), Rd_(NULL)
{
#if 0
	// Phi = exp(A * Ts) = [ 1 0 ; 0 1 ]
	//	A = [ 0 0 ; 0 0 ]
	Phi_ = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(Phi_);

	// C = [ 1 1 ]
	C_ = gsl_matrix_alloc(outputDim, stateDim);
	gsl_matrix_set_identity(C_);
	gsl_matrix_set(C_, 0, 0, 1.0);  gsl_matrix_set(C_, 0, 1, 1.0);

	// continuous system  -->  discrete system
	// Qd = integrate(Phi(t) * W(t) * Q(t) * W(t)^T * Phi(t)^T, {t, 0, Ts})
	//	Q = [ Qw 0 ; 0 Qb ]
	//	W = I
	//	Qd = [ Qw*Ts 0 ; 0 Qb*Ts ]
	Qd_ = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_zero(Qd_);
	gsl_matrix_set(Qd_, 0, 0, Qw_*Ts_);  gsl_matrix_set(Qd_, 1, 1, Qb_*Ts_);
	// Rd = V * R * V^T where V = I
	Rd_ = gsl_matrix_alloc(outputDim, outputDim);
	gsl_matrix_set_all(Rd_, Rg_);

	// no control input: Bu = Bd * u where u(t) = 1
	//	Bu = Bd * u where Bd = integrate(exp(A * t), {t, 0, Ts}) * B or A^-1 * (Ad - I) * B if A is nonsingular
	//	Ad = Phi = exp(A * Ts)
	//	B = [ 0 ; 0 ]
#else
	// Phi = exp(A * Ts) = [ 1 0 ; 0 exp(-beta * Ts) ]
	//	A = [ 0 0 ; 0 -beta ]
	Phi_ = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(Phi_);
	gsl_matrix_set(Phi_, 1, 1, std::exp(-beta_ * Ts_));

	// C = [ 1 1 ]
	C_ = gsl_matrix_alloc(outputDim, stateDim);
	gsl_matrix_set_identity(C_);
	gsl_matrix_set(C_, 0, 0, 1.0);  gsl_matrix_set(C_, 0, 1, 1.0);

	// continuous system  -->  discrete system
	// Qd = integrate(Phi(t) * W(t) * Q(t) * W(t)^T * Phi(t)^T, {t, 0, Ts})
	//	Q = [ Qw 0 ; 0 Qb ]
	//	W = I
	//	Qd = [ Qw*Ts 0 ; 0 Qb*Ts ]
	Qd_ = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_zero(Qd_);
	gsl_matrix_set(Qd_, 0, 0, Qw_*Ts_);  gsl_matrix_set(Qd_, 1, 1, Qb_*Ts_);
	// Rd = V * R * V^T where V = I
	Rd_ = gsl_matrix_alloc(outputDim, outputDim);
	gsl_matrix_set_all(Rd_, Rg_);

	// no control input: Bu = Bd * u where u(t) = 1
	//	Bu = Bd * u where Bd = integrate(exp(A * t), {t, 0, Ts}) * B or A^-1 * (Ad - I) * B if A is nonsingular
	//	Ad = Phi = exp(A * Ts)
	//	B = [ 0 ; 0 ]
#endif
}

GyroSystem::~GyroSystem()
{
	gsl_matrix_free(Phi_);  Phi_ = NULL;
	gsl_matrix_free(C_);  C_ = NULL;
	gsl_matrix_free(Qd_);  Qd_ = NULL;
	gsl_matrix_free(Rd_);  Rd_ = NULL;
}

}  // namespace swl
