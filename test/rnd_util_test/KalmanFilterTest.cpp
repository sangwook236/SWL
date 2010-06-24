#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/KalmanFilter.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

// "Kalman Filtering: Theory and Practice Using MATLAB" Example 4.1 (pp. 123)
class SimpleSystemKalmanFilter: public swl::KalmanFilter
{
public:
	typedef swl::KalmanFilter base_type;

public:
	SimpleSystemKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: base_type(x0, P0, stateDim, inputDim, outputDim),
	  Phi_(NULL), G_(NULL), H_(NULL), Qd_(NULL), R_(NULL), y_tilde_(NULL)
	{
		// Phi = exp(T * A)
		Phi_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set(Phi_, 0, 0, 1.0);

		//
		G_ = gsl_matrix_alloc(stateDim_, inputDim_);
		gsl_matrix_set(G_, 0, 0, 1.0);

		//
		H_ = gsl_matrix_alloc(outputDim_, stateDim_);
		gsl_matrix_set(H_, 0, 0, 1.0);

		// Qd = Gamma * Q * Gamma^T
		Qd_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set(Qd_, 0, 0, 1.0);

		//
		R_ = gsl_matrix_alloc(outputDim_, outputDim_);
		gsl_matrix_set(R_, 0, 0, 2.0);

		// no control input
		Bu_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(Bu_);

		//
		y_tilde_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(y_tilde_);
	}
	~SimpleSystemKalmanFilter()
	{
		gsl_matrix_free(Phi_);  Phi_ = NULL;
		gsl_matrix_free(G_);  G_ = NULL;
		gsl_matrix_free(H_);  H_ = NULL;
		gsl_matrix_free(Qd_);  Qd_ = NULL;
		gsl_matrix_free(R_);  R_ = NULL;

		gsl_vector_free(Bu_);  Bu_ = NULL;
		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;
	}

private:
	SimpleSystemKalmanFilter(const SimpleSystemKalmanFilter &rhs);
	SimpleSystemKalmanFilter & operator=(const SimpleSystemKalmanFilter &rhs);

private:
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		if (1 == step) gsl_vector_set(y_tilde_, 0, 2.0);
		else if (2 == step) gsl_vector_set(y_tilde_, 0, 3.0);
		else throw std::runtime_error("undefined measurement");

		return y_tilde_;
	}

	// for continuous Kalman filter
	/*virtual*/ gsl_matrix * getSystemMatrix(const size_t step) const
	{  throw std::runtime_error("this function doesn't have to be called");  }

	// for discrete Kalman filter
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step) const  {  return Phi_;  }
	/*virtual*/ gsl_matrix * getInputMatrix(const size_t step) const
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step) const  {  return H_;  }
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd, but not Q
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return R_;  }

	/*virtual*/ gsl_vector * getControlInput(const size_t step) const  {  return Bu_;  }  // Bu = Bd(t) * u(t)

protected:
	gsl_matrix *Phi_;
	gsl_matrix *G_;  // Gamma
	gsl_matrix *H_;
	gsl_matrix *Qd_;
	gsl_matrix *R_;

	// control input: Bu = Bd(t) * u(t). Bd = A^-1 * (Ad - I) * B
	gsl_vector *Bu_;
	// actual measurement
	gsl_vector *y_tilde_;
};

// "The Global Positioning System and Inertial Navigation" Example (pp. 110)
class AidedINSKalmanFilter: public swl::KalmanFilter
{
public:
	typedef swl::KalmanFilter base_type;

private:
	static const double Rv;
	static const double Qb;
	static const double Rp;

public:
	AidedINSKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: base_type(x0, P0, stateDim, inputDim, outputDim),
	  Phi_(NULL), G_(NULL), H_(NULL), Qd_(NULL), R_(NULL), y_tilde_(NULL)
	{
		Phi_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set_identity(Phi_);
		gsl_matrix_set(Phi_, 0, 1, 1.0);  gsl_matrix_set(Phi_, 0, 2, 0.5);  gsl_matrix_set(Phi_, 1, 2, 1.0);

		//
		G_ = gsl_matrix_alloc(stateDim_, inputDim_);
		gsl_matrix_set_zero(G_);
		gsl_matrix_set(G_, 1, 0, 1.0);  gsl_matrix_set(G_, 2, 1, 1.0);

		//
		H_ = gsl_matrix_alloc(outputDim_, stateDim_);
		gsl_matrix_set_identity(H_);
		gsl_matrix_set(H_, 0, 0, 1.0);  gsl_matrix_set(H_, 0, 1, 0.0);  gsl_matrix_set(H_, 0, 2, 1.0);

		// Qd = Gamma * Q * Gamma^T
		Qd_ = gsl_matrix_alloc(stateDim_, stateDim_);
#if 0
		// Q, but not Qd
		gsl_matrix_set_zero(Qd_);
		gsl_matrix_set(Qd_, 0, 0, Rv);  gsl_matrix_set(Qd_, 1, 1, Qb);
#else
		// Qd, but not Q
		gsl_matrix_set(Qd_, 0, 0, Rv/3+Qb/20);  gsl_matrix_set(Qd_, 0, 1, Rv/2+Qb/8);  gsl_matrix_set(Qd_, 0, 2, Qb/6);
		gsl_matrix_set(Qd_, 1, 0, Rv/2+Qb/8);  gsl_matrix_set(Qd_, 1, 1, Rv+Qb/3);  gsl_matrix_set(Qd_, 1, 2, Qb/2);
		gsl_matrix_set(Qd_, 2, 0, Qb/6);  gsl_matrix_set(Qd_, 2, 1, Qb/2);  gsl_matrix_set(Qd_, 2, 2, Qb);
#endif

		//
		R_ = gsl_matrix_alloc(outputDim_, outputDim_);
		gsl_matrix_set_all(R_, Rp);

		// no control input
		Bu_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(Bu_);

		//
		y_tilde_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(y_tilde_);
	}
	~AidedINSKalmanFilter()
	{
		gsl_matrix_free(Phi_);  Phi_ = NULL;
		gsl_matrix_free(G_);  G_ = NULL;
		gsl_matrix_free(H_);  H_ = NULL;
		gsl_matrix_free(Qd_);  Qd_ = NULL;
		gsl_matrix_free(R_);  R_ = NULL;

		gsl_vector_free(Bu_);  Bu_ = NULL;
		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;
	}

private:
	AidedINSKalmanFilter(const AidedINSKalmanFilter &rhs);
	AidedINSKalmanFilter & operator=(const AidedINSKalmanFilter &rhs);

private:
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		const gsl_vector *x = getState();

#if 0
		gsl_vector_set(y_tilde_, 0, gsl_vector_get(x, 0));  // measurement (no noise)
#elif 0
		typedef boost::minstd_rand base_generator_type;
		typedef boost::uniform_real<> distribution_type;
		typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

		base_generator_type generator(static_cast<unsigned int>(std::time(0)));
		//generator.seed(static_cast<unsigned int>(std::time(0)));
		generator_type uni_gen(generator, distribution_type(-100, 101));

		gsl_vector_set(y_tilde_, 0, gsl_vector_get(x, 0) + uni_gen());  // measurement (white noise)
#else
		typedef boost::minstd_rand base_generator_type;
		typedef boost::normal_distribution<> distribution_type;
		typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

		base_generator_type generator(static_cast<unsigned int>(std::time(0)));
		//generator.seed(static_cast<unsigned int>(std::time(0)));
		generator_type normal_gen(generator, distribution_type(0.0, std::sqrt(Rp)));

		gsl_vector_set(y_tilde_, 0, gsl_vector_get(x, 0) + normal_gen());  // measurement (gaussian noise)
#endif

		return y_tilde_;
	}

	// for continuous Kalman filter
	/*virtual*/ gsl_matrix * getSystemMatrix(const size_t step) const
	{  throw std::runtime_error("this function doesn't have to be called");  }

	// for discrete Kalman filter
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step) const  {  return Phi_;  }
	/*virtual*/ gsl_matrix * getInputMatrix(const size_t step) const
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step) const  {  return H_;  }
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd, but not Q
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return R_;  }

	/*virtual*/ gsl_vector * getControlInput(const size_t step) const  {  return Bu_;  }  // Bu = Bd(t) * u(t)

protected:
	gsl_matrix *Phi_;
	gsl_matrix *G_;  // Gamma
	gsl_matrix *H_;
	gsl_matrix *Qd_;
	gsl_matrix *R_;

	// control input: Bu = Bd(t) * u(t). Bd = A^-1 * (Ad - I) * B
	gsl_vector *Bu_;
	// actual measurement
	gsl_vector *y_tilde_;
};

/*static*/ const double AidedINSKalmanFilter::Rv = 2.5e-3;
/*static*/ const double AidedINSKalmanFilter::Qb = 1.0e-6;
/*static*/ const double AidedINSKalmanFilter::Rp = 3.0;

// "Kalman Filtering: Theory and Practice Using MATLAB" Example 4.3 (pp. 156)
//	1) continuous ==> discrete
//	2) driving force exits
class LinearMassStringDamperSystemKalmanFilter: public swl::KalmanFilter
{
public:
	typedef swl::KalmanFilter base_type;

private:
	static const double T;
	static const double zeta;
	static const double omega;
	static const double Q;
	static const double R;

public:
	LinearMassStringDamperSystemKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: base_type(x0, P0, stateDim, inputDim, outputDim),
	  Phi_(NULL), G_(NULL), H_(NULL), Qd_(NULL), R_(NULL), y_tilde_(NULL)
	{
		//A_ = gsl_matrix_alloc(stateDim_, stateDim_);
		//gsl_matrix_set(A_, 0, 0, 0.0);  gsl_matrix_set(A_, 0, 1, 1.0);
		//gsl_matrix_set(A_, 1, 0, -omega * omega);  gsl_matrix_set(A_, 1, 1, -2.0 * zeta * omega);

		const double lambda = std::exp(-T * omega * zeta);
		const double psi    = 1.0 - zeta * zeta;
		const double xi     = std::sqrt(psi);
		const double theta  = xi * omega * T;
		const double c = cos(theta);
		const double s = sin(theta);

		// Phi = exp(T * A)
		Phi_ = gsl_matrix_alloc(stateDim_, stateDim_);
#if 1
		gsl_matrix_set(Phi_, 0, 0, lambda*c + zeta*s/xi);  gsl_matrix_set(Phi_, 0, 1, lambda*s/(omega*xi));
		gsl_matrix_set(Phi_, 1, 0, -omega*lambda*s/xi);  gsl_matrix_set(Phi_, 1, 1, lambda*c - zeta*s/xi);
#else
		// my calculation result
		gsl_matrix_set(Phi_, 0, 0, lambda*(c+zeta*s/xi));  gsl_matrix_set(Phi_, 0, 1, lambda*s/(omega*xi));
		gsl_matrix_set(Phi_, 1, 0, -omega*lambda*s/xi);  gsl_matrix_set(Phi_, 1, 1, lambda*(c-zeta*s/xi));
#endif

		//
		G_ = gsl_matrix_alloc(stateDim_, inputDim_);
		gsl_matrix_set(G_, 0, 0, 0.0);  gsl_matrix_set(G_, 1, 0, 1.0);

		//
		H_ = gsl_matrix_alloc(outputDim_, stateDim_);
		gsl_matrix_set(H_, 0, 0, 1.0);  gsl_matrix_set(H_, 0, 1, 0.0);

		// Qd = Gamma * Q * Gamma^T
		Qd_ = gsl_matrix_alloc(stateDim_, stateDim_);
		const double l1  = zeta * zeta;
		const double l4  = std::exp(-2.0 * omega * zeta * T);
		const double l6  = std::sqrt(1-l1);
		const double l8  = l6 * omega * T;
		const double l9  = std::cos(2.0 * l8);
		const double l11 = l4 * l9 * l1;
		const double l15 = l4 * std::sin(2.0 * l8) * l6 * zeta;
		const double l19 = 1.0 / (l1 - 1.0);
		const double l20 = omega * omega;
		const double l24 = 1.0 / zeta;
		const double l32 = l4 * (l9 - 1.0) * l19 / l20;

		gsl_matrix_set(Qd_, 0, 0, 0.25*Q * (l1-l11+l15-1+l4)*l19*l24);  gsl_matrix_set(Qd_, 0, 1, 0.25*Q * l32);
		gsl_matrix_set(Qd_, 1, 0, 0.25*Q * l32);  gsl_matrix_set(Qd_, 1, 1, 0.25*Q * (l1-l11-l15-1+l4)*l19*l24/omega);

		//
		R_ = gsl_matrix_alloc(outputDim_, outputDim_);
		gsl_matrix_set_all(R_, R);

		// driving force: Bu = Bd(t) * u(t)
		//	Bd = A^-1 * (Ad - I) * B. Ad = Phi = exp(A * T). B = [0 ; 12]. u(t) = 1
		Bu_ = gsl_vector_alloc(stateDim_);
#if 0
		gsl_vector_set(Bu_, 0, 12.0 * (1.0 - lambda*(c-zeta*s*xi/psi)) / (omega*omega));  gsl_vector_set(Bu_, 1, 12.0 * lambda*s / (omega*xi));
#else
		gsl_vector_set(Bu_, 0, 12.0 * (1.0 - lambda*(c-zeta*s/xi)) / (omega*omega));  gsl_vector_set(Bu_, 1, 12.0 * lambda*s / (omega*xi));
#endif

		//
		y_tilde_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(y_tilde_);
	}
	~LinearMassStringDamperSystemKalmanFilter()
	{
		gsl_matrix_free(Phi_);  Phi_ = NULL;
		gsl_matrix_free(G_);  G_ = NULL;
		gsl_matrix_free(H_);  H_ = NULL;
		gsl_matrix_free(Qd_);  Qd_ = NULL;
		gsl_matrix_free(R_);  R_ = NULL;

		gsl_vector_free(Bu_);  Bu_ = NULL;
		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;
	}

private:
	LinearMassStringDamperSystemKalmanFilter(const LinearMassStringDamperSystemKalmanFilter &rhs);
	LinearMassStringDamperSystemKalmanFilter & operator=(const LinearMassStringDamperSystemKalmanFilter &rhs);

private:
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		const gsl_vector *x = getState();
		gsl_vector_set(y_tilde_, 0, gsl_vector_get(x, 0));  // measurement (no noise)
		return y_tilde_;
	}

	// for continuous Kalman filter
	/*virtual*/ gsl_matrix * getSystemMatrix(const size_t step) const
	{  throw std::runtime_error("this function doesn't have to be called");  }

	// for discrete Kalman filter
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step) const  {  return Phi_;  }
	/*virtual*/ gsl_matrix * getInputMatrix(const size_t step) const
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step) const  {  return H_;  }
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd, but not Q
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return R_;  }

	/*virtual*/ gsl_vector * getControlInput(const size_t step) const  {  return Bu_;  }  // Bu = Bd(t) * u(t)

protected:
	gsl_matrix *Phi_;
	gsl_matrix *G_;  // Gamma
	gsl_matrix *H_;
	gsl_matrix *Qd_;
	gsl_matrix *R_;

	// control input: Bu = Bd(t) * u(t). Bd = A^-1 * (Ad - I) * B
	gsl_vector *Bu_;
	// actual measurement
	gsl_vector *y_tilde_;

	// driving force
	gsl_vector *force_;
};

/*static*/ const double LinearMassStringDamperSystemKalmanFilter::T = 0.01;
/*static*/ const double LinearMassStringDamperSystemKalmanFilter::zeta = 0.2;
/*static*/ const double LinearMassStringDamperSystemKalmanFilter::omega = 5.0;
/*static*/ const double LinearMassStringDamperSystemKalmanFilter::Q = 4.47;
/*static*/ const double LinearMassStringDamperSystemKalmanFilter::R = 0.01;

// "Kalman Filtering: Theory and Practice Using MATLAB" Example 4.4 (pp. 156)
class RadarTrackingSystemKalmanFilter: public swl::KalmanFilter
{
public:
	typedef swl::KalmanFilter base_type;

public:
	static const double T;
	static const double rho;
	static const double var_r;
	static const double var_theta;
	static const double var_1;
	static const double var_2;

public:
	RadarTrackingSystemKalmanFilter(const gsl_vector *x0, const gsl_matrix *P0, const size_t stateDim, const size_t inputDim, const size_t outputDim)
	: base_type(x0, P0, stateDim, inputDim, outputDim),
	  Phi_(NULL), G_(NULL), H_(NULL), Qd_(NULL), R_(NULL), y_tilde_(NULL)
	{
		Phi_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set_identity(Phi_);
		gsl_matrix_set(Phi_, 0, 1, T);  gsl_matrix_set(Phi_, 1, 2, 1.0);  gsl_matrix_set(Phi_, 2, 2, rho);
		gsl_matrix_set(Phi_, 3, 4, T);  gsl_matrix_set(Phi_, 4, 5, 1.0);  gsl_matrix_set(Phi_, 5, 5, rho);

		//
		G_ = gsl_matrix_alloc(stateDim_, inputDim_);
		gsl_matrix_set_zero(G_);
		gsl_matrix_set(G_, 2, 0, 1.0);  gsl_matrix_set(G_, 5, 1, 1.0);

		//
		H_ = gsl_matrix_alloc(outputDim_, stateDim_);
		gsl_matrix_set_zero(H_);
		gsl_matrix_set(H_, 0, 0, 1.0);  gsl_matrix_set(H_, 1, 3, 1.0);

		// Qd = Gamma * Q * Gamma^T
		Qd_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set_zero(Qd_);
		gsl_matrix_set(Qd_, 2, 2, var_1);  gsl_matrix_set(Qd_, 5, 5, var_2);

		//
		R_ = gsl_matrix_alloc(outputDim_, outputDim_);
		gsl_matrix_set_zero(R_);
		gsl_matrix_set(R_, 0, 0, var_r);  gsl_matrix_set(R_, 1, 1, var_theta);

		// no control input
		Bu_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(Bu_);

		//
		y_tilde_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(y_tilde_);
	}
	~RadarTrackingSystemKalmanFilter()
	{
		gsl_matrix_free(Phi_);  Phi_ = NULL;
		gsl_matrix_free(G_);  G_ = NULL;
		gsl_matrix_free(H_);  H_ = NULL;
		gsl_matrix_free(Qd_);  Qd_ = NULL;
		gsl_matrix_free(R_);  R_ = NULL;

		gsl_vector_free(Bu_);  Bu_ = NULL;
		gsl_vector_free(y_tilde_);  y_tilde_ = NULL;
	}

private:
	RadarTrackingSystemKalmanFilter(const RadarTrackingSystemKalmanFilter &rhs);
	RadarTrackingSystemKalmanFilter & operator=(const RadarTrackingSystemKalmanFilter &rhs);

private:
	/*virtual*/ gsl_vector * getMeasurement(const size_t step) const
	{
		const gsl_vector *x = getState();

		typedef boost::minstd_rand base_generator_type;
		typedef boost::normal_distribution<> distribution_type;
		typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

		base_generator_type generator(static_cast<unsigned int>(std::time(0)));
		//generator.seed(static_cast<unsigned int>(std::time(0)));
		generator_type v1_noise(generator, distribution_type(0.0, std::sqrt(var_r)));
		generator_type v2_noise(generator, distribution_type(0.0, std::sqrt(var_theta)));

		gsl_vector_set(y_tilde_, 0, gsl_vector_get(x, 0) + v1_noise());  // measurement (white noise)
		gsl_vector_set(y_tilde_, 0, gsl_vector_get(x, 3) + v2_noise());  // measurement (white noise)

		return y_tilde_;
	}

	// for continuous Kalman filter
	/*virtual*/ gsl_matrix * getSystemMatrix(const size_t step) const
	{
		throw std::runtime_error("this function doesn't have to be called");
	}

	// for discrete Kalman filter
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step) const  {  return Phi_;  }
	/*virtual*/ gsl_matrix * getInputMatrix(const size_t step) const
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step) const  {  return H_;  }
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd, but not Q
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return R_;  }

	/*virtual*/ gsl_vector * getControlInput(const size_t step) const  {  return Bu_;  }  // Bu = Bd(t) * u(t)

protected:
	gsl_matrix *Phi_;
	gsl_matrix *G_;  // Gamma
	gsl_matrix *H_;
	gsl_matrix *Qd_;
	gsl_matrix *R_;

	// control input: Bu = Bd(t) * u(t). Bd = A^-1 * (Ad - I) * B
	gsl_vector *Bu_;
	// actual measurement
	gsl_vector *y_tilde_;
};

///*static*/ const double RadarTrackingSystemKalmanFilter::T = 5.0;
///*static*/ const double RadarTrackingSystemKalmanFilter::T = 10.0;
/*static*/ const double RadarTrackingSystemKalmanFilter::T = 15.0;
/*static*/ const double RadarTrackingSystemKalmanFilter::rho = 0.5;
/*static*/ const double RadarTrackingSystemKalmanFilter::var_r = 1000.0 * 1000.0;
/*static*/ const double RadarTrackingSystemKalmanFilter::var_theta = 0.017 * 0.017;
/*static*/ const double RadarTrackingSystemKalmanFilter::var_1 = (103.0/3.0) * (103.0/3.0);
/*static*/ const double RadarTrackingSystemKalmanFilter::var_2 = 1.3e-8;

void simple_system_kalman_filter()
{
	const size_t stateDim = 1;
	const size_t inputDim = 1;
	const size_t outputDim = 1;

	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_vector_set(x0, 0, 1.0);
	gsl_matrix_set(P0, 0, 0, 10.0);

	SimpleSystemKalmanFilter filter(x0, P0, stateDim, inputDim, outputDim);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	const size_t Nstep = 2;
	std::vector<double> state, gain, errVar;
	state.reserve(Nstep);
	gain.reserve(Nstep);
	errVar.reserve(Nstep);

	for (size_t i = 0; i < Nstep; ++i)
	{
		const bool retval = filter.propagate(i + 1);  // 1-based time step. 0-th time step is initial
		
		const gsl_vector *x = filter.getState();
		const gsl_matrix *K = filter.getKalmanGain();
		const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

		state.push_back(gsl_vector_get(x, 0));
		gain.push_back(gsl_matrix_get(K, 0, 0));
		errVar.push_back(gsl_matrix_get(P, 0, 0));
	}
}

void aided_INS_kalman_filter()
{
	const size_t stateDim = 3;
	const size_t inputDim = 2;
	const size_t outputDim = 1;

	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_vector_set_zero(x0);
	gsl_matrix_set_zero(P0);

	AidedINSKalmanFilter filter(x0, P0, stateDim, inputDim, outputDim);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	const size_t Nstep = 100;
	std::vector<double> pos, vel, bias;
	std::vector<double> posGain, velGain, biasGain;
	std::vector<double> posErrVar, velErrVar, biasErrVar;
	pos.reserve(Nstep);
	vel.reserve(Nstep);
	bias.reserve(Nstep);
	posGain.reserve(Nstep);
	velGain.reserve(Nstep);
	biasGain.reserve(Nstep);
	posErrVar.reserve(Nstep);
	velErrVar.reserve(Nstep);
	biasErrVar.reserve(Nstep);

	for (size_t i = 0; i < Nstep; ++i)
	{
		const bool retval = filter.propagate(i + 1);  // 1-based time step. 0-th time step is initial
		
		const gsl_vector *x = filter.getState();
		const gsl_matrix *K = filter.getKalmanGain();
		const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

		pos.push_back(gsl_vector_get(x, 0));
		vel.push_back(gsl_vector_get(x, 1));
		bias.push_back(gsl_vector_get(x, 2));
		posGain.push_back(gsl_matrix_get(K, 0, 0));
		velGain.push_back(gsl_matrix_get(K, 1, 0));
		biasGain.push_back(gsl_matrix_get(K, 2, 0));
		posErrVar.push_back(gsl_matrix_get(P, 0, 0));
		velErrVar.push_back(gsl_matrix_get(P, 1, 1));
		biasErrVar.push_back(gsl_matrix_get(P, 2, 2));
	}
}

void linear_mass_spring_damper_system_kalman_filter()
{
	const size_t stateDim = 2;
	const size_t inputDim = 1;
	const size_t outputDim = 1;

	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_vector_set_zero(x0);
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_zero(P0);
	gsl_matrix_set(P0, 0, 0, 2.0);  gsl_matrix_set(P0, 1, 1, 2.0);

	LinearMassStringDamperSystemKalmanFilter filter(x0, P0, stateDim, inputDim, outputDim);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	const size_t Nstep = 100;
	std::vector<double> pos, vel;
	std::vector<double> posGain, velGain;
	std::vector<double> posErrVar, velErrVar;
	std::vector<double> corrCoeff;
	pos.reserve(Nstep);
	vel.reserve(Nstep);
	posGain.reserve(Nstep);
	velGain.reserve(Nstep);
	posErrVar.reserve(Nstep);
	velErrVar.reserve(Nstep);
	corrCoeff.reserve(Nstep);

	for (size_t i = 0; i < Nstep; ++i)
	{
		const bool retval = filter.propagate(i + 1);  // 1-based time step. 0-th time step is initial
		
		const gsl_vector *x = filter.getState();
		const gsl_matrix *K = filter.getKalmanGain();
		const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

		pos.push_back(gsl_vector_get(x, 0));
		vel.push_back(gsl_vector_get(x, 1));
		posGain.push_back(gsl_matrix_get(K, 0, 0));
		velGain.push_back(gsl_matrix_get(K, 1, 0));
		posErrVar.push_back(gsl_matrix_get(P, 0, 0));
		velErrVar.push_back(gsl_matrix_get(P, 1, 1));

		corrCoeff.push_back(gsl_matrix_get(P, 0, 1) / (std::sqrt(posErrVar[i]) * std::sqrt(velErrVar[i])));
	}
}

void radar_tracking_system_kalman_filter()
{
	const size_t stateDim = 6;
	const size_t inputDim = 2;
	const size_t outputDim = 2;

	const double &T = RadarTrackingSystemKalmanFilter::T;
	const double &var_r = RadarTrackingSystemKalmanFilter::var_r;
	const double &var_theta = RadarTrackingSystemKalmanFilter::var_theta;
	const double &var_1 = RadarTrackingSystemKalmanFilter::var_1;
	const double &var_2 = RadarTrackingSystemKalmanFilter::var_2;

	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_vector_set_zero(x0);
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_zero(P0);
	gsl_matrix_set(P0, 0, 0, var_r);  gsl_matrix_set(P0, 0, 1, var_r / T);
	gsl_matrix_set(P0, 1, 0, var_r / T);  gsl_matrix_set(P0, 1, 1, 2.0 * var_r / (T*T) + var_1);
	gsl_matrix_set(P0, 2, 2, var_1);
	gsl_matrix_set(P0, 3, 3, var_theta);  gsl_matrix_set(P0, 3, 4, var_theta / T);
	gsl_matrix_set(P0, 4, 3, var_theta / T);  gsl_matrix_set(P0, 4, 4, 2.0 * var_theta / (T*T) + var_2);
	gsl_matrix_set(P0, 5, 5, var_2);

	RadarTrackingSystemKalmanFilter filter(x0, P0, stateDim, inputDim, outputDim);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	const size_t Nstep = 150;
	std::vector<double> range, rangeRate, rangeRateNoise, bearing, bearingRate, bearingRateNoise;
	std::vector<double> rangeGain, rangeRateGain, rangeRateNoiseGain, bearingGain, bearingRateGain, bearingRateNoiseGain;
	std::vector<double> rangeErrVar, rangeRateErrVar, rangeRateNoiseErrVar, bearingErrVar, bearingRateErrVar, bearingRateNoiseErrVar;
	range.reserve(Nstep);
	rangeRate.reserve(Nstep);
	rangeRateNoise.reserve(Nstep);
	bearing.reserve(Nstep);
	bearingRate.reserve(Nstep);
	bearingRateNoise.reserve(Nstep);
	rangeGain.reserve(Nstep);
	rangeRateGain.reserve(Nstep);
	rangeRateNoiseGain.reserve(Nstep);
	bearingGain.reserve(Nstep);
	bearingRateGain.reserve(Nstep);
	bearingRateNoiseGain.reserve(Nstep);
	rangeErrVar.reserve(Nstep);
	rangeRateErrVar.reserve(Nstep);
	rangeRateNoiseErrVar.reserve(Nstep);
	bearingErrVar.reserve(Nstep);
	bearingRateErrVar.reserve(Nstep);
	bearingRateNoiseErrVar.reserve(Nstep);

	for (size_t i = 0; i < Nstep; ++i)
	{
		const bool retval = filter.propagate(i + 1);  // 1-based time step. 0-th time step is initial
		
		const gsl_vector *x = filter.getState();
		const gsl_matrix *K = filter.getKalmanGain();
		const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

		range.push_back(gsl_vector_get(x, 0));
		rangeRate.push_back(gsl_vector_get(x, 1));
		rangeRateNoise.push_back(gsl_vector_get(x, 2));
		bearing.push_back(gsl_vector_get(x, 3));
		bearingRate.push_back(gsl_vector_get(x, 4));
		bearingRateNoise.push_back(gsl_vector_get(x, 5));

		rangeGain.push_back(std::sqrt(std::pow(gsl_matrix_get(K, 0, 0),2)+std::pow(gsl_matrix_get(K, 0, 1),2)));
		rangeRateGain.push_back(std::sqrt(std::pow(gsl_matrix_get(K, 1, 0),2)+std::pow(gsl_matrix_get(K, 1, 1),2)));
		rangeRateNoiseGain.push_back(std::sqrt(std::pow(gsl_matrix_get(K, 2, 0),2)+std::pow(gsl_matrix_get(K, 2, 1),2)));
		bearingGain.push_back(std::sqrt(std::pow(gsl_matrix_get(K, 3, 0),2)+std::pow(gsl_matrix_get(K, 3, 1),2)));
		bearingRateGain.push_back(std::sqrt(std::pow(gsl_matrix_get(K, 4, 0),2)+std::pow(gsl_matrix_get(K, 4, 1),2)));
		bearingRateNoiseGain.push_back(std::sqrt(std::pow(gsl_matrix_get(K, 5, 0),2)+std::pow(gsl_matrix_get(K, 5, 1),2)));

		rangeErrVar.push_back(gsl_matrix_get(P, 0, 0));
		rangeRateErrVar.push_back(gsl_matrix_get(P, 1, 1));
		rangeRateNoiseErrVar.push_back(gsl_matrix_get(P, 2, 2));
		bearingErrVar.push_back(gsl_matrix_get(P, 3, 3));
		bearingRateErrVar.push_back(gsl_matrix_get(P, 4, 4));
		bearingRateNoiseErrVar.push_back(gsl_matrix_get(P, 5, 5));
	}
}

}  // unnamed namespace

void kalman_filter()
{
	simple_system_kalman_filter();
	aided_INS_kalman_filter();
	linear_mass_spring_damper_system_kalman_filter();
	radar_tracking_system_kalman_filter();
}
