#include "stdafx.h"
#include "swl/Config.h"
#include "ImuUnscentedKalmanFilterRunner.h"
#include "AdisUsbz.h"
#include "swl/rnd_util/UnscentedKalmanFilterWithAdditiveNoise.h"
#include "swl/rnd_util/DiscreteNonlinearStochasticSystem.h"
#include <gsl/gsl_blas.h>
#include <iostream>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

class ImuSystem: public swl::DiscreteNonlinearStochasticSystem
{
public:
	typedef swl::DiscreteNonlinearStochasticSystem base_type;

public:
	ImuSystem(const double Ts, const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim, const gsl_vector *initial_gravity)
	: base_type(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim),
	  Ts_(Ts), f_eval_(NULL), h_eval_(NULL), initial_gravity_(NULL)
	{
		//
		f_eval_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(f_eval_);

		//
		h_eval_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(h_eval_);

		// initial gravity
		initial_gravity_ = gsl_vector_alloc(initial_gravity->size);
		gsl_vector_memcpy(initial_gravity_, initial_gravity);
	}
	~ImuSystem()
	{
		gsl_vector_free(f_eval_);  f_eval_ = NULL;
		gsl_vector_free(h_eval_);  h_eval_ = NULL;

		gsl_vector_free(initial_gravity_);  initial_gravity_ = NULL;
	}

private:
	ImuSystem(const ImuSystem &rhs);
	ImuSystem & operator=(const ImuSystem &rhs);

public:
	// the stochastic differential equation: f = f(k, x(k), u(k), v(k))
	/*virtual*/ gsl_vector * evaluatePlantEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		// update of true state w/o noise
		const double &Px = gsl_vector_get(state, 0);
		const double &Py = gsl_vector_get(state, 1);
		const double &Pz = gsl_vector_get(state, 2);
		const double &Vx = gsl_vector_get(state, 3);
		const double &Vy = gsl_vector_get(state, 4);
		const double &Vz = gsl_vector_get(state, 5);
		const double &Ap = gsl_vector_get(state, 6);
		const double &Aq = gsl_vector_get(state, 7);
		const double &Ar = gsl_vector_get(state, 8);
		const double &E0 = gsl_vector_get(state, 9);
		const double &E1 = gsl_vector_get(state, 10);
		const double &E2 = gsl_vector_get(state, 11);
		const double &E3 = gsl_vector_get(state, 12);
		const double &Wp = gsl_vector_get(state, 13);
		const double &Wq = gsl_vector_get(state, 14);
		const double &Wr = gsl_vector_get(state, 15);
		const double &Abp = gsl_vector_get(state, 16);
		const double &Abq = gsl_vector_get(state, 17);
		const double &Abr = gsl_vector_get(state, 18);
		const double &Wbp = gsl_vector_get(state, 19);
		const double &Wbq = gsl_vector_get(state, 20);
		const double &Wbr = gsl_vector_get(state, 21);

		const double &w0 = noise ? gsl_vector_get(noise, 0) : 0.0;
		const double &w1 = noise ? gsl_vector_get(noise, 1) : 0.0;
		const double &w2 = noise ? gsl_vector_get(noise, 2) : 0.0;
		const double &w3 = noise ? gsl_vector_get(noise, 3) : 0.0;
		const double &w4 = noise ? gsl_vector_get(noise, 4) : 0.0;
		const double &w5 = noise ? gsl_vector_get(noise, 5) : 0.0;
		const double &w6 = noise ? gsl_vector_get(noise, 6) : 0.0;
		const double &w7 = noise ? gsl_vector_get(noise, 7) : 0.0;
		const double &w8 = noise ? gsl_vector_get(noise, 8) : 0.0;
		const double &w9 = noise ? gsl_vector_get(noise, 9) : 0.0;
		const double &w10 = noise ? gsl_vector_get(noise, 10) : 0.0;
		const double &w11 = noise ? gsl_vector_get(noise, 11) : 0.0;
		const double &w12 = noise ? gsl_vector_get(noise, 12) : 0.0;
		const double &w13 = noise ? gsl_vector_get(noise, 13) : 0.0;
		const double &w14 = noise ? gsl_vector_get(noise, 14) : 0.0;
		const double &w15 = noise ? gsl_vector_get(noise, 15) : 0.0;
		const double &w16 = noise ? gsl_vector_get(noise, 16) : 0.0;
		const double &w17 = noise ? gsl_vector_get(noise, 17) : 0.0;
		const double &w18 = noise ? gsl_vector_get(noise, 18) : 0.0;
		const double &w19 = noise ? gsl_vector_get(noise, 19) : 0.0;
		const double &w20 = noise ? gsl_vector_get(noise, 20) : 0.0;
		const double &w21 = noise ? gsl_vector_get(noise, 21) : 0.0;

		const double &g_ix = gsl_vector_get(initial_gravity_, 0);
		const double &g_iy = gsl_vector_get(initial_gravity_, 1);
		const double &g_iz = gsl_vector_get(initial_gravity_, 2);

		const double dvdt_x = 2.0 * ((0.5 - E2*E2 - E3*E3)*Ap + (E1*E2 - E0*E3)*Aq + (E1*E3 + E0*E2)*Ar) + g_ix;
		const double dvdt_y = 2.0 * ((E1*E2 + E0*E3)*Ap + (0.5 - E1*E1 - E3*E3)*Aq + (E2*E3 - E0*E1)*Ar) + g_iy;
		const double dvdt_z = 2.0 * ((E1*E3 - E0*E2)*Ap + (E2*E3 + E0*E1)*Aq + (0.5 - E1*E1 - E2*E2)*Ar) + g_iz;

		const double dPhi = Wp * Ts_;
		const double dTheta = Wq * Ts_;
		const double dPsi = Wr * Ts_;
		const double s = 0.5 * std::sqrt(dPhi*dPhi + dTheta*dTheta + dPsi*dPsi);
		const double lambda = 1.0 - std::sqrt(E0*E0 + E1*E1 + E2*E2 + E3*E3);
		// TODO [check] >>
		const double eta_dt = 0.9;  // eta * dt < 1.0

		const double eps = 1.0e-10;
		const double coeff1 = std::cos(s) + eta_dt * lambda;
		const double coeff2 = std::fabs(s) <= eps ? 0.0 : 0.5 * std::sin(s) / s;

		const double f0 = Px + Vx * Ts_;
		const double f1 = Py + Vy * Ts_;
		const double f2 = Pz + Vz * Ts_;
		const double f3 = Vx + dvdt_x * Ts_;
		const double f4 = Vy + dvdt_y * Ts_;
		const double f5 = Vz + dvdt_z * Ts_;
		const double f6 = Ap + w6 * Ts_;
		const double f7 = Aq + w7 * Ts_;
		const double f8 = Ar + w8 * Ts_;
		const double f9 = coeff1 * E0 - coeff2 * (dPhi*E1 + dTheta*E2 + dPsi*E3);
		const double f10 = coeff1 * E1 - coeff2 * (-dPhi*E0 - dPsi*E2 + dTheta*E3);
		const double f11 = coeff1 * E2 - coeff2 * (-dTheta*E0 + dPsi*E1 - dPhi*E3);
		const double f12 = coeff1 * E3 - coeff2 * (-dPsi*E0 - dTheta*E1 + dPhi*E2);
		const double f13 = Wp + w13 * Ts_;
		const double f14 = Wq + w14 * Ts_;
		const double f15 = Wr + w15 * Ts_;
		const double f16 = Abp + w16 * Ts_;
		const double f17 = Abq + w17 * Ts_;
		const double f18 = Abr + w18 * Ts_;
		const double f19 = Wbp + w19 * Ts_;
		const double f20 = Wbq + w20 * Ts_;
		const double f21 = Wbr + w21 * Ts_;

		gsl_vector_set(f_eval_, 0, f0);
		gsl_vector_set(f_eval_, 1, f1);
		gsl_vector_set(f_eval_, 2, f2);
		gsl_vector_set(f_eval_, 3, f3);
		gsl_vector_set(f_eval_, 4, f4);
		gsl_vector_set(f_eval_, 5, f5);
		gsl_vector_set(f_eval_, 6, f6);
		gsl_vector_set(f_eval_, 7, f7);
		gsl_vector_set(f_eval_, 8, f8);
		gsl_vector_set(f_eval_, 9, f9);
		gsl_vector_set(f_eval_, 10, f10);
		gsl_vector_set(f_eval_, 11, f11);
		gsl_vector_set(f_eval_, 12, f12);
		gsl_vector_set(f_eval_, 13, f13);
		gsl_vector_set(f_eval_, 14, f14);
		gsl_vector_set(f_eval_, 15, f15);
		gsl_vector_set(f_eval_, 16, f16);
		gsl_vector_set(f_eval_, 17, f17);
		gsl_vector_set(f_eval_, 18, f18);
		gsl_vector_set(f_eval_, 19, f19);
		gsl_vector_set(f_eval_, 20, f20);
		gsl_vector_set(f_eval_, 21, f21);

		return f_eval_;
	}
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const  // Phi(k) = exp(A(k) * Ts) where A(k) = df(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const  // Bu(k) = Bd(k) * u(k)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const  {  return W_;  }  // W(k) = df(k, x(k), u(k), 0)/dw

	// the observation equation: h = h(k, x(k), u(k), v(k))
	/*virtual*/ gsl_vector * evaluateMeasurementEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const 
	{
		const double &Px = gsl_vector_get(state, 0);
		const double &Py = gsl_vector_get(state, 1);
		const double &Pz = gsl_vector_get(state, 2);
		const double &Vx = gsl_vector_get(state, 3);
		const double &Vy = gsl_vector_get(state, 4);
		const double &Vz = gsl_vector_get(state, 5);
		const double &Ap = gsl_vector_get(state, 6);
		const double &Aq = gsl_vector_get(state, 7);
		const double &Ar = gsl_vector_get(state, 8);
		const double &E0 = gsl_vector_get(state, 9);
		const double &E1 = gsl_vector_get(state, 10);
		const double &E2 = gsl_vector_get(state, 11);
		const double &E3 = gsl_vector_get(state, 12);
		const double &Wp = gsl_vector_get(state, 13);
		const double &Wq = gsl_vector_get(state, 14);
		const double &Wr = gsl_vector_get(state, 15);
		const double &Abp = gsl_vector_get(state, 16);
		const double &Abq = gsl_vector_get(state, 17);
		const double &Abr = gsl_vector_get(state, 18);
		const double &Wbp = gsl_vector_get(state, 19);
		const double &Wbq = gsl_vector_get(state, 20);
		const double &Wbr = gsl_vector_get(state, 21);

		const double &v0 = noise ? gsl_vector_get(noise, 0) : 0.0;
		const double &v1 = noise ? gsl_vector_get(noise, 1) : 0.0;
		const double &v2 = noise ? gsl_vector_get(noise, 2) : 0.0;
		const double &v3 = noise ? gsl_vector_get(noise, 3) : 0.0;
		const double &v4 = noise ? gsl_vector_get(noise, 4) : 0.0;
		const double &v5 = noise ? gsl_vector_get(noise, 5) : 0.0;

		gsl_vector_set(h_eval_, 0, Ap + Abp + v0);
		gsl_vector_set(h_eval_, 1, Aq + Abq + v1);
		gsl_vector_set(h_eval_, 2, Ar + Abr + v2);
		gsl_vector_set(h_eval_, 3, Wp + Wbp + v3);
		gsl_vector_set(h_eval_, 4, Wq + Wbq + v4);
		gsl_vector_set(h_eval_, 5, Wr + Wbr + v5);

		return h_eval_;
	}
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const  // Cd(k) = dh(k, x(k), u(k), 0)/dx
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const  // Du(k) = D(k) * u(k) (D == Dd)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const  {  return V_;  }  // V(k) = dh(k, x(k), u(k), 0)/dv

	// noise covariance matrices
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  // Qd = W * Q * W^T, but not Q
	{  throw std::runtime_error("this function doesn't have to be called");  }
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  // Rd = V * R * V^T, but not R
	{  throw std::runtime_error("this function doesn't have to be called");  }

private:
	const double Ts_;

	// evalution of the plant equation: f = f(k, x(k), u(k), 0)
	gsl_vector *f_eval_;
	// evalution of the measurement equation: h = h(k, x(k), u(k), 0)
	gsl_vector *h_eval_;

	// initial gravity
	gsl_vector *initial_gravity_;
};

}  // unnamed namespace

// "Sigma-Point Kalman Filters for Integrated Navigation", R. van der Merwe and Eric A. Wan,
//	Annual Meeting of The Institute of Navigation, 2004
// "A new multi-position calibration method for MEMS inertial navigation systems", Z. F. Syed, P. Aggarwal, C. Goodall, X. Niu, and N. El-Sheimy,
//	Measurement Science and Technology, vol. 18, pp. 1897-1907, 2007

void imu_unscented_Kalman_filter_with_calibration()
{
	const size_t stateDim = 22;
	const size_t inputDim = 3;
	const size_t outputDim = 6;
	const size_t processNoiseDim = stateDim;
	const size_t observationNoiseDim = outputDim;

	const size_t L = stateDim + processNoiseDim + observationNoiseDim;
	const double alpha = 1.0e-3;
	const double beta = 2.0;  // for Gaussian distribution
	const double kappa = 0.0;  // 3.0 - L;
	const double x_init = 0.1;
	const double sigma_init = std::sqrt(2.0);

	//
#if defined(__USE_RECEIVED_DATA_FROM_ADISUSBZ)
	// sampling interval
	const double Ts = 0.01;  // [sec]

	const size_t Ninitial = 10000;

	// test ADISUSBZ
	//ImuUnscentedKalmanFilterRunner::testAdisUsbz(Ntest);
#else
	std::vector<ImuUnscentedKalmanFilterRunner::Acceleration> accels;
	std::vector<ImuUnscentedKalmanFilterRunner::Gyro> gyros;

	// load validation data
	//const size_t Nsample = 10000;
	//const double Ts = 29.46875 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100801\\03_x_pos.csv", Nsample, accels, gyros);  // 10000 sample, 29.46875 sec
	//const size_t Nsample = 10000;
	//const double Ts = 30.03125 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100801\\04_x_neg.csv", Nsample, accels, gyros);  // 10000 sample, 30.03125 sec
	//const size_t Nsample = 10000;
	//const double Ts = 31.07813 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100801\\05_y_pos.csv", Nsample, accels, gyros);  // 10000 sample, 31.07813 sec
	//const size_t Nsample = 10000;
	//const double Ts = 29.28125 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100801\\06_y_neg.csv", Nsample, accels, gyros);  // 10000 sample, 29.28125 sec
	//const size_t Nsample = 10000;
	//const double Ts = 30.29688 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100801\\01_z_pos.csv", Nsample, accels, gyros);  // 10000 sample, 30.29688 sec
	//const size_t Nsample = 10000;
	//const double Ts = 29.04688 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100801\\02_z_neg.csv", Nsample, accels, gyros);  // 10000 sample, 29.04688 sec

	const size_t Nsample = 300;
	const double Ts = 12.89111 / Nsample;
	ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\x_pos_50cm_40msec_1.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 12.89111 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 12.82764 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\x_pos_50cm_40msec_2.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 12.82764 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 12.70313 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\x_pos_50cm_40msec_3.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 12.70313 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 12.78076 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\x_pos_50cm_40msec_4.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 12.78076 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 12.875 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\x_pos_50cm_40msec_5.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 12.875 sec, 50 cm
	//const size_t Nsample = 250;
	//const double Ts = 10.88989 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\y_neg_50cm_40msec_1.csv", Nsample, accels, gyros);  // 250 sample, 40 msec, 10.88989 sec, 50 cm
	//const size_t Nsample = 250;
	//const double Ts = 10.86011 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\y_neg_50cm_40msec_2.csv", Nsample, accels, gyros);  // 250 sample, 40 msec, 10.86011 sec, 50 cm
	//const size_t Nsample = 250;
	//const double Ts = 10.86011 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\y_neg_50cm_40msec_3.csv", Nsample, accels, gyros);  // 250 sample, 40 msec, 10.86011 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 13.04712 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\y_neg_50cm_40msec_4.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 13.04712 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 12.96802 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\y_neg_50cm_40msec_5.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 12.96802 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 12.93701 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\z_pos_50cm_40msec_1.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 12.93701 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 12.96875 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\z_pos_50cm_40msec_2.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 12.96875 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 13.01611 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\z_pos_50cm_40msec_3.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 13.01611 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 13.01514 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\z_pos_50cm_40msec_4.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 13.01514 sec, 50 cm
	//const size_t Nsample = 300;
	//const double Ts = 13.03076 / Nsample;
	//ImuUnscentedKalmanFilterRunner::loadSavedImuData("..\\data\\adis16350_data_20100813\\z_pos_50cm_40msec_5.csv", Nsample, accels, gyros);  // 300 sample, 40 msec, 13.03076 sec, 50 cm

	const size_t Ninitial = Nsample;
#endif

	//
#if defined(__USE_RECEIVED_DATA_FROM_ADISUSBZ)
	AdisUsbz adis;

#if defined(UNICODE) || defined(_UNICODE)
	if (!adis.Initialize(L"\\\\.\\Ezusb-0"))
#else
	if (!adis.Initialize("\\\\.\\Ezusb-0"))
#endif
	{
		std::cout << "fail to initialize ADISUSBZ" << std::endl;
		return;
	}

	ImuUnscentedKalmanFilterRunner runner(Ts, stateDim, inputDim, outputDim, &adis);
#else
	ImuUnscentedKalmanFilterRunner runner(Ts, stateDim, inputDim, outputDim, NULL);
#endif

	// load calibration parameters
	std::cout << "load calibration parameters ..." << std::endl;
	const std::string calibration_filename("..\\data\\adis16350_data_20100801\\imu_calibration_result.txt");
	runner.loadCalibrationParam(calibration_filename);

	// set an initial gravity
	std::cout << "set an initial gravity ..." << std::endl;
#if defined(__USE_RECEIVED_DATA_FROM_ADISUSBZ)
	runner.initializeGravity(Ninitial);
#else
	runner.initializeGravity(Ninitial, accels, gyros);
#endif

	const gsl_vector *initialGravity = runner.getInitialGravity();

	//
	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_vector_set_zero(x0);
	gsl_vector_set(x0, 6, -gsl_vector_get(initialGravity, 0));  // a_p = g_initial_x
	gsl_vector_set(x0, 7, -gsl_vector_get(initialGravity, 1));  // a_q = g_initial_y
	gsl_vector_set(x0, 8, -gsl_vector_get(initialGravity, 2));  // a_r = g_initial_z
	gsl_vector_set(x0, 9, 1.0);  // e0 = 1.0
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(P0);
	gsl_matrix_scale(P0, 1.0e-8);  // the initial estimate is completely unknown

	const ImuSystem system(Ts, stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim, initialGravity);
	swl::UnscentedKalmanFilterWithAdditiveNoise filter(system, alpha, beta, kappa, x0, P0);

	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;

	//
	gsl_vector *w = gsl_vector_alloc(processNoiseDim);
	gsl_vector_set_zero(w);
	gsl_vector *v = gsl_vector_alloc(observationNoiseDim);
	gsl_vector_set_zero(v);
	gsl_matrix *Q = gsl_matrix_alloc(processNoiseDim, processNoiseDim);
	gsl_matrix_set_identity(Q);
	gsl_matrix *R = gsl_matrix_alloc(observationNoiseDim, observationNoiseDim);
	gsl_matrix_set_identity(R);

	// FIXME [modify] >>
	const double QQ = 1.0e-10;
	gsl_matrix_set(Q, 0, 0, QQ);
	gsl_matrix_set(Q, 1, 1, QQ);
	gsl_matrix_set(Q, 2, 2, QQ);
	gsl_matrix_set(Q, 3, 3, QQ);
	gsl_matrix_set(Q, 4, 4, QQ);
	gsl_matrix_set(Q, 5, 5, QQ);
	gsl_matrix_set(Q, 6, 6, QQ);
	gsl_matrix_set(Q, 7, 7, QQ);
	gsl_matrix_set(Q, 8, 8, QQ);
	gsl_matrix_set(Q, 9, 9, QQ);
	gsl_matrix_set(Q, 10, 10, QQ);
	gsl_matrix_set(Q, 11, 11, QQ);
	gsl_matrix_set(Q, 12, 12, QQ);
	gsl_matrix_set(Q, 13, 13, QQ);
	gsl_matrix_set(Q, 14, 14, QQ);
	gsl_matrix_set(Q, 15, 15, QQ);
	gsl_matrix_set(Q, 16, 16, QQ);
	gsl_matrix_set(Q, 17, 17, QQ);
	gsl_matrix_set(Q, 18, 18, QQ);
	gsl_matrix_set(Q, 19, 19, QQ);
	gsl_matrix_set(Q, 20, 20, QQ);
	gsl_matrix_set(Q, 21, 21, QQ);

	// FIXME [modify] >>
	const double RR = 1.0e-10;
	gsl_matrix_set(R, 0, 0, RR);
	gsl_matrix_set(R, 1, 1, RR);
	gsl_matrix_set(R, 2, 2, RR);
	gsl_matrix_set(R, 3, 3, RR);
	gsl_matrix_set(R, 4, 4, RR);
	gsl_matrix_set(R, 5, 5, RR);

	// unscented Kalman filtering
	std::cout << "start unscented Kalman filtering ..." << std::endl;

#if defined(__USE_RECEIVED_DATA_FROM_ADISUSBZ)
	const size_t Nstep = 10000;
#else
	const size_t Nstep = Ninitial;
#endif

	gsl_vector *measuredAccel = gsl_vector_alloc(3);
	gsl_vector *measuredAngularVel = gsl_vector_alloc(3);

	size_t step = 0;
	while (step < Nstep)
	{
#if defined(__USE_RECEIVED_DATA_FROM_ADISUSBZ)
		runner.readAdisData(measuredAccel, measuredAngularVel);
#else
		gsl_vector_set(measuredAccel, 0, accels[step].x);
		gsl_vector_set(measuredAccel, 1, accels[step].y);
		gsl_vector_set(measuredAccel, 2, accels[step].z);
		gsl_vector_set(measuredAngularVel, 0, gyros[step].x);
		gsl_vector_set(measuredAngularVel, 1, gyros[step].y);
		gsl_vector_set(measuredAngularVel, 2, gyros[step].z);
#endif

		if (!runner.runImuFilter(filter, step, measuredAccel, measuredAngularVel, Q, R))
		{
			std::cout << "IMU filtering error !!!" << std::endl;
			return;
		}

		const gsl_vector *pos = runner.getFilteredPos();
		const gsl_vector *vel = runner.getFilteredVel();
		const gsl_vector *accel = runner.getFilteredAccel();
		const gsl_vector *quat = runner.getFilteredQuaternion();
		const gsl_vector *angVel = runner.getFilteredAngularVel();

		std::cout << (step + 1) << ": " << gsl_vector_get(pos, 0) << ", " << gsl_vector_get(pos, 1) << ", " << gsl_vector_get(pos, 2) << " ; " <<
			gsl_vector_get(quat, 0) << ", " << gsl_vector_get(quat, 1) << ", " << gsl_vector_get(quat, 2) << ", " << gsl_vector_get(quat, 3) << std::endl;

		++step;
	}

	gsl_vector_free(w);  w = NULL;
	gsl_vector_free(v);  v = NULL;
	gsl_matrix_free(Q);  Q = NULL;
	gsl_matrix_free(R);  R = NULL;
	gsl_vector_free(measuredAccel);  measuredAccel = NULL;
	gsl_vector_free(measuredAngularVel);  measuredAngularVel = NULL;
}
