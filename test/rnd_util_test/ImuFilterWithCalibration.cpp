#include "stdafx.h"
#include "swl/Config.h"
#include "AdisUsbz.h"
#include "swl/rnd_util/DiscreteNonlinearStochasticSystem.h"
#include "swl/rnd_util/ExtendedKalmanFilter.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <boost/math/constants/constants.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#define __USE_ADIS16350_DATA 1

#if defined(__USE_ADIS16350_DATA)
const char ADIS16350_SUPPLY_OUT =	0x02;
const char ADIS16350_XGYRO_OUT =	0x04;
const char ADIS16350_YGYRO_OUT =	0x06;
const char ADIS16350_ZGYRO_OUT =	0x08;
const char ADIS16350_XACCL_OUT =	0x0A;
const char ADIS16350_YACCL_OUT =	0x0C;
const char ADIS16350_ZACCL_OUT =	0x0E;
const char ADIS16350_XTEMP_OUT =	0x10;
const char ADIS16350_YTEMP_OUT =	0x12;
const char ADIS16350_ZTEMP_OUT =	0x14;
const char ADIS16350_AUX_ADC =		0x16;

const double ADIS16350_SUPLY_SCALE_FACTOR =		1.8315e-1;  // binary, [V]
const double ADIS16350_GYRO_SCALE_FACTOR =		0.07326;  // 2's complement, [deg/sec]
//const double ADIS16350_GYRO_SCALE_FACTOR =	0.07326 * boost::math::constants::pi<double>() / 180.0;  // 2's complement, [rad/sec]
const double ADIS16350_ACCL_SCALE_FACTOR =		2.522e-3;  // 2's complement, [g]
const double ADIS16350_TEMP_SCALE_FACTOR =		0.1453;  // 2's complement, [deg]
const double ADIS16350_ADC_SCALE_FACTOR =		0.6105e-3;  // binary, [V]
#endif


namespace {

void calculate_calibrated_acceleration(const gsl_vector *param, const gsl_vector *lg, gsl_vector *a_calibrated)
{
	const double &b_gx = gsl_vector_get(param, 0);
	const double &b_gy = gsl_vector_get(param, 1);
	const double &b_gz = gsl_vector_get(param, 2);
	const double &s_gx = gsl_vector_get(param, 3);
	const double &s_gy = gsl_vector_get(param, 4);
	const double &s_gz = gsl_vector_get(param, 5);
	const double &theta_gyz = gsl_vector_get(param, 6);
	const double &theta_gzx = gsl_vector_get(param, 7);
	const double &theta_gzy = gsl_vector_get(param, 8);

	const double &l_gx = gsl_vector_get(lg, 0);
	const double &l_gy = gsl_vector_get(lg, 1);
	const double &l_gz = gsl_vector_get(lg, 2);

	const double tan_gyz = std::tan(theta_gyz);
	const double tan_gzx = std::tan(theta_gzx);
	const double tan_gzy = std::tan(theta_gzy);
	const double cos_gyz = std::cos(theta_gyz);
	const double cos_gzx = std::cos(theta_gzx);
	const double cos_gzy = std::cos(theta_gzy);

	const double g_x = (l_gx - b_gx) / (1.0 + s_gx);
	const double g_y = tan_gyz * (l_gx - b_gx) / (1.0 + s_gx) + (l_gy - b_gy) / ((1.0 + s_gy) * cos_gyz);
	const double g_z = (tan_gzx * tan_gyz - tan_gzy / cos_gzx) * (l_gx - b_gx) / (1.0 + s_gx) +
		((l_gy - b_gy) * tan_gzx) / ((1.0 + s_gy) * cos_gyz) + (l_gz - b_gz) / ((1.0 + s_gz) * cos_gzx * cos_gzy);

	gsl_vector_set(a_calibrated, 0, g_x);
	gsl_vector_set(a_calibrated, 1, g_y);
	gsl_vector_set(a_calibrated, 2, g_z);
}

void calculate_calibrated_angular_rate(const gsl_vector *param, const gsl_vector *lw, gsl_vector *w_calibrated)
{
	const double &b_wx = gsl_vector_get(param, 0);
	const double &b_wy = gsl_vector_get(param, 1);
	const double &b_wz = gsl_vector_get(param, 2);

	const double &l_wx = gsl_vector_get(lw, 0);
	const double &l_wy = gsl_vector_get(lw, 1);
	const double &l_wz = gsl_vector_get(lw, 2);

	const double w_x = l_wx - b_wx;
	const double w_y = l_wy - b_wy;
	const double w_z = l_wz - b_wz;

	gsl_vector_set(w_calibrated, 0, w_x);
	gsl_vector_set(w_calibrated, 1, w_y);
	gsl_vector_set(w_calibrated, 2, w_z);
}

void load_calibration_param(const std::string &filename, const size_t numAccelParam, const size_t numGyroParam, gsl_vector *accel_calibration_param, gsl_matrix *accel_calibration_covar, gsl_vector *gyro_calibration_param, gsl_matrix *gyro_calibration_covar)
{
	std::ifstream stream(filename.c_str());
	if (!stream)
	{
		std::cout << "file not found !!!" << std::endl;
		return;
	}

	std::string line_str;
	double val;

	// load acceleration parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	if (!stream.eof())
	{
		std::getline(stream, line_str);

		std::istringstream sstream(line_str);
		for (size_t i = 0; i < numAccelParam; ++i)
		{
			sstream >> val;
			gsl_vector_set(accel_calibration_param, i, val);
		}
	}
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	// load covariance of acceleration parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	if (!stream.eof())
	{
		for (size_t i = 0; i < numAccelParam; ++i)
		{
			std::getline(stream, line_str);

			std::istringstream sstream(line_str);
			for (size_t j = 0; j < numAccelParam; ++j)
			{
				sstream >> val;
				gsl_matrix_set(accel_calibration_covar, i, j, val);
			}
		}
	}
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	// load gyro parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	if (!stream.eof())
	{
		std::getline(stream, line_str);

		std::istringstream sstream(line_str);
		for (size_t i = 0; i < numGyroParam; ++i)
		{
			sstream >> val;
			gsl_vector_set(gyro_calibration_param, i, val);
		}
	}
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	// load covariance of gyro parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	if (!stream.eof())
	{
		for (size_t i = 0; i < numGyroParam; ++i)
		{
			std::getline(stream, line_str);

			std::istringstream sstream(line_str);
			for (size_t j = 0; j < numGyroParam; ++j)
			{
				sstream >> val;
				gsl_matrix_set(gyro_calibration_covar, i, j, val);
			}
		}
	}
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return;
	}

	stream.close();
}

class ImuSystem: public swl::DiscreteNonlinearStochasticSystem
{
public:
	typedef swl::DiscreteNonlinearStochasticSystem base_type;

public:
	ImuSystem(const double Ts, const size_t stateDim, const size_t inputDim, const size_t outputDim, const gsl_vector *initial_gravity, const gsl_matrix *Qd, const gsl_matrix *Rd)
	: base_type(stateDim, inputDim, outputDim, (size_t)-1, (size_t)-1),
	  Ts_(Ts), Phi_(NULL), A_(NULL), B_(NULL), Bd_(NULL), Bu_(NULL), Cd_(NULL), Qd_(NULL), Rd_(NULL), f_eval_(NULL), h_eval_(NULL), initial_gravity_(NULL),
	  A_tmp_(NULL)
	{
		// Phi = exp(A * Ts) -> I + A * Ts where A = df/dx
		//	the EKF approximation for Phi is I + A * Ts
		Phi_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set_zero(Phi_);

		A_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_set_zero(A_);

		// B = [ 0 0 0 ; 0 0 0 ; 0 0 0 ; 1 0 0 ; 0 1 0 ; 0 0 1 ; 0 0 0 ; ... ; 0 0 0 ]
		B_ = gsl_matrix_alloc(stateDim_, inputDim_);
		gsl_matrix_set_zero(B_);
		gsl_matrix_set(B_, 3, 0, 1.0);
		gsl_matrix_set(B_, 4, 1, 1.0);
		gsl_matrix_set(B_, 5, 2, 1.0);

		Bd_ = gsl_matrix_alloc(stateDim_, inputDim_);
		Bu_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(Bu_);

		// C = [
		//	0 0 0  0 0 0  1 0 0   0 0 0 0  0 0 0  1 0 0  0 0 0
		//	0 0 0  0 0 0  0 1 0   0 0 0 0  0 0 0  0 1 0  0 0 0
		//	0 0 0  0 0 0  0 0 1   0 0 0 0  0 0 0  0 0 1  0 0 0
		//	0 0 0  0 0 0  0 0 0   0 0 0 0  1 0 0  0 0 0  1 0 0
		//	0 0 0  0 0 0  0 0 0   0 0 0 0  0 1 0  0 0 0  0 1 0
		//	0 0 0  0 0 0  0 0 0   0 0 0 0  0 0 1  0 0 0  0 0 1
		// ]
		Cd_ = gsl_matrix_alloc(outputDim_, stateDim_);
		gsl_matrix_set_zero(Cd_);
		gsl_matrix_set(Cd_, 0, 6, 1.0);  gsl_matrix_set(Cd_, 0, 16, 1.0);
		gsl_matrix_set(Cd_, 1, 7, 1.0);  gsl_matrix_set(Cd_, 1, 17, 1.0);
		gsl_matrix_set(Cd_, 2, 8, 1.0);  gsl_matrix_set(Cd_, 2, 18, 1.0);
		gsl_matrix_set(Cd_, 3, 13, 1.0);  gsl_matrix_set(Cd_, 3, 19, 1.0);
		gsl_matrix_set(Cd_, 4, 14, 1.0);  gsl_matrix_set(Cd_, 4, 20, 1.0);
		gsl_matrix_set(Cd_, 5, 15, 1.0);  gsl_matrix_set(Cd_, 5, 21, 1.0);

		// Qd = W * Q * W^T 
		//	the EKF approximation of Qd will be W * [ Q * Ts ] * W^T
		Qd_ = gsl_matrix_alloc(stateDim_, stateDim_);
		gsl_matrix_memcpy(Qd_, Qd);

		// Rd = V * R * V^T
		Rd_ = gsl_matrix_alloc(outputDim_, outputDim_);
		gsl_matrix_memcpy(Rd_, Rd);

		//
		f_eval_ = gsl_vector_alloc(stateDim_);
		gsl_vector_set_zero(f_eval_);

		//
		h_eval_ = gsl_vector_alloc(outputDim_);
		gsl_vector_set_zero(h_eval_);

		// initial gravity
		initial_gravity_ = gsl_vector_alloc(initial_gravity->size);
		gsl_vector_memcpy(initial_gravity_, initial_gravity);

		//
		A_tmp_ = gsl_matrix_alloc(stateDim_, stateDim_);
	}
	~ImuSystem()
	{
		gsl_matrix_free(Phi_);  Phi_ = NULL;
		gsl_matrix_free(A_);  A_ = NULL;
		gsl_matrix_free(B_);  B_ = NULL;
		gsl_matrix_free(Bd_);  Bd_ = NULL;
		gsl_vector_free(Bu_);  Bu_ = NULL;
		gsl_matrix_free(Cd_);  Cd_ = NULL;
		gsl_matrix_free(Qd_);  Qd_ = NULL;
		gsl_matrix_free(Rd_);  Rd_ = NULL;

		gsl_vector_free(f_eval_);  f_eval_ = NULL;
		gsl_vector_free(h_eval_);  h_eval_ = NULL;

		gsl_vector_free(initial_gravity_);  initial_gravity_ = NULL;

		gsl_matrix_free(A_tmp_);  A_tmp_ = NULL;
	}

private:
	ImuSystem(const ImuSystem &rhs);
	ImuSystem & operator=(const ImuSystem &rhs);

public:
	// the stochastic differential equation: f = f(k, x(k), u(k), v(k))
	/*virtual*/ gsl_vector * evaluatePlantEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
	{
		const gsl_matrix *Phi = getStateTransitionMatrix(step, state);
		const gsl_vector *Bu = getControlInput(step, state);
		gsl_vector_memcpy(f_eval_, Bu);
		gsl_blas_dgemv(CblasNoTrans, 1.0, Phi, state, 1.0, f_eval_);

		return f_eval_;
	}
	/*virtual*/ gsl_matrix * getStateTransitionMatrix(const size_t step, const gsl_vector *state) const  // Phi(k) = exp(A(k) * Ts) where A(k) = df(k, x(k), u(k), 0)/dx
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

		gsl_matrix_set(A_, 0, 3, 1.0);
		gsl_matrix_set(A_, 1, 4, 1.0);
		gsl_matrix_set(A_, 2, 5, 1.0);

		gsl_matrix_set(A_, 3, 6, 2.0 * (0.5 - E2*E2 - E3*E3));
		gsl_matrix_set(A_, 3, 7, 2.0 * (E1*E2 - E0*E3));
		gsl_matrix_set(A_, 3, 8, 2.0 * (E1*E3 + E0*E2));
		gsl_matrix_set(A_, 3, 9, 2.0 * (-E3*Aq + E2*Ar));
		gsl_matrix_set(A_, 3, 10, 2.0 * (E2*Aq + E3*Ar));
		gsl_matrix_set(A_, 3, 11, 2.0 * (-2.0*E2*Ap + E1*Aq + E0*Ar));
		gsl_matrix_set(A_, 3, 12, 2.0 * (-2.0*E3*Ap - E0*Aq + E1*Ar));

		gsl_matrix_set(A_, 4, 6, 2.0 * (E1*E2 + E0*E3));
		gsl_matrix_set(A_, 4, 7, 2.0 * (0.5 - E1*E1 - E3*E3));
		gsl_matrix_set(A_, 4, 8, 2.0 * (E2*E3 - E0*E1));
		gsl_matrix_set(A_, 4, 9, 2.0 * (E3*Ap - E1*Ar));
		gsl_matrix_set(A_, 4, 10, 2.0 * (E2*Ap - 2.0*E1*Aq - E0*Ar));
		gsl_matrix_set(A_, 4, 11, 2.0 * (E1*Ap + E3*Ar));
		gsl_matrix_set(A_, 4, 12, 2.0 * (E0*Ap - 2.0*E3*Aq + E2*Ar));

		gsl_matrix_set(A_, 5, 6, 2.0 * (E1*E3 - E0*E2));
		gsl_matrix_set(A_, 5, 7, 2.0 * (E2*E3 + E0*E1));
		gsl_matrix_set(A_, 5, 8, 2.0 * (0.5 - E1*E1 - E2*E2));
		gsl_matrix_set(A_, 5, 9, 2.0 * (-E2*Ap + E1*Aq));
		gsl_matrix_set(A_, 5, 10, 2.0 * (E3*Ap + E0*Aq - 2.0*E1*Ar));
		gsl_matrix_set(A_, 5, 11, 2.0 * (-E0*Ap + E3*Aq - 2.0*E2*Ar));
		gsl_matrix_set(A_, 5, 12, 2.0 * (E1*Ap + E2*Aq));

		gsl_matrix_set(A_, 9, 9, 0.0);
		gsl_matrix_set(A_, 9, 10, -0.5 * Wp);
		gsl_matrix_set(A_, 9, 11, -0.5 * Wq);
		gsl_matrix_set(A_, 9, 12, -0.5 * Wr);
		gsl_matrix_set(A_, 9, 13, -0.5 * E1);
		gsl_matrix_set(A_, 9, 14, -0.5 * E2);
		gsl_matrix_set(A_, 9, 15, -0.5 * E3);

		gsl_matrix_set(A_, 10, 9, -0.5 * -Wp);
		gsl_matrix_set(A_, 10, 10, 0.0);
		gsl_matrix_set(A_, 10, 11, -0.5 * -Wr);
		gsl_matrix_set(A_, 10, 12, -0.5 * Wq);
		gsl_matrix_set(A_, 10, 13, -0.5 * -E0);
		gsl_matrix_set(A_, 10, 14, -0.5 * E3);
		gsl_matrix_set(A_, 10, 15, -0.5 * -E2);

		gsl_matrix_set(A_, 11, 9, -0.5 * -Wq);
		gsl_matrix_set(A_, 11, 10, -0.5 * Wr);
		gsl_matrix_set(A_, 11, 11, 0.0);
		gsl_matrix_set(A_, 11, 12, -0.5 * -Wp);
		gsl_matrix_set(A_, 11, 13, -0.5 * -E3);
		gsl_matrix_set(A_, 11, 14, -0.5 * -E0);
		gsl_matrix_set(A_, 11, 15, -0.5 * E1);

		gsl_matrix_set(A_, 12, 9, -0.5 * -Wr);
		gsl_matrix_set(A_, 12, 10, -0.5 * -Wq);
		gsl_matrix_set(A_, 12, 11, -0.5 * Wp);
		gsl_matrix_set(A_, 12, 12, 0.0);
		gsl_matrix_set(A_, 12, 13, -0.5 * E2);
		gsl_matrix_set(A_, 12, 14, -0.5 * -E1);
		gsl_matrix_set(A_, 12, 15, -0.5 * -E0);

		// Phi = exp(A * Ts) -> I + A * Ts where A = df/dx
		//	the EKF approximation for Phi is I + A * Ts

		gsl_matrix_memcpy(Phi_, A_);
		gsl_matrix_scale(Phi_, Ts_);
		for (size_t i = 0; i < stateDim_; ++i)
			gsl_matrix_set(Phi_, i, i, gsl_matrix_get(Phi_, i, i) + 1.0);

		return Phi_;
	}
	/*virtual*/ gsl_vector * getControlInput(const size_t step, const gsl_vector *state) const  // Bu(k) = Bd(k) * u(k)
	{
		// Bu = Bd * u where u(t) = initial gravity
		//	Bd = integrate(exp(A * t), {t, 0, Ts}) * B or A^-1 * (Ad - I) * B if A is nonsingular
		//	Ad = Phi = exp(A * Ts) -> I + A * Ts where A = df/dx

		// TODO [check] >>
		//	integrate(exp(A * t), {t, 0, Ts}) -> integrate(I + A * t, {t, 0, Ts}) -> I * Ts + A * Ts^2 / 2
		//	Bd = integrate(exp(A * t), {t, 0, Ts}) * B -> integrate(I + A * t, {t, 0, Ts}) * B -> (I * Ts + A * Ts^2 / 2) * B ???
		gsl_matrix_memcpy(A_tmp_, A_);
		gsl_matrix_scale(A_tmp_, 0.5 * Ts_*Ts_);
		for (size_t i = 0; i < stateDim_; ++i)
			gsl_matrix_set(A_tmp_, i, i, gsl_matrix_get(A_tmp_, i, i) + Ts_);

		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A_tmp_, B_, 0.0, Bd_);
		gsl_blas_dgemv(CblasNoTrans, 1.0, Bd_, initial_gravity_, 0.0, Bu_);

		return Bu_;
	}
	///*virtual*/ gsl_matrix * getProcessNoiseCouplingMatrix(const size_t step) const  {  return W_;  }  // W(k) = df(k, x(k), u(k), 0)/dw

	// the observation equation: h = h(k, x(k), u(k), v(k))
	/*virtual*/ gsl_vector * evaluateMeasurementEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const 
	{
		gsl_blas_dgemv(CblasNoTrans, 1.0, Cd_, state, 0.0, h_eval_);
		return h_eval_;
	}
	/*virtual*/ gsl_matrix * getOutputMatrix(const size_t step, const gsl_vector *state) const  // Cd(k) = dh(k, x(k), u(k), 0)/dx
	{  return Cd_;  }
	/*virtual*/ gsl_vector * getMeasurementInput(const size_t step, const gsl_vector *state) const  // Du(k) = D(k) * u(k) (D == Dd)
	{  throw std::runtime_error("this function doesn't have to be called");  }
	///*virtual*/ gsl_matrix * getMeasurementNoiseCouplingMatrix(const size_t step) const  {  return V_;  }  // V(k) = dh(k, x(k), u(k), 0)/dv

	// noise covariance matrices
	/*virtual*/ gsl_matrix * getProcessNoiseCovarianceMatrix(const size_t step) const  {  return Qd_;  }  // Qd = W * Q * W^T, but not Q
	/*virtual*/ gsl_matrix * getMeasurementNoiseCovarianceMatrix(const size_t step) const  {  return Rd_;  }  // Rd = V * R * V^T, but not R

private:
	const double Ts_;

	gsl_matrix *Phi_;
	gsl_matrix *A_;  // A = df/dx
	gsl_matrix *B_;  // B = df/du
	gsl_matrix *Bd_;  // Bd = integrate(exp(A * t), {t, 0, Ts}) * B or A^-1 * (Ad - I) * B if A is nonsingular
	gsl_vector *Bu_;  // Bu = Bd * u
	gsl_matrix *Cd_;  // Cd = C
	gsl_matrix *Qd_;
	gsl_matrix *Rd_;

	// evalution of the plant equation: f = f(k, x(k), u(k), 0)
	gsl_vector *f_eval_;
	// evalution of the measurement equation: h = h(k, x(k), u(k), 0)
	gsl_vector *h_eval_;

	// initial gravity
	gsl_vector *initial_gravity_;

	gsl_matrix *A_tmp_;
};

struct Acceleration
{
	Acceleration(const double &_x, const double &_y, const double &_z)
	: x(_x), y(_y), z(_z)
	{}
	Acceleration(const Acceleration &rhs)
	: x(rhs.x), y(rhs.y), z(rhs.z)
	{}

	double x, y, z;
};

struct Gyro
{
	Gyro(const double &_x, const double &_y, const double &_z)
	: x(_x), y(_y), z(_z)
	{}
	Gyro(const Gyro &rhs)
	: x(rhs.x), y(rhs.y), z(rhs.z)
	{}

	double x, y, z;
};

void load_imu_data(const std::string &filename, const double ref_gravity, std::vector<Acceleration> &accels, std::vector<Gyro> &gyros)
{
	std::ifstream stream(filename.c_str());

	// data format:
	//	Sample #,Time (sec),XgND,X Gryo,YgND,Y Gyro,ZgND,Z Gyro,XaND,X acc,YaND,Y acc,ZaND,Z acc,

	if (!stream.is_open())
	{
		std::cout << "file open error !!!" << std::endl;
		return;
	}

	// eliminate the 1st 7 lines
	{
		std::string str;
		for (int i = 0; i < 7; ++i)
		{
			if (!stream.eof())
				std::getline(stream, str);
			else
			{
				std::cout << "file format error !!!" << std::endl;
				return;
			}
		}
	}

	//
	double xAccel, yAccel, zAccel, xGyro, yGyro, zGyro;

	const double deg2rad = boost::math::constants::pi<double>() / 180.0;
	int dummy;
	double dummy1;
	char comma;
	while (!stream.eof())
	{
		stream >> dummy >> comma >> dummy1 >> comma >>
			dummy >> comma >> xGyro >> comma >>
			dummy >> comma >> yGyro >> comma >>
			dummy >> comma >> zGyro >> comma >>
			dummy >> comma >> xAccel >> comma >>
			dummy >> comma >> yAccel >> comma >>
			dummy >> comma >> zAccel >> comma;
		if (stream)
		{
			accels.push_back(Acceleration(xAccel * ref_gravity, yAccel * ref_gravity, zAccel * ref_gravity));  // [m/sec^2]
			gyros.push_back(Gyro(xGyro * deg2rad, yGyro * deg2rad, zGyro * deg2rad));  // [rad/sec]
		}
	}

	if (accels.empty() || gyros.empty())
	{
		std::cout << "data error !!!" << std::endl;
		return;
	}

	stream.close();
}

void read_adis16350(AdisUsbz &adis, const double ref_gravity, Acceleration &accel, Gyro &gyro)
{
	const short rawXGyro = adis.ReadInt14(ADIS16350_XGYRO_OUT) & 0x3FFF;
	const short rawYGyro = adis.ReadInt14(ADIS16350_YGYRO_OUT) & 0x3FFF;
	const short rawZGyro = adis.ReadInt14(ADIS16350_ZGYRO_OUT) & 0x3FFF;
	const short rawXAccel = adis.ReadInt14(ADIS16350_XACCL_OUT) & 0x3FFF;
	const short rawYAccel = adis.ReadInt14(ADIS16350_YACCL_OUT) & 0x3FFF;
	const short rawZAccel = adis.ReadInt14(ADIS16350_ZACCL_OUT) & 0x3FFF;

	accel.x = ((rawXAccel & 0x2000) == 0x2000 ? (rawXAccel - 0x4000) : rawXAccel) * ADIS16350_ACCL_SCALE_FACTOR;
	accel.y = ((rawYAccel & 0x2000) == 0x2000 ? (rawYAccel - 0x4000) : rawYAccel) * ADIS16350_ACCL_SCALE_FACTOR;
	accel.z = ((rawZAccel & 0x2000) == 0x2000 ? (rawZAccel - 0x4000) : rawZAccel) * ADIS16350_ACCL_SCALE_FACTOR;

	gyro.x = ((rawXGyro & 0x2000) == 0x2000 ? (rawXGyro - 0x4000) : rawXGyro) * ADIS16350_GYRO_SCALE_FACTOR;
	gyro.y = ((rawYGyro & 0x2000) == 0x2000 ? (rawYGyro - 0x4000) : rawYGyro) * ADIS16350_GYRO_SCALE_FACTOR;
	gyro.z = ((rawZGyro & 0x2000) == 0x2000 ? (rawZGyro - 0x4000) : rawZGyro) * ADIS16350_GYRO_SCALE_FACTOR;
}

void adis_usbz_test(AdisUsbz &adis, const double ref_gravity)
{
	Acceleration accel(0.0, 0.0, 0.0);
	Gyro gyro(0.0, 0.0, 0.0);
	while (true)
	{
		read_adis16350(adis, ref_gravity, accel, gyro);

		std::cout << accel.x << ", " << accel.y << ", " << accel.z << " ; " << gyro.x << ", " << gyro.y << ", " << gyro.z << std::endl;
	}
}

}  // unnamed namespace

// "Sigma-Point Kalman Filters for Integrated Navigation", R. van der Merwe and Eric A. Wan,
//	Annual Meeting of The Institute of Navigation, 2004
// "A new multi-position calibration method for MEMS inertial navigation systems", Z. F. Syed, P. Aggarwal, C. Goodall, X. Niu, and N. El-Sheimy,
//	Measurement Science and Technology, vol. 18, pp. 1897-1907, 2007
void imu_filter_with_calibration()
{
	// [ref] wikipedia
	// (latitude, longitude, altitude) = (phi, lambda, h) = (36.36800, 127.35532, ?)
	// g(phi, h) = 9.780327 * (1 + 0.0053024 * sin(phi)^2 - 0.0000058 * sin(2 * phi)^2) - 3.086 * 10^-6 * h
	const double deg2rad = boost::math::constants::pi<double>() / 180.0;
	const double phi = 36.36800 * deg2rad;  // latitude [rad]
	const double lambda = 127.35532 * deg2rad;  // longitude [rad]
	const double h = 0.0;  // altitude: unknown [m]
	const double sin_phi = std::sin(phi);
	const double sin_2phi = std::sin(2 * phi);
	const double g_true = 9.780327 * (1 + 0.0053024 * sin_phi*sin_phi - 0.0000058 * sin_2phi*sin_2phi) - 3.086e-6 * h;  // [m/sec^2]

	// [ref] "The Global Positioning System and Inertial Navigation", Jay Farrell & Mattthew Barth, pp. 22
	const double w_true = 7.292115e-5;  // [rad/sec]

#if 1
	const size_t numAccelParam = 9;
#else
	const size_t numAccelParam = 6;
#endif
	const size_t numGyroParam = 3;
	const size_t numMeasurement = 14;

	const size_t axisDim = 3;

	const size_t stateDim = 22;
	const size_t inputDim = 3;
	const size_t outputDim = 6;

	//const double Ts = 0.01;  // [sec]
	const double Ts = 0.016 / 5;  // [sec]

	//
	gsl_vector *accel_calibration_param = gsl_vector_alloc(numAccelParam);
	gsl_matrix *accel_calibration_covar = gsl_matrix_alloc(numAccelParam, numAccelParam);
	gsl_vector *gyro_calibration_param = gsl_vector_alloc(numGyroParam);
	gsl_matrix *gyro_calibration_covar = gsl_matrix_alloc(numGyroParam, numGyroParam);

	gsl_vector *accel_measurement = gsl_vector_alloc(axisDim);
	gsl_vector *gyro_measurement = gsl_vector_alloc(axisDim);
	gsl_vector *accel_measurement_calibrated = gsl_vector_alloc(axisDim);
	gsl_vector *gyro_measurement_calibrated = gsl_vector_alloc(axisDim);

	// load calibration parameters
	std::cout << "load calibration parameters ..." << std::endl;
	const std::string calibration_filename("..\\data\\adis16350_data_20100801\\imu_calibration_result.txt");
	load_calibration_param(calibration_filename, numAccelParam, numGyroParam, accel_calibration_param, accel_calibration_covar, gyro_calibration_param, gyro_calibration_covar);

	gsl_matrix_free(accel_calibration_covar);
	gsl_matrix_free(gyro_calibration_covar);

#if defined(__USE_ADIS16350_DATA)
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

	// test ADISUSBZ
	//adis_usbz_test(adis, g_true);

		Acceleration accel_measured(0.0, 0.0, 0.0);
		Gyro gyro_measured(0.0, 0.0, 0.0);
#else
	// load validation data
	const size_t Ninitial = 10000;
	const size_t Nstep = Ninitial;

	std::vector<Acceleration> accels;
	std::vector<Gyro> gyros;
	accels.reserve(Ninitial);
	gyros.reserve(Ninitial);
	//load_imu_data("..\\data\\adis16350_data_20100801\\01_z_pos.csv", g_true, accels, gyros);
	load_imu_data("..\\data\\adis16350_data_20100801\\02_z_neg.csv", g_true, accels, gyros);
#endif

	// set an initial gravity
	std::cout << "set an initial gravity ..." << std::endl;
	gsl_vector *g_initial = gsl_vector_alloc(inputDim);
	//gsl_vector *w_initial = gsl_vector_alloc(inputDim);

	{
#if defined(__USE_ADIS16350_DATA)
		const size_t Ninitial = 10000;
#endif

		double accel_x_sum = 0.0, accel_y_sum = 0.0, accel_z_sum = 0.0;
		double gyro_x_sum = 0.0, gyro_y_sum = 0.0, gyro_z_sum = 0.0;
		for (size_t i = 0; i < Ninitial; ++i)
		{
#if defined(__USE_ADIS16350_DATA)
			read_adis16350(adis, g_true, accel_measured, gyro_measured);

			gsl_vector_set(accel_measurement, 0, accel_measured.x);
			gsl_vector_set(accel_measurement, 1, accel_measured.y);
			gsl_vector_set(accel_measurement, 2, accel_measured.z);
			gsl_vector_set(gyro_measurement, 0, gyro_measured.x);
			gsl_vector_set(gyro_measurement, 1, gyro_measured.y);
			gsl_vector_set(gyro_measurement, 2, gyro_measured.z);
#else
			gsl_vector_set(accel_measurement, 0, accels[i].x);
			gsl_vector_set(accel_measurement, 1, accels[i].y);
			gsl_vector_set(accel_measurement, 2, accels[i].z);
			gsl_vector_set(gyro_measurement, 0, accels[i].x);
			gsl_vector_set(gyro_measurement, 1, accels[i].y);
			gsl_vector_set(gyro_measurement, 2, accels[i].z);
#endif

			calculate_calibrated_acceleration(accel_calibration_param, accel_measurement, accel_measurement_calibrated);
			//calculate_calibrated_angular_rate(gyro_calibration_param, gyro_measurement, gyro_measurement_calibrated);

			accel_x_sum += gsl_vector_get(accel_measurement_calibrated, 0);
			accel_y_sum += gsl_vector_get(accel_measurement_calibrated, 1);
			accel_z_sum += gsl_vector_get(accel_measurement_calibrated, 2);
			//gyro_x_sum += gsl_vector_get(gyro_measurement_calibrated, 0);
			//gyro_y_sum += gsl_vector_get(gyro_measurement_calibrated, 1);
			//gyro_z_sum += gsl_vector_get(gyro_measurement_calibrated, 2);
		}

		gsl_vector_set(g_initial, 0, accel_x_sum / Ninitial);
		gsl_vector_set(g_initial, 1, accel_y_sum / Ninitial);
		gsl_vector_set(g_initial, 2, accel_z_sum / Ninitial);
		//gsl_vector_set(w_initial, 0, gyro_x_sum / Ninitial);
		//gsl_vector_set(w_initial, 1, gyro_y_sum / Ninitial);
		//gsl_vector_set(w_initial, 2, gyro_z_sum / Ninitial);
	}

	// extended Kalman filtering
	gsl_vector *x0 = gsl_vector_alloc(stateDim);
	gsl_vector_set_zero(x0);
	gsl_vector_set(x0, 6, -gsl_vector_get(g_initial, 0));  // a_p = g_initial_x
	gsl_vector_set(x0, 7, -gsl_vector_get(g_initial, 1));  // a_q = g_initial_y
	gsl_vector_set(x0, 8, -gsl_vector_get(g_initial, 2));  // a_r = g_initial_z
	gsl_vector_set(x0, 9, 1.0);  // e0 = 1.0
	gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_identity(P0);
	gsl_matrix_scale(P0, 1.0e-8);  // the initial estimate is completely unknown

	// Qd = W * Q * W^T where W = I
	//	the EKF approximation of Qd will be W * [ Q * Ts ] * W^T
	gsl_matrix *Qd = gsl_matrix_alloc(stateDim, stateDim);
	gsl_matrix_set_zero(Qd);
	// FIXME [modify] >>
	const double QQ = 1.0e-8;
	gsl_matrix_set(Qd, 0, 0, QQ);
	gsl_matrix_set(Qd, 1, 1, QQ);
	gsl_matrix_set(Qd, 2, 2, QQ);
	gsl_matrix_set(Qd, 3, 3, QQ);
	gsl_matrix_set(Qd, 4, 4, QQ);
	gsl_matrix_set(Qd, 5, 5, QQ);
	gsl_matrix_set(Qd, 6, 6, QQ);
	gsl_matrix_set(Qd, 7, 7, QQ);
	gsl_matrix_set(Qd, 8, 8, QQ);
	gsl_matrix_set(Qd, 9, 9, QQ);
	gsl_matrix_set(Qd, 10, 10, QQ);
	gsl_matrix_set(Qd, 11, 11, QQ);
	gsl_matrix_set(Qd, 12, 12, QQ);
	gsl_matrix_set(Qd, 13, 13, QQ);
	gsl_matrix_set(Qd, 14, 14, QQ);
	gsl_matrix_set(Qd, 15, 15, QQ);
	gsl_matrix_set(Qd, 16, 16, QQ);
	gsl_matrix_set(Qd, 17, 17, QQ);
	gsl_matrix_set(Qd, 18, 18, QQ);
	gsl_matrix_set(Qd, 19, 19, QQ);
	gsl_matrix_set(Qd, 20, 20, QQ);
	gsl_matrix_set(Qd, 21, 21, QQ);
	gsl_matrix_scale(Qd, Ts);

	// Rd = V * R * V^T where V = I
	gsl_matrix *Rd = gsl_matrix_alloc(outputDim, outputDim);
	gsl_matrix_set_zero(Rd);
	// FIXME [modify] >>
	const double RR = 1.0e-8;
	gsl_matrix_set(Rd, 0, 0, RR);
	gsl_matrix_set(Rd, 1, 1, RR);
	gsl_matrix_set(Rd, 2, 2, RR);
	gsl_matrix_set(Rd, 3, 3, RR);
	gsl_matrix_set(Rd, 4, 4, RR);
	gsl_matrix_set(Rd, 5, 5, RR);

	const ImuSystem system(Ts, stateDim, inputDim, outputDim, g_initial, Qd, Rd);
	swl::DiscreteExtendedKalmanFilter filter(system, x0, P0);

	gsl_vector_free(g_initial);  g_initial = NULL;
	gsl_vector_free(x0);  x0 = NULL;
	gsl_matrix_free(P0);  P0 = NULL;
	gsl_matrix_free(Qd);  Qd = NULL;
	gsl_matrix_free(Rd);  Rd = NULL;

	std::cout << "start extended Kalman filtering ..." << std::endl;

	gsl_vector *pos_curr = gsl_vector_alloc(axisDim);
	gsl_vector *pos_prev = gsl_vector_alloc(axisDim);
	gsl_vector *vel_curr = gsl_vector_alloc(axisDim);
	gsl_vector *vel_prev = gsl_vector_alloc(axisDim);
	gsl_vector *ang_curr = gsl_vector_alloc(axisDim);
	gsl_vector *ang_prev = gsl_vector_alloc(axisDim);
	gsl_vector_set_zero(pos_curr);  // initially stationary
	gsl_vector_set_zero(pos_prev);  // initially stationary
	gsl_vector_set_zero(vel_curr);  // initially stationary
	gsl_vector_set_zero(vel_prev);  // initially stationary
	gsl_vector_set_zero(ang_curr);  // initially stationary
	gsl_vector_set_zero(ang_prev);  // initially stationary

	// method #1
	// 1-based time step. 0-th time step is initial
	gsl_vector *actualMeasurement = gsl_vector_alloc(outputDim);
#if defined(__USE_ADIS16350_DATA)
	const size_t Nstep = 10000;
#endif
	size_t step = 0;
	while (step < Nstep)
	{
		// 0. initial estimates: x(0) & P(0)

		// 1. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		const bool retval1 = filter.updateTime(step, NULL);
		assert(retval1);

		// save x-(k+1) & P-(k+1)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			const double Ax = gsl_vector_get(x_hat, 6);
			const double Ay = gsl_vector_get(x_hat, 7);
			const double Az = gsl_vector_get(x_hat, 8);
			const double Wx = gsl_vector_get(x_hat, 13);
			const double Wy = gsl_vector_get(x_hat, 14);
			const double Wz = gsl_vector_get(x_hat, 15);
			//gsl_matrix_get(P, 6, 6);
			//gsl_matrix_get(P, 7, 7);
			//gsl_matrix_get(P, 8, 8);
		}

		//
#if defined(__USE_ADIS16350_DATA)
		read_adis16350(adis, g_true, accel_measured, gyro_measured);

		gsl_vector_set(accel_measurement, 0, accel_measured.x);
		gsl_vector_set(accel_measurement, 1, accel_measured.y);
		gsl_vector_set(accel_measurement, 2, accel_measured.z);
		gsl_vector_set(gyro_measurement, 0, gyro_measured.x);
		gsl_vector_set(gyro_measurement, 1, gyro_measured.y);
		gsl_vector_set(gyro_measurement, 2, gyro_measured.z);
#else
		gsl_vector_set(accel_measurement, 0, accels[step].x);
		gsl_vector_set(accel_measurement, 1, accels[step].y);
		gsl_vector_set(accel_measurement, 2, accels[step].z);
		gsl_vector_set(gyro_measurement, 0, gyros[step].x);
		gsl_vector_set(gyro_measurement, 1, gyros[step].y);
		gsl_vector_set(gyro_measurement, 2, gyros[step].z);
#endif

		calculate_calibrated_acceleration(accel_calibration_param, accel_measurement, accel_measurement_calibrated);
		calculate_calibrated_angular_rate(gyro_calibration_param, gyro_measurement, gyro_measurement_calibrated);

		gsl_vector_set(actualMeasurement, 0, gsl_vector_get(accel_measurement_calibrated, 0));
		gsl_vector_set(actualMeasurement, 1, gsl_vector_get(accel_measurement_calibrated, 1));
		gsl_vector_set(actualMeasurement, 2, gsl_vector_get(accel_measurement_calibrated, 2));
		gsl_vector_set(actualMeasurement, 3, gsl_vector_get(gyro_measurement_calibrated, 0));
		gsl_vector_set(actualMeasurement, 4, gsl_vector_get(gyro_measurement_calibrated, 1));
		gsl_vector_set(actualMeasurement, 5, gsl_vector_get(gyro_measurement_calibrated, 2));

		// advance time step
		++step;

		// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		const bool retval2 = filter.updateMeasurement(step, actualMeasurement, NULL);
		assert(retval2);

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter.getEstimatedState();
			const gsl_matrix *K = filter.getKalmanGain();
			const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

			const double Ax = gsl_vector_get(x_hat, 6);
			const double Ay = gsl_vector_get(x_hat, 7);
			const double Az = gsl_vector_get(x_hat, 8);
			const double Wx = gsl_vector_get(x_hat, 13);
			const double Wy = gsl_vector_get(x_hat, 14);
			const double Wz = gsl_vector_get(x_hat, 15);
			//gsl_matrix_get(K, 6, 6);
			//gsl_matrix_get(K, 7, 7);
			//gsl_matrix_get(K, 8, 8);
			//gsl_matrix_get(P, 6, 6);
			//gsl_matrix_get(P, 7, 7);
			//gsl_matrix_get(P, 8, 8);

			//std::cout << Ax << ", " << Ay << ", " << Az << " ; " << Wx << ", " << Wy << ", " << Wz << std::endl;

			gsl_vector_set(vel_curr, 0, gsl_vector_get(vel_prev, 0) + Ax * Ts);
			gsl_vector_set(pos_curr, 0, gsl_vector_get(pos_prev, 0) + gsl_vector_get(vel_prev, 0) * Ts + 0.5 * Ax * Ts*Ts);
			gsl_vector_set(vel_curr, 1, gsl_vector_get(vel_prev, 1) + Ay * Ts);
			gsl_vector_set(pos_curr, 1, gsl_vector_get(pos_prev, 1) + gsl_vector_get(vel_prev, 1) * Ts + 0.5 * Ay * Ts*Ts);
			gsl_vector_set(vel_curr, 2, gsl_vector_get(vel_prev, 2) + Az * Ts);
			gsl_vector_set(pos_curr, 2, gsl_vector_get(pos_prev, 2) + gsl_vector_get(vel_prev, 2) * Ts + 0.5 * Az * Ts*Ts);
			gsl_vector_set(ang_curr, 0, gsl_vector_get(ang_prev, 0) + Wx * Ts);
			gsl_vector_set(ang_curr, 1, gsl_vector_get(ang_prev, 1) + Wy * Ts);
			gsl_vector_set(ang_curr, 2, gsl_vector_get(ang_prev, 2) + Wz * Ts);

			gsl_vector_memcpy(pos_prev, pos_curr);
			gsl_vector_memcpy(vel_prev, vel_curr);
			gsl_vector_memcpy(ang_prev, ang_curr);

			std::cout << step << " : " << gsl_vector_get(pos_curr, 0) << ", " << gsl_vector_get(pos_curr, 1) << ", " << gsl_vector_get(pos_curr, 2) << " ; " <<
				gsl_vector_get(ang_curr, 0) << ", " << gsl_vector_get(ang_curr, 1) << ", " << gsl_vector_get(ang_curr, 2) << std::endl;
		}
	}

	gsl_vector_free(pos_curr);
	gsl_vector_free(pos_prev);
	gsl_vector_free(vel_curr);
	gsl_vector_free(vel_prev);
	gsl_vector_free(ang_curr);
	gsl_vector_free(ang_prev);

	gsl_vector_free(accel_calibration_param);
	gsl_vector_free(gyro_calibration_param);
	gsl_vector_free(accel_measurement);
	gsl_vector_free(gyro_measurement);
	gsl_vector_free(accel_measurement_calibrated);
	gsl_vector_free(gyro_measurement_calibrated);
	gsl_vector_free(g_initial);
	gsl_vector_free(actualMeasurement);
}
