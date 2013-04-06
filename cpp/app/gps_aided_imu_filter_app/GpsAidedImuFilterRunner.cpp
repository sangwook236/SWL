#include "stdafx.h"
#include "swl/Config.h"
#include "GpsAidedImuFilterRunner.h"
#include "Adis16350Interface.h"
#include "GpsInterface.h"
#include "swl/rnd_util/UnscentedKalmanFilterWithAdditiveNoise.h"
#include "swl/rnd_util/DiscreteNonlinearStochasticSystem.h"
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


namespace swl {

GpsAidedImuFilterRunner::GpsAidedImuFilterRunner(const double Ts, const ImuData::Accel &initialGravity, const ImuData::Gyro &initialAngularVel)
: dim_(3), Ts_(Ts),
  filter_(), system_(),
  initialGravity_(initialGravity), initialAngularVel_(initialAngularVel),
  actualMeasurement_(),
  //currAccel_(NULL), currAngularVel_(NULL),
  currPos_(NULL), prevPos_(NULL), currVel_(NULL), prevVel_(NULL), currQuaternion_(NULL), prevQuaternion_(NULL),
  step_(0)
{
	currPos_ = gsl_vector_alloc(dim_);
	prevPos_ = gsl_vector_alloc(dim_);
	currVel_ = gsl_vector_alloc(dim_);
	prevVel_ = gsl_vector_alloc(dim_);
	currQuaternion_ = gsl_vector_alloc(4);
	prevQuaternion_ = gsl_vector_alloc(4);

	gsl_vector_set_zero(currPos_);  // initially stationary
	gsl_vector_set_zero(prevPos_);  // initially stationary
	gsl_vector_set_zero(currVel_);  // initially stationary
	gsl_vector_set_zero(prevVel_);  // initially stationary
	gsl_vector_set_zero(currQuaternion_);  // initially stationary
	gsl_vector_set_zero(prevQuaternion_);  // initially stationary

	//currAccel_ = gsl_vector_alloc(dim_);
	//currAngularVel_ = gsl_vector_alloc(dim_);
}

GpsAidedImuFilterRunner::~GpsAidedImuFilterRunner()
{
	gsl_vector_free(currPos_);  currPos_ = NULL;
	gsl_vector_free(prevPos_);  prevPos_ = NULL;
	gsl_vector_free(currVel_);  currVel_ = NULL;
	gsl_vector_free(prevVel_);  prevVel_ = NULL;
	gsl_vector_free(currQuaternion_);  currQuaternion_ = NULL;
	gsl_vector_free(prevQuaternion_);  prevQuaternion_ = NULL;

	//gsl_vector_free(currAccel_);  currAccel_ = NULL;
	//gsl_vector_free(currAngularVel_);  currAngularVel_ = NULL;
}

void GpsAidedImuFilterRunner::initialize(const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim, const gsl_vector *x0, gsl_matrix *P0)
{
	step_ = 0;

	// TODO [check] >>
#if 0
	// for unscented Kalman filter
	const size_t L = stateDim + processNoiseDim + observationNoiseDim;
#else
	// for unscented Kalman filter with additive noise
	const size_t L = stateDim;
#endif
	const double alpha = 1.0e-1;
	const double beta = 2.0;  // for Gaussian distribution
	const double kappa = 3.0 - L; //0.0;

	// create a GPS-aided IMU system
	system_.reset(new GpsAidedImuSystem(Ts_, stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim, initialGravity_, initialAngularVel_));
	if (!system_)
		throw std::runtime_error("fail to create a GPS-aided IMU system");

	// create a GPS-aided IMU filter
	filter_.reset(new UnscentedKalmanFilterWithAdditiveNoise(*system_, alpha, beta, kappa, x0, P0));
	if (!filter_)
		throw std::runtime_error("fail to create a GPS-aided IMU filter");

	actualMeasurement_ = gsl_vector_alloc(outputDim);
}

void GpsAidedImuFilterRunner::finalize()
{
	gsl_vector_free(actualMeasurement_);  actualMeasurement_ = NULL;
}

bool GpsAidedImuFilterRunner::runStep(const gsl_matrix *Q, const gsl_matrix *R, const ImuData::Accel &measuredAccel, const ImuData::Gyro &measuredAngularVel, const EarthData::ECEF &measuredGpsECEF, const EarthData::ECEF &measuredGpsVel, const EarthData::Speed &measuredGpsSpeed)
{
	{
		// method #1
		// 1-based time step. 0-th time step is initial

		// 1. unscented transformation
		filter_->performUnscentedTransformation();

		// compensate the local gravity & the Earth's angular velocity
		ImuData::Accel accel(0.0, 0.0, 0.0);
		ImuData::Gyro angularVel(0.0, 0.0, 0.0);
		{
			const gsl_vector *x_hat = filter_->getEstimatedState();
			const double &E0 = gsl_vector_get(x_hat, 6);
			const double &E1 = gsl_vector_get(x_hat, 7);
			const double &E2 = gsl_vector_get(x_hat, 8);
			const double &E3 = gsl_vector_get(x_hat, 9);
		
			const double &g_ix = initialGravity_.x;
			const double &g_iy = initialGravity_.y;
			const double &g_iz = initialGravity_.z;
			const double &wc_ix = initialAngularVel_.x;
			const double &wc_iy = initialAngularVel_.y;
			const double &wc_iz = initialAngularVel_.z;

			const double g_p = 2.0 * ((0.5 - E2*E2 - E3*E3)*g_ix + (E1*E2 + E0*E3)*g_iy + (E1*E3 - E0*E2)*g_iz);
			const double g_q = 2.0 * ((E1*E2 - E0*E3)*g_ix + (0.5 - E1*E1 - E3*E3)*g_iy + (E2*E3 + E0*E1)*g_iz);
			const double g_r = 2.0 * ((E1*E3 + E0*E2)*g_ix + (E2*E3 - E0*E1)*g_iy + (0.5 - E1*E1 - E2*E2)*g_iz);
			const double wc_p = 2.0 * ((0.5 - E2*E2 - E3*E3)*wc_ix + (E1*E2 + E0*E3)*wc_iy + (E1*E3 - E0*E2)*wc_iz);
			const double wc_q = 2.0 * ((E1*E2 - E0*E3)*wc_ix + (0.5 - E1*E1 - E3*E3)*wc_iy + (E2*E3 + E0*E1)*wc_iz);
			const double wc_r = 2.0 * ((E1*E3 + E0*E2)*wc_ix + (E2*E3 - E0*E1)*wc_iy + (0.5 - E1*E1 - E2*E2)*wc_iz);

			// FIXME [check] >>
			accel.x = measuredAccel.x - g_p;
			accel.y = measuredAccel.y - g_q;
			accel.z = measuredAccel.z - g_r;
			angularVel.x = measuredAngularVel.x - wc_p;
			angularVel.y = measuredAngularVel.y - wc_q;
			angularVel.z = measuredAngularVel.z - wc_r;

			system_->setImuMeasurement(accel, angularVel);
		}

		// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
		if (!filter_->updateTime(step_, NULL, Q)) return false;

		// save x-(k+1) & P-(k+1)
		// FIXME [comment] >>
		{
			const gsl_vector *x_hat = filter_->getEstimatedState();
			const gsl_matrix *P = filter_->getStateErrorCovarianceMatrix();

			const double &Px = gsl_vector_get(x_hat, 0);
			const double &Py = gsl_vector_get(x_hat, 1);
			const double &Pz = gsl_vector_get(x_hat, 2);
			const double &Vx = gsl_vector_get(x_hat, 3);
			const double &Vy = gsl_vector_get(x_hat, 4);
			const double &Vz = gsl_vector_get(x_hat, 5);
			const double &E0 = gsl_vector_get(x_hat, 6);
			const double &E1 = gsl_vector_get(x_hat, 7);
			const double &E2 = gsl_vector_get(x_hat, 8);
			const double &E3 = gsl_vector_get(x_hat, 9);
			//gsl_matrix_get(P, 6, 6);
			//gsl_matrix_get(P, 7, 7);
			//gsl_matrix_get(P, 8, 8);
		}

		// advance time step
		++step_;

		//
		gsl_vector_set(actualMeasurement_, 0, measuredGpsECEF.x);
		gsl_vector_set(actualMeasurement_, 1, measuredGpsECEF.y);
		gsl_vector_set(actualMeasurement_, 2, measuredGpsECEF.z);
		gsl_vector_set(actualMeasurement_, 3, measuredGpsVel.x);
		gsl_vector_set(actualMeasurement_, 4, measuredGpsVel.y);
		gsl_vector_set(actualMeasurement_, 5, measuredGpsVel.z);
		//gsl_vector_set(actualMeasurement_, 6, measuredGpsSpeed.val);

		// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		if (!filter_->updateMeasurement(step_, actualMeasurement_, NULL, R)) return false;

		// save K(k), x(k) & P(k)
		{
			const gsl_vector *x_hat = filter_->getEstimatedState();
			const gsl_matrix *K = filter_->getKalmanGain();
			const gsl_matrix *P = filter_->getStateErrorCovarianceMatrix();

#if 0
			// FIXME [modify] >> wrt body frame, but not initial frame
			const double &E0 = gsl_vector_get(x_hat, 6);
			const double &E1 = gsl_vector_get(x_hat, 7);
			const double &E2 = gsl_vector_get(x_hat, 8);
			const double &E3 = gsl_vector_get(x_hat, 9);
			//gsl_matrix_get(K, 6, 6);
			//gsl_matrix_get(K, 7, 7);
			//gsl_matrix_get(K, 8, 8);
			//gsl_matrix_get(P, 6, 6);
			//gsl_matrix_get(P, 7, 7);
			//gsl_matrix_get(P, 8, 8);
#else
			const double &Px = gsl_vector_get(x_hat, 0);
			const double &Py = gsl_vector_get(x_hat, 1);
			const double &Pz = gsl_vector_get(x_hat, 2);
			const double &Vx = gsl_vector_get(x_hat, 3);
			const double &Vy = gsl_vector_get(x_hat, 4);
			const double &Vz = gsl_vector_get(x_hat, 5);
			const double &E0 = gsl_vector_get(x_hat, 6);
			const double &E1 = gsl_vector_get(x_hat, 7);
			const double &E2 = gsl_vector_get(x_hat, 8);
			const double &E3 = gsl_vector_get(x_hat, 9);
			//const double &Abp = gsl_vector_get(x_hat, 10);
			//const double &Abq = gsl_vector_get(x_hat, 11);
			//const double &Abr = gsl_vector_get(x_hat, 12);
			//const double &Wbp = gsl_vector_get(x_hat, 13);
			//const double &Wbq = gsl_vector_get(x_hat, 14);
			//const double &Wbr = gsl_vector_get(x_hat, 15);
#endif

			gsl_vector_set(currPos_, 0, Py);
			gsl_vector_set(currPos_, 1, Py);
			gsl_vector_set(currPos_, 2, Pz);
			gsl_vector_set(currVel_, 0, Vx);
			gsl_vector_set(currVel_, 1, Vy);
			gsl_vector_set(currVel_, 2, Vz);
			gsl_vector_set(currQuaternion_, 0, E0);
			gsl_vector_set(currQuaternion_, 1, E1);
			gsl_vector_set(currQuaternion_, 2, E2);
			gsl_vector_set(currQuaternion_, 3, E3);

			gsl_vector_memcpy(prevPos_, currPos_);
			gsl_vector_memcpy(prevVel_, currVel_);
			gsl_vector_memcpy(prevQuaternion_, currQuaternion_);
		}
	}

	return true;
}

}  // namespace swl
