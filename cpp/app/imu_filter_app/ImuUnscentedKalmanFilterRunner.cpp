//#include "stdafx.h"
#include "swl/Config.h"
#include "ImuUnscentedKalmanFilterRunner.h"
#include "swl/rnd_util/UnscentedKalmanFilterWithAdditiveNoise.h"
#include "swl/rnd_util/DiscreteNonlinearStochasticSystem.h"
#include "adisusbz/AdisUsbz.h"
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


namespace {
namespace local {

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

const double deg2rad = boost::math::constants::pi<double>() / 180.0;
const double lambda = 36.368 * deg2rad;  // latitude [rad]
const double phi = 127.364 * deg2rad;  // longitude [rad]
const double h = 71.0;  // altitude: 71 ~ 82 [m]
const double sin_lambda = std::sin(lambda);
const double sin_2lambda = std::sin(2 * lambda);

}  // namespace local
}  // unnamed namespace

// [ref] wikipedia: Gravity of Earth
// (latitude, longitude, altitude) = (lambda, phi, h) = (36.368, 127.364, 71.0)
// g(lambda, h) = 9.780327 * (1 + 0.0053024 * sin(lambda)^2 - 0.0000058 * sin(2 * lambda)^2) - 3.086 * 10^-6 * h
/*static*/ const double ImuUnscentedKalmanFilterRunner::REF_GRAVITY_ = 9.780327 * (1.0 + 0.0053024 * local::sin_lambda*local::sin_lambda - 0.0000058 * local::sin_2lambda*local::sin_2lambda) - 3.086e-6 * local::h;  // [m/sec^2]

// [ref] "The Global Positioning System and Inertial Navigation", Jay Farrell & Matthew Barth, pp. 22
/*static*/ const double ImuUnscentedKalmanFilterRunner::REF_ANGULAR_VEL_ = 7.292115e-5;  // [rad/sec]

ImuUnscentedKalmanFilterRunner::ImuUnscentedKalmanFilterRunner(const double Ts, const size_t stateDim, const size_t inputDim, const size_t outputDim, AdisUsbz *adis)
: numAccelParam_(9), numGyroParam_(3), dim_(3), Ts_(Ts), adis_(adis),
  initialGravity_(NULL), accelCalibrationParam_(NULL), accelCalibrationCovariance_(NULL), gyroCalibrationParam_(NULL), gyroCalibrationCovariance_(NULL),
  measuredAccel_(NULL), measuredAngularVel_(NULL), calibratedAccel_(NULL), calibratedAngularVel_(NULL), currAccel_(NULL), currAngularVel_(NULL),
  actualMeasurement_(NULL), currPos_(NULL), prevPos_(NULL), currVel_(NULL), prevVel_(NULL), currQuaternion_(NULL), prevQuaternion_(NULL)
{
	initialGravity_ = gsl_vector_alloc(dim_);
	//initialAngularVel_ = gsl_vector_alloc(dim_);

	accelCalibrationParam_ = gsl_vector_alloc(numAccelParam_);
	accelCalibrationCovariance_ = gsl_matrix_alloc(numAccelParam_, numAccelParam_);
	gyroCalibrationParam_ = gsl_vector_alloc(numGyroParam_);
	gyroCalibrationCovariance_ = gsl_matrix_alloc(numGyroParam_, numGyroParam_);

	measuredAccel_ = gsl_vector_alloc(dim_);
	measuredAngularVel_ = gsl_vector_alloc(dim_);
	calibratedAccel_ = gsl_vector_alloc(dim_);
	calibratedAngularVel_ = gsl_vector_alloc(dim_);

	actualMeasurement_ = gsl_vector_alloc(outputDim);

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

	currAccel_ = gsl_vector_alloc(dim_);
	currAngularVel_ = gsl_vector_alloc(dim_);
}

ImuUnscentedKalmanFilterRunner::~ImuUnscentedKalmanFilterRunner()
{
	gsl_vector_free(initialGravity_);  initialGravity_ = NULL;
	//gsl_vector_free(initialAngularVel_);  initialAngularVel_ = NULL;

	gsl_vector_free(accelCalibrationParam_);  accelCalibrationParam_ = NULL;
	gsl_matrix_free(accelCalibrationCovariance_);  accelCalibrationCovariance_ = NULL;
	gsl_vector_free(gyroCalibrationParam_);  gyroCalibrationParam_ = NULL;
	gsl_matrix_free(gyroCalibrationCovariance_);  gyroCalibrationCovariance_ = NULL;

	gsl_vector_free(measuredAccel_);  measuredAccel_ = NULL;
	gsl_vector_free(measuredAngularVel_);  measuredAngularVel_ = NULL;
	gsl_vector_free(calibratedAccel_);  calibratedAccel_ = NULL;
	gsl_vector_free(calibratedAngularVel_);  calibratedAngularVel_ = NULL;

	gsl_vector_free(actualMeasurement_);  actualMeasurement_ = NULL;

	gsl_vector_free(currPos_);  currPos_ = NULL;
	gsl_vector_free(prevPos_);  prevPos_ = NULL;
	gsl_vector_free(currVel_);  currVel_ = NULL;
	gsl_vector_free(prevVel_);  prevVel_ = NULL;
	gsl_vector_free(currQuaternion_);  currQuaternion_ = NULL;
	gsl_vector_free(prevQuaternion_);  prevQuaternion_ = NULL;

	gsl_vector_free(currAccel_);  currAccel_ = NULL;
	gsl_vector_free(currAngularVel_);  currAngularVel_ = NULL;
}

void ImuUnscentedKalmanFilterRunner::calculateCalibratedAcceleration(const gsl_vector *lg, gsl_vector *a_calibrated) const
{
	const double &b_gx = gsl_vector_get(accelCalibrationParam_, 0);
	const double &b_gy = gsl_vector_get(accelCalibrationParam_, 1);
	const double &b_gz = gsl_vector_get(accelCalibrationParam_, 2);
	const double &s_gx = gsl_vector_get(accelCalibrationParam_, 3);
	const double &s_gy = gsl_vector_get(accelCalibrationParam_, 4);
	const double &s_gz = gsl_vector_get(accelCalibrationParam_, 5);
	const double &theta_gyz = gsl_vector_get(accelCalibrationParam_, 6);
	const double &theta_gzx = gsl_vector_get(accelCalibrationParam_, 7);
	const double &theta_gzy = gsl_vector_get(accelCalibrationParam_, 8);

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

void ImuUnscentedKalmanFilterRunner::calculateCalibratedAngularRate(const gsl_vector *lw, gsl_vector *w_calibrated) const
{
	const double &b_wx = gsl_vector_get(gyroCalibrationParam_, 0);
	const double &b_wy = gsl_vector_get(gyroCalibrationParam_, 1);
	const double &b_wz = gsl_vector_get(gyroCalibrationParam_, 2);

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

bool ImuUnscentedKalmanFilterRunner::loadCalibrationParam(const std::string &filename)
{
	std::ifstream stream(filename.c_str());
	if (!stream)
	{
		std::ostringstream stream;
		stream << "file not found at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	std::string line_str;
	double val;

	// load acceleration parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	if (!stream.eof())
	{
		std::getline(stream, line_str);

		std::istringstream sstream(line_str);
		for (size_t i = 0; i < numAccelParam_; ++i)
		{
			sstream >> val;
			gsl_vector_set(accelCalibrationParam_, i, val);
		}
	}
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	// load covariance of acceleration parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	if (!stream.eof())
	{
		for (size_t i = 0; i < numAccelParam_; ++i)
		{
			std::getline(stream, line_str);

			std::istringstream sstream(line_str);
			for (size_t j = 0; j < numAccelParam_; ++j)
			{
				sstream >> val;
				gsl_matrix_set(accelCalibrationCovariance_, i, j, val);
			}
		}
	}
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	// load gyro parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	if (!stream.eof())
	{
		std::getline(stream, line_str);

		std::istringstream sstream(line_str);
		for (size_t i = 0; i < numGyroParam_; ++i)
		{
			sstream >> val;
			gsl_vector_set(gyroCalibrationParam_, i, val);
		}
	}
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
	}

	// load covariance of gyro parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	if (!stream.eof())
	{
		for (size_t i = 0; i < numGyroParam_; ++i)
		{
			std::getline(stream, line_str);

			std::istringstream sstream(line_str);
			for (size_t j = 0; j < numGyroParam_; ++j)
			{
				sstream >> val;
				gsl_matrix_set(gyroCalibrationCovariance_, i, j, val);
			}
		}
	}
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	stream.close();

	return true;
}

bool ImuUnscentedKalmanFilterRunner::readAdisData(gsl_vector *accel, gsl_vector *gyro) const
{
	if (!adis_) return false;

	const short rawXGyro = adis_->ReadInt14(local::ADIS16350_XGYRO_OUT) & 0x3FFF;
	const short rawYGyro = adis_->ReadInt14(local::ADIS16350_YGYRO_OUT) & 0x3FFF;
	const short rawZGyro = adis_->ReadInt14(local::ADIS16350_ZGYRO_OUT) & 0x3FFF;
	const short rawXAccel = adis_->ReadInt14(local::ADIS16350_XACCL_OUT) & 0x3FFF;
	const short rawYAccel = adis_->ReadInt14(local::ADIS16350_YACCL_OUT) & 0x3FFF;
	const short rawZAccel = adis_->ReadInt14(local::ADIS16350_ZACCL_OUT) & 0x3FFF;

	// [m/sec^2]
	gsl_vector_set(accel, 0, ((rawXAccel & 0x2000) == 0x2000 ? (rawXAccel - 0x4000) : rawXAccel) * local::ADIS16350_ACCL_SCALE_FACTOR * REF_GRAVITY_);
	gsl_vector_set(accel, 1, ((rawYAccel & 0x2000) == 0x2000 ? (rawYAccel - 0x4000) : rawYAccel) * local::ADIS16350_ACCL_SCALE_FACTOR * REF_GRAVITY_);
	gsl_vector_set(accel, 2, ((rawZAccel & 0x2000) == 0x2000 ? (rawZAccel - 0x4000) : rawZAccel) * local::ADIS16350_ACCL_SCALE_FACTOR * REF_GRAVITY_);

	// [rad/sec]
	gsl_vector_set(gyro, 0, ((rawXGyro & 0x2000) == 0x2000 ? (rawXGyro - 0x4000) : rawXGyro) * local::ADIS16350_GYRO_SCALE_FACTOR * local::deg2rad);
	gsl_vector_set(gyro, 1, ((rawYGyro & 0x2000) == 0x2000 ? (rawYGyro - 0x4000) : rawYGyro) * local::ADIS16350_GYRO_SCALE_FACTOR * local::deg2rad);
	gsl_vector_set(gyro, 2, ((rawZGyro & 0x2000) == 0x2000 ? (rawZGyro - 0x4000) : rawZGyro) * local::ADIS16350_GYRO_SCALE_FACTOR * local::deg2rad);

	return true;
}

void ImuUnscentedKalmanFilterRunner::initializeGravity(const size_t Ninitial)
{
	double accel_x_sum = 0.0, accel_y_sum = 0.0, accel_z_sum = 0.0;
	//double gyro_x_sum = 0.0, gyro_y_sum = 0.0, gyro_z_sum = 0.0;

	for (size_t i = 0; i < Ninitial; ++i)
	{
#if defined(__USE_RECEIVED_DATA_FROM_ADISUSBZ)
		ImuUnscentedKalmanFilterRunner::readAdisData(measuredAccel_, measuredAngularVel_);
#endif

		calculateCalibratedAcceleration(measuredAccel_, calibratedAccel_);
		//calculateCalibratedAngularRate(measuredAngularVel_, calibratedAngularVel_);

		accel_x_sum += gsl_vector_get(calibratedAccel_, 0);
		accel_y_sum += gsl_vector_get(calibratedAccel_, 1);
		accel_z_sum += gsl_vector_get(calibratedAccel_, 2);
		//gyro_x_sum += gsl_vector_get(calibratedAngularVel_, 0);
		//gyro_y_sum += gsl_vector_get(calibratedAngularVel_, 1);
		//gyro_z_sum += gsl_vector_get(calibratedAngularVel_, 2);
	}

	gsl_vector_set(initialGravity_, 0, accel_x_sum / Ninitial);
	gsl_vector_set(initialGravity_, 1, accel_y_sum / Ninitial);
	gsl_vector_set(initialGravity_, 2, accel_z_sum / Ninitial);
	//gsl_vector_set(initialAngularVel, 0, gyro_x_sum / Ninitial);
	//gsl_vector_set(initialAngularVel, 1, gyro_y_sum / Ninitial);
	//gsl_vector_set(initialAngularVel, 2, gyro_z_sum / Ninitial);
}

void ImuUnscentedKalmanFilterRunner::initializeGravity(const size_t Ninitial, const std::vector<Acceleration> &accels, const std::vector<Gyro> &gyros)
{
	double accel_x_sum = 0.0, accel_y_sum = 0.0, accel_z_sum = 0.0;
	//double gyro_x_sum = 0.0, gyro_y_sum = 0.0, gyro_z_sum = 0.0;

	for (size_t i = 0; i < Ninitial; ++i)
	{
		gsl_vector_set(measuredAccel_, 0, accels[i].x);
		gsl_vector_set(measuredAccel_, 1, accels[i].y);
		gsl_vector_set(measuredAccel_, 2, accels[i].z);
		//gsl_vector_set(measuredAngularVel_, 0, gyros[i].x);
		//gsl_vector_set(measuredAngularVel_, 1, gyros[i].y);
		//gsl_vector_set(measuredAngularVel_, 2, gyros[i].z);

		calculateCalibratedAcceleration(measuredAccel_, calibratedAccel_);
		//calculateCalibratedAngularRate(measuredAngularVel_, calibratedAngularVel_);

		accel_x_sum += gsl_vector_get(calibratedAccel_, 0);
		accel_y_sum += gsl_vector_get(calibratedAccel_, 1);
		accel_z_sum += gsl_vector_get(calibratedAccel_, 2);
		//gyro_x_sum += gsl_vector_get(calibratedAngularVel_, 0);
		//gyro_y_sum += gsl_vector_get(calibratedAngularVel_, 1);
		//gyro_z_sum += gsl_vector_get(calibratedAngularVel_, 2);
	}

	gsl_vector_set(initialGravity_, 0, accel_x_sum / Ninitial);
	gsl_vector_set(initialGravity_, 1, accel_y_sum / Ninitial);
	gsl_vector_set(initialGravity_, 2, accel_z_sum / Ninitial);
	//gsl_vector_set(initialAngularVel, 0, gyro_x_sum / Ninitial);
	//gsl_vector_set(initialAngularVel, 1, gyro_y_sum / Ninitial);
	//gsl_vector_set(initialAngularVel, 2, gyro_z_sum / Ninitial);
}

bool ImuUnscentedKalmanFilterRunner::testAdisUsbz(const size_t loopCount)
{
	size_t loop = 0;
	while (loop++ < loopCount)
	{
		if (!readAdisData(measuredAccel_, measuredAngularVel_))
			return false;

		std::cout << gsl_vector_get(measuredAccel_, 0) << ", " << gsl_vector_get(measuredAccel_, 1) << ", " << gsl_vector_get(measuredAccel_, 2) << " ; " <<
			gsl_vector_get(measuredAngularVel_, 0) << ", " << gsl_vector_get(measuredAngularVel_, 1) << ", " << gsl_vector_get(measuredAngularVel_, 2) << std::endl;
	}

	return true;
}

/*static*/ bool ImuUnscentedKalmanFilterRunner::loadSavedImuData(const std::string &filename, const size_t &Nsample, std::vector<Acceleration> &accels, std::vector<Gyro> &gyros)
{
	std::ifstream stream(filename.c_str());

	// data format:
	//	Sample #,Time (sec),XgND,X Gryo,YgND,Y Gyro,ZgND,Z Gyro,XaND,X acc,YaND,Y acc,ZaND,Z acc,

	if (!stream.is_open())
	{
		std::ostringstream stream;
		stream << "file open error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	accels.reserve(Nsample);
	gyros.reserve(Nsample);

	// eliminate the 1st 7 lines
	{
		std::string str;
		for (int i = 0; i < 7; ++i)
		{
			if (!stream.eof())
				std::getline(stream, str);
			else
			{
				std::ostringstream stream;
				stream << "file format error at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return false;
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
			accels.push_back(Acceleration(xAccel * REF_GRAVITY_, yAccel * REF_GRAVITY_, zAccel * REF_GRAVITY_));  // [m/sec^2]
			gyros.push_back(Gyro(xGyro * deg2rad, yGyro * deg2rad, zGyro * deg2rad));  // [rad/sec]
		}
	}

	if (accels.empty() || gyros.empty())
	{
		std::ostringstream stream;
		stream << "data error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return false;
	}

	stream.close();

	return true;
}

bool ImuUnscentedKalmanFilterRunner::runImuFilter(swl::UnscentedKalmanFilterWithAdditiveNoise &filter, const size_t step, const gsl_vector *measuredAccel, const gsl_vector *measuredAngularVel, const gsl_matrix *Q, const gsl_matrix *R, const gsl_vector *initialGravity)
{
	size_t step2 = step;

	// method #1
	// 1-based time step. 0-th time step is initial

	// 1. unscented transformation
	filter.performUnscentedTransformation();

	// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
	if (!filter.updateTime(step2, NULL, Q)) return false;
/*
	// save x-(k+1) & P-(k+1)
	{
		const gsl_vector *x_hat = filter.getEstimatedState();
		const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

		const double &Ax = gsl_vector_get(x_hat, 6);
		const double &Ay = gsl_vector_get(x_hat, 7);
		const double &Az = gsl_vector_get(x_hat, 8);
		const double &Wx = gsl_vector_get(x_hat, 13);
		const double &Wy = gsl_vector_get(x_hat, 14);
		const double &Wz = gsl_vector_get(x_hat, 15);
		//gsl_matrix_get(P, 6, 6);
		//gsl_matrix_get(P, 7, 7);
		//gsl_matrix_get(P, 8, 8);
	}
*/
	//
	calculateCalibratedAcceleration(measuredAccel, calibratedAccel_);
	calculateCalibratedAngularRate(measuredAngularVel, calibratedAngularVel_);

	// compensate the local gravity & the earth's angular rate
	{
		const gsl_vector *x_hat = filter.getEstimatedState();
		const double &E0 = gsl_vector_get(x_hat, 9);
		const double &E1 = gsl_vector_get(x_hat, 10);
		const double &E2 = gsl_vector_get(x_hat, 11);
		const double &E3 = gsl_vector_get(x_hat, 12);
	
		const double &g_ix = gsl_vector_get(initialGravity, 0);
		const double &g_iy = gsl_vector_get(initialGravity, 1);
		const double &g_iz = gsl_vector_get(initialGravity, 2);

		const double g_p = 2.0 * ((0.5 - E2*E2 - E3*E3)*g_ix + (E1*E2 + E0*E3)*g_iy + (E1*E3 - E0*E2)*g_iz);
		const double g_q = 2.0 * ((E1*E2 - E0*E3)*g_ix + (0.5 - E1*E1 - E3*E3)*g_iy + (E2*E3 + E0*E1)*g_iz);
		const double g_r = 2.0 * ((E1*E3 + E0*E2)*g_ix + (E2*E3 - E0*E1)*g_iy + (0.5 - E1*E1 - E2*E2)*g_iz);
		const double wc_p = 0.0;
		const double wc_q = 0.0;
		const double wc_r = 0.0;

		gsl_vector_set(actualMeasurement_, 0, gsl_vector_get(calibratedAccel_, 0) - g_p);
		gsl_vector_set(actualMeasurement_, 1, gsl_vector_get(calibratedAccel_, 1) - g_q);
		gsl_vector_set(actualMeasurement_, 2, gsl_vector_get(calibratedAccel_, 2) - g_r);
		gsl_vector_set(actualMeasurement_, 3, gsl_vector_get(calibratedAngularVel_, 0) - wc_p);
		gsl_vector_set(actualMeasurement_, 4, gsl_vector_get(calibratedAngularVel_, 1) - wc_q);
		gsl_vector_set(actualMeasurement_, 5, gsl_vector_get(calibratedAngularVel_, 2) - wc_r);
	}

	// advance time step
	++step2;

	// 2. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
	if (!filter.updateMeasurement(step2, actualMeasurement_, NULL, R)) return false;

	// save K(k), x(k) & P(k)
	{
		const gsl_vector *x_hat = filter.getEstimatedState();
		const gsl_matrix *K = filter.getKalmanGain();
		const gsl_matrix *P = filter.getStateErrorCovarianceMatrix();

#if 0
		// FIXME [modify] >> wrt body frame, but not initial frame
		const double &Ax = gsl_vector_get(x_hat, 6);
		const double &Ay = gsl_vector_get(x_hat, 7);
		const double &Az = gsl_vector_get(x_hat, 8);
		const double &E0 = gsl_vector_get(x_hat, 9);
		const double &E1 = gsl_vector_get(x_hat, 10);
		const double &E2 = gsl_vector_get(x_hat, 11);
		const double &E3 = gsl_vector_get(x_hat, 12);
		const double &Wx = gsl_vector_get(x_hat, 13);
		const double &Wy = gsl_vector_get(x_hat, 14);
		const double &Wz = gsl_vector_get(x_hat, 15);
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
		const double &Ap = gsl_vector_get(x_hat, 6);
		const double &Aq = gsl_vector_get(x_hat, 7);
		const double &Ar = gsl_vector_get(x_hat, 8);
		const double &E0 = gsl_vector_get(x_hat, 9);
		const double &E1 = gsl_vector_get(x_hat, 10);
		const double &E2 = gsl_vector_get(x_hat, 11);
		const double &E3 = gsl_vector_get(x_hat, 12);
		const double &Wp = gsl_vector_get(x_hat, 13);
		const double &Wq = gsl_vector_get(x_hat, 14);
		const double &Wr = gsl_vector_get(x_hat, 15);

		const double Ax = 2.0 * ((0.5 - E2*E2 - E3*E3) * Ap + (E1*E2 - E0*E3) * Aq + (E1*E3 + E0*E2) * Ar);
		const double Ay = 2.0 * ((E1*E2 + E0*E3) * Ap + (0.5 - E1*E1 - E3*E3) * Aq + (E2*E3 - E0*E1) * Ar);
		const double Az = 2.0 * ((E1*E3 - E0*E2) * Ap + (E2*E3 + E0*E1) * Aq + (0.5 - E1*E1 - E2*E2) * Ar);
		const double Wx = 2.0 * ((0.5 - E2*E2 - E3*E3) * Wp + (E1*E2 - E0*E3) * Wq + (E1*E3 + E0*E2) * Wr);
		const double Wy = 2.0 * ((E1*E2 + E0*E3) * Wp + (0.5 - E1*E1 - E3*E3) * Wq + (E2*E3 - E0*E1) * Wr);
		const double Wz = 2.0 * ((E1*E3 - E0*E2) * Wp + (E2*E3 + E0*E1) * Wq + (0.5 - E1*E1 - E2*E2) * Wr);
#endif

		gsl_vector_set(currAccel_, 0, Ax);
		gsl_vector_set(currAccel_, 1, Ay);
		gsl_vector_set(currAccel_, 2, Az);
		gsl_vector_set(currAngularVel_, 0, Wx);
		gsl_vector_set(currAngularVel_, 1, Wy);
		gsl_vector_set(currAngularVel_, 2, Wz);

		// TODO [check] >>
#if 1
		gsl_vector_set(currVel_, 0, gsl_vector_get(prevVel_, 0) + Ax * Ts_);
		gsl_vector_set(currPos_, 0, gsl_vector_get(prevPos_, 0) + gsl_vector_get(prevVel_, 0) * Ts_ + 0.5 * Ax * Ts_*Ts_);
		gsl_vector_set(currVel_, 1, gsl_vector_get(prevVel_, 1) + Ay * Ts_);
		gsl_vector_set(currPos_, 1, gsl_vector_get(prevPos_, 1) + gsl_vector_get(prevVel_, 1) * Ts_ + 0.5 * Ay * Ts_*Ts_);
		gsl_vector_set(currVel_, 2, gsl_vector_get(prevVel_, 2) + Az * Ts_);
		gsl_vector_set(currPos_, 2, gsl_vector_get(prevPos_, 2) + gsl_vector_get(prevVel_, 2) * Ts_ + 0.5 * Az * Ts_*Ts_);
#else
		gsl_vector_set(currPos_, 0, Py);
		gsl_vector_set(currPos_, 1, Py);
		gsl_vector_set(currPos_, 2, Pz);
		gsl_vector_set(currVel_, 0, Vx);
		gsl_vector_set(currVel_, 1, Vy);
		gsl_vector_set(currVel_, 2, Vz);
#endif
		//gsl_vector_set(currAngle_, 0, gsl_vector_get(prevAngle_, 0) + Wx * Ts_);
		//gsl_vector_set(currAngle_, 1, gsl_vector_get(prevAngle_, 1) + Wy * Ts_);
		//gsl_vector_set(currAngle_, 2, gsl_vector_get(prevAngle_, 2) + Wz * Ts_);
		gsl_vector_set(currQuaternion_, 0, E0);
		gsl_vector_set(currQuaternion_, 1, E1);
		gsl_vector_set(currQuaternion_, 2, E2);
		gsl_vector_set(currQuaternion_, 3, E3);

		gsl_vector_memcpy(prevPos_, currPos_);
		gsl_vector_memcpy(prevVel_, currVel_);
		gsl_vector_memcpy(prevQuaternion_, currQuaternion_);
	}

	return true;
}
