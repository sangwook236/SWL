#include "stdafx.h"
#include "swl/Config.h"
#include "Adis16350Interface.h"
#include "adisusbz/AdisUsbz.h"
#include <boost/math/constants/constants.hpp>
#include <fstream>
#include <iostream>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

namespace {

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

//-----------------------------------------------------------------------------
//

struct Adis16350ThreadFunctor
{
	Adis16350ThreadFunctor()
	{}
	~Adis16350ThreadFunctor()
	{}

public:
	void operator()()
	{
		std::cout << "ADIS16350 worker thread is started" << std::endl;

		while (true)
		{
			// FIXME [add] >>

			boost::this_thread::yield();
		}

		std::cout << "ADIS16350 worker thread is terminated" << std::endl;
	}

private:
};

}  // unnamed namespace

Adis16350Interface::Adis16350Interface()
: adis_(new AdisUsbz()),
#if defined(UNICODE) || defined(_UNICODE)
  deviceName_(L"\\\\.\\Ezusb-0"),
#else
  deviceName_("\\\\.\\Ezusb-0"),
#endif
  accelCalibrationParam_(ACCEL_CALIB_PARAM_NUM, 0.0), accelCalibrationCovariance_(ACCEL_CALIB_PARAM_NUM * ACCEL_CALIB_PARAM_NUM, 0.0),
  gyroCalibrationParam_(GYRO_CALIB_PARAM_NUM, 0.0), gyroCalibrationCovariance_(GYRO_CALIB_PARAM_NUM * GYRO_CALIB_PARAM_NUM, 0.0)
{
	if (!adis_)
		throw std::runtime_error("fail to create ADIS16350");

	if (!adis_->Initialize(deviceName_.c_str()))
		throw std::runtime_error("fail to initialize ADIS16350");

	// create ADIS16350 worker thread
	//workerThread_.reset(new boost::thread(Adis16350ThreadFunctor()));
}

Adis16350Interface::~Adis16350Interface()
{
	//workerThread_.reset();
	adis_.reset();
}

bool Adis16350Interface::readData(ImuData::Accel &accel, ImuData::Gyro &gyro) const
{
	if (!adis_) return false;

	const short rawXGyro = adis_->ReadInt14(ADIS16350_XGYRO_OUT) & 0x3FFF;
	const short rawYGyro = adis_->ReadInt14(ADIS16350_YGYRO_OUT) & 0x3FFF;
	const short rawZGyro = adis_->ReadInt14(ADIS16350_ZGYRO_OUT) & 0x3FFF;
	const short rawXAccel = adis_->ReadInt14(ADIS16350_XACCL_OUT) & 0x3FFF;
	const short rawYAccel = adis_->ReadInt14(ADIS16350_YACCL_OUT) & 0x3FFF;
	const short rawZAccel = adis_->ReadInt14(ADIS16350_ZACCL_OUT) & 0x3FFF;

	accel.x = ((rawXAccel & 0x2000) == 0x2000 ? (rawXAccel - 0x4000) : rawXAccel) * ADIS16350_ACCL_SCALE_FACTOR;
	accel.y = ((rawYAccel & 0x2000) == 0x2000 ? (rawYAccel - 0x4000) : rawYAccel) * ADIS16350_ACCL_SCALE_FACTOR;
	accel.z = ((rawZAccel & 0x2000) == 0x2000 ? (rawZAccel - 0x4000) : rawZAccel) * ADIS16350_ACCL_SCALE_FACTOR;

	gyro.x = ((rawXGyro & 0x2000) == 0x2000 ? (rawXGyro - 0x4000) : rawXGyro) * ADIS16350_GYRO_SCALE_FACTOR;
	gyro.y = ((rawYGyro & 0x2000) == 0x2000 ? (rawYGyro - 0x4000) : rawYGyro) * ADIS16350_GYRO_SCALE_FACTOR;
	gyro.z = ((rawZGyro & 0x2000) == 0x2000 ? (rawZGyro - 0x4000) : rawZGyro) * ADIS16350_GYRO_SCALE_FACTOR;

	return true;
}

bool Adis16350Interface::testAdisUsbz(const size_t loopCount)
{
	ImuData::Accel accel(0.0, 0.0, 0.0);
	ImuData::Gyro gyro(0.0, 0.0, 0.0);

	size_t loop = 0;
	while (loop++ < loopCount)
	{
		if (!readData(accel, gyro))
			return false;

		std::cout << accel.x << ", " << accel.y << ", " << accel.z << " ; " <<
			gyro.x << ", " << gyro.y << ", " << gyro.z << std::endl;
	}

	return true;
}

bool Adis16350Interface::loadCalibrationParam(const std::string &filename)
{
	std::ifstream stream(filename.c_str());
	if (!stream)
	{
		std::cout << "file not found !!!" << std::endl;
		return false;
	}

	std::string line_str;
	double val;

	// load acceleration parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	if (!stream.eof())
	{
		std::getline(stream, line_str);

		std::istringstream sstream(line_str);
		for (size_t i = 0; i < ACCEL_CALIB_PARAM_NUM; ++i)
		{
			sstream >> val;
			accelCalibrationParam_[i] = val;
		}
	}
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	// load covariance of acceleration parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	if (!stream.eof())
	{
		for (size_t i = 0; i < ACCEL_CALIB_PARAM_NUM; ++i)
		{
			std::getline(stream, line_str);

			std::istringstream sstream(line_str);
			for (size_t j = 0; j < ACCEL_CALIB_PARAM_NUM; ++j)
			{
				sstream >> val;
				accelCalibrationCovariance_[i*ACCEL_CALIB_PARAM_NUM + j] = val;
			}
		}
	}
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	// load gyro parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	if (!stream.eof())
	{
		std::getline(stream, line_str);

		std::istringstream sstream(line_str);
		for (size_t i = 0; i < GYRO_CALIB_PARAM_NUM; ++i)
		{
			sstream >> val;
			gyroCalibrationParam_[i] =  val;
		}
	}
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	// load covariance of gyro parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	if (!stream.eof())
	{
		for (size_t i = 0; i < GYRO_CALIB_PARAM_NUM; ++i)
		{
			std::getline(stream, line_str);

			std::istringstream sstream(line_str);
			for (size_t j = 0; j < GYRO_CALIB_PARAM_NUM; ++j)
			{
				sstream >> val;
				gyroCalibrationCovariance_[i*GYRO_CALIB_PARAM_NUM + j] = val;
			}
		}
	}
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::cout << "file format error !!!" << std::endl;
		return false;
	}

	stream.close();

	return true;
}

void Adis16350Interface::calculateCalibratedAcceleration(const ImuData::Accel &lg, ImuData::Accel &a_calibrated) const
{
	const double &b_gx = accelCalibrationParam_[0];
	const double &b_gy = accelCalibrationParam_[1];
	const double &b_gz = accelCalibrationParam_[2];
	const double &s_gx = accelCalibrationParam_[3];
	const double &s_gy = accelCalibrationParam_[4];
	const double &s_gz = accelCalibrationParam_[5];
	const double &theta_gyz = accelCalibrationParam_[6];
	const double &theta_gzx = accelCalibrationParam_[7];
	const double &theta_gzy = accelCalibrationParam_[8];

	const double &l_gx = lg.x;
	const double &l_gy = lg.y;
	const double &l_gz = lg.z;

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

	a_calibrated.x = g_x;
	a_calibrated.y = g_y;
	a_calibrated.z = g_z;
}

void Adis16350Interface::calculateCalibratedAngularRate(const ImuData::Gyro &lw, ImuData::Gyro &w_calibrated) const
{
	const double &b_wx = gyroCalibrationParam_[0];
	const double &b_wy = gyroCalibrationParam_[1];
	const double &b_wz = gyroCalibrationParam_[2];

	const double &l_wx = lw.x;
	const double &l_wy = lw.y;
	const double &l_wz = lw.z;

	const double w_x = l_wx - b_wx;
	const double w_y = l_wy - b_wy;
	const double w_z = l_wz - b_wz;

	w_calibrated.x = w_x;
	w_calibrated.y = w_y;
	w_calibrated.z = w_z;
}

bool Adis16350Interface::setInitialAttitude(const size_t Ninitial, ImuData::Accel &initialGravity, ImuData::Gyro &initialAngularVel) const
{
	ImuData::Accel sumAccel(0.0, 0.0, 0.0);
	ImuData::Gyro sumGyro(0.0, 0.0, 0.0);

	ImuData::Accel measuredAccel(0.0, 0.0, 0.0), calibratedAccel(0.0, 0.0, 0.0);
	ImuData::Gyro measuredAngularVel(0.0, 0.0, 0.0), calibratedAngularVel(0.0, 0.0, 0.0);
	for (size_t i = 0; i < Ninitial; ++i)
	{
		if (!readData(measuredAccel, measuredAngularVel))
			return false;

		calculateCalibratedAcceleration(measuredAccel, calibratedAccel);
		calculateCalibratedAngularRate(measuredAngularVel, calibratedAngularVel);

		sumAccel.x += calibratedAccel.x;
		sumAccel.y += calibratedAccel.y;
		sumAccel.z += calibratedAccel.z;
		sumGyro.x += calibratedAngularVel.x;
		sumGyro.y += calibratedAngularVel.y;
		sumGyro.z += calibratedAngularVel.z;
	}

	initialGravity.x = sumAccel.x / Ninitial;
	initialGravity.y = sumAccel.y / Ninitial;
	initialGravity.z = sumAccel.z / Ninitial;
	initialAngularVel.x = sumGyro.x / Ninitial;
	initialAngularVel.y = sumGyro.y / Ninitial;
	initialAngularVel.z = sumGyro.z / Ninitial;

	return true;
}

void Adis16350Interface::setInitialAttitude(const std::vector<ImuData::Accel> &accels, const std::vector<ImuData::Gyro> &gyros, ImuData::Accel &initialGravity, ImuData::Gyro &initialAngularVel) const
{
	const size_t Naccel = accels.size();

	ImuData::Accel sumAccel(0.0, 0.0, 0.0);
	ImuData::Accel calibratedAccel(0.0, 0.0, 0.0);
	for (std::vector<ImuData::Accel>::const_iterator it = accels.begin(); it != accels.end(); ++it)
	{
		calculateCalibratedAcceleration(*it, calibratedAccel);

		sumAccel.x += calibratedAccel.x;
		sumAccel.y += calibratedAccel.y;
		sumAccel.z += calibratedAccel.z;
	}

	//
	const size_t Ngyro = gyros.size();

	ImuData::Gyro sumGyro(0.0, 0.0, 0.0);
	ImuData::Gyro calibratedAngularVel(0.0, 0.0, 0.0);
	for (std::vector<ImuData::Gyro>::const_iterator it = gyros.begin(); it != gyros.end(); ++it)
	{
		calculateCalibratedAngularRate(*it, calibratedAngularVel);

		sumGyro.x += calibratedAngularVel.x;
		sumGyro.y += calibratedAngularVel.y;
		sumGyro.z += calibratedAngularVel.z;
	}

	//
	initialGravity.x = sumAccel.x / Naccel;
	initialGravity.y = sumAccel.y / Naccel;
	initialGravity.z = sumAccel.z / Naccel;
	initialAngularVel.x = sumGyro.x / Ngyro;
	initialAngularVel.y = sumGyro.y / Ngyro;
	initialAngularVel.z = sumGyro.z / Ngyro;
}

/*static*/ bool Adis16350Interface::loadSavedImuData(const std::string &filename, const double REF_GRAVITY, const size_t Nsample, std::vector<ImuData::Accel> &accels, std::vector<ImuData::Gyro> &gyros)
{
	std::ifstream stream(filename.c_str());

	// data format:
	//	Sample #,Time (sec),XgND,X Gryo,YgND,Y Gyro,ZgND,Z Gyro,XaND,X acc,YaND,Y acc,ZaND,Z acc,

	if (!stream.is_open())
	{
		std::cout << "file open error !!!" << std::endl;
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
				std::cout << "file format error !!!" << std::endl;
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
			accels.push_back(ImuData::Accel(xAccel * REF_GRAVITY, yAccel * REF_GRAVITY, zAccel * REF_GRAVITY));  // [m/sec^2]
			gyros.push_back(ImuData::Gyro(xGyro * deg2rad, yGyro * deg2rad, zGyro * deg2rad));  // [rad/sec]
		}
	}

	if (accels.empty() || gyros.empty())
	{
		std::cout << "data error !!!" << std::endl;
		return false;
	}

	stream.close();

	return true;
}

}  // namespace swl
