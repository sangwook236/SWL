#if !defined(__SWL_GPS_AIDED_IMU_FILTER_APP__ADIS16350_INTERFACE__H_)
#define __SWL_GPS_AIDED_IMU_FILTER_APP__ADIS16350_INTERFACE__H_ 1


#include "DataDefinition.h"
#include <boost/smart_ptr.hpp>
//#include <boost/thread.hpp>
#include <string>
#include <vector>

class AdisUsbz;


namespace swl {

class Adis16350Interface
{
public:
	//typedef Adis16350Interface base_type;

public:
	Adis16350Interface();
	~Adis16350Interface();

private:
	Adis16350Interface(const Adis16350Interface &rhs);
	Adis16350Interface & operator=(const Adis16350Interface &rhs);

public:
	bool readData(ImuData::Accel &accel, ImuData::Gyro &gyro) const;

	bool loadCalibrationParam(const std::string &filename);

	bool setInitialAttitude(const size_t Ninitial, ImuData::Accel &initialGravity, ImuData::Gyro &initialAngularVel) const;
	void setInitialAttitude(const std::vector<ImuData::Accel> &accels, const std::vector<ImuData::Gyro> &gyros, ImuData::Accel &initialGravity, ImuData::Gyro &initialAngularVel) const;

	void calculateCalibratedAcceleration(const ImuData::Accel &lg, ImuData::Accel &a_calibrated) const;
	void calculateCalibratedAngularRate(const ImuData::Gyro &lw, ImuData::Gyro &w_calibrated) const;

	static bool loadSavedImuData(const std::string &filename, const double REF_GRAVITY, const size_t Nsample, std::vector<ImuData::Accel> &accels, std::vector<ImuData::Gyro> &gyros);

	bool testAdisUsbz(const size_t loopCount);

private:
	boost::scoped_ptr<AdisUsbz> adis_;

	//boost::scoped_ptr<boost::thread> workerThread_;

#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring deviceName_;
#else
	const std::string deviceName_;
#endif

	static const size_t ACCEL_CALIB_PARAM_NUM = 9;
	static const size_t GYRO_CALIB_PARAM_NUM = 3;

	std::vector<double> accelCalibrationParam_;
	std::vector<double> accelCalibrationCovariance_;
	std::vector<double> gyroCalibrationParam_;
	std::vector<double> gyroCalibrationCovariance_;
};

}  // namespace swl


#endif  // __SWL_GPS_AIDED_IMU_FILTER_APP__ADIS16350_INTERFACE__H_
