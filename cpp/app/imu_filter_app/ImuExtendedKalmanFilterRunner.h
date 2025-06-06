#if !defined(__SWL_IMU_FILTER_APP__IMU_EXTENDED_KALMAN_FILTER_WITH_CALIBRATION__H_)
#define __SWL_IMU_FILTER_APP__IMU_EXTENDED_KALMAN_FILTER_WITH_CALIBRATION__H_ 1


#include <gsl/gsl_blas.h>
#include <boost/smart_ptr.hpp>
#include <vector>

//#define __USE_RECEIVED_DATA_FROM_ADISUSBZ 1


namespace swl {
class DiscreteExtendedKalmanFilter;
}

class AdisUsbz;


class ImuExtendedKalmanFilterRunner
{
public:
	//typedef ImuExtendedKalmanFilterRunner base_type;

private:
	static const double REF_GRAVITY_;
	static const double REF_ANGULAR_VEL_;

public:
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

public:
	ImuExtendedKalmanFilterRunner(const double Ts, const size_t stateDim, const size_t inputDim, const size_t outputDim, AdisUsbz *adis);
	~ImuExtendedKalmanFilterRunner();

private:
	ImuExtendedKalmanFilterRunner(const ImuExtendedKalmanFilterRunner &rhs);
	ImuExtendedKalmanFilterRunner & operator=(const ImuExtendedKalmanFilterRunner &rhs);

public:
	bool runImuFilter(swl::DiscreteExtendedKalmanFilter &filter, const size_t step, const gsl_vector *measuredAccel, const gsl_vector *measuredAngularVel, const gsl_vector *initialGravity);

	void initializeGravity(const size_t Ninitial);
	void initializeGravity(const size_t Ninitial, const std::vector<Acceleration> &accels, const std::vector<Gyro> &gyros);

	bool loadCalibrationParam(const std::string &filename);
	bool readAdisData(gsl_vector *accel, gsl_vector *gyro) const;

	const gsl_vector * getInitialGravity() const  {  return initialGravity_;  }

	const gsl_vector * getFilteredPos() const  {  return currPos_;  }
	const gsl_vector * getFilteredVel() const  {  return currVel_;  }
	const gsl_vector * getFilteredAccel() const  {  return currAccel_;  }
	const gsl_vector * getFilteredQuaternion() const  {  return currQuaternion_;  }
	const gsl_vector * getFilteredAngularVel() const  {  return currAngularVel_;  }

	bool testAdisUsbz(const size_t loopCount);

	static bool loadSavedImuData(const std::string &filename, const size_t &Nsample, std::vector<Acceleration> &accels, std::vector<Gyro> &gyros);

private:
	void calculateCalibratedAcceleration(const gsl_vector *lg, gsl_vector *a_calibrated) const;
	void calculateCalibratedAngularRate(const gsl_vector *lw, gsl_vector *w_calibrated) const;

private:
	const size_t numAccelParam_;
	const size_t numGyroParam_;
	const size_t dim_;

	const double Ts_;

	boost::scoped_ptr<AdisUsbz> adis_;

	gsl_vector *initialGravity_;
	//gsl_vector *initialAngularVel_;

	gsl_vector *accelCalibrationParam_;
	gsl_matrix *accelCalibrationCovariance_;
	gsl_vector *gyroCalibrationParam_;
	gsl_matrix *gyroCalibrationCovariance_;

	gsl_vector *measuredAccel_;
	gsl_vector *measuredAngularVel_;
	gsl_vector *calibratedAccel_;
	gsl_vector *calibratedAngularVel_;

	gsl_vector *actualMeasurement_;

	gsl_vector *currPos_;  // wrt initial frame
	gsl_vector *prevPos_;  // wrt initial frame
	gsl_vector *currVel_;  // wrt initial frame
	gsl_vector *prevVel_;  // wrt initial frame
	gsl_vector *currQuaternion_;  // wrt initial frame
	gsl_vector *prevQuaternion_;  // wrt initial frame

	gsl_vector *currAccel_;  // wrt initial frame
	gsl_vector *currAngularVel_;  // wrt initial frame
};


#endif  // __SWL_IMU_FILTER_APP__IMU_EXTENDED_KALMAN_FILTER_WITH_CALIBRATION__H_
