#if !defined(__SWL_GPS_AIDED_IMU_FILTER_APP__GPS_AIDED_IMU_FILTER_RUNNER__H_)
#define __SWL_GPS_AIDED_IMU_FILTER_APP__GPS_AIDED_IMU_FILTER_RUNNER__H_ 1


#include "swl/rnd_util/UnscentedKalmanFilterWithAdditiveNoise.h"
#include "GpsAidedImuSystem.h"
#include "DataDefinition.h"
#include <gsl/gsl_blas.h>
#include <boost/smart_ptr.hpp>
#include <vector>


namespace swl {

class UnscentedKalmanFilterWithAdditiveNoise;
class GpsAidedImuSystem;

class GpsAidedImuFilterRunner
{
public:
	//typedef GpsAidedImuFilterRunner base_type;

public:
	GpsAidedImuFilterRunner(const ImuData::Accel &initialGravity, const ImuData::Gyro &initialAngularVel);
	~GpsAidedImuFilterRunner();

private:
	GpsAidedImuFilterRunner(const GpsAidedImuFilterRunner &rhs);
	GpsAidedImuFilterRunner & operator=(const GpsAidedImuFilterRunner &rhs);

public:
	//
	void initialize();
	void finalize();

	//
	bool runStep(const ImuData::Accel &measuredAccel, const ImuData::Gyro &measuredAngularVel, const EarthData::ECEF &measuredGpsECEF, const EarthData::Speed &measuredGpsSpeed);

	//
	const gsl_vector * getFilteredPos() const  {  return currPos_;  }
	const gsl_vector * getFilteredVel() const  {  return currVel_;  }
	const gsl_vector * getFilteredAccel() const  {  return currAccel_;  }
	const gsl_vector * getFilteredQuaternion() const  {  return currQuaternion_;  }
	const gsl_vector * getFilteredAngularVel() const  {  return currAngularVel_;  }

private:
	const size_t dim_;
	const double Ts_;

	boost::scoped_ptr<UnscentedKalmanFilterWithAdditiveNoise> filter_;
	boost::scoped_ptr<GpsAidedImuSystem> system_;

	const ImuData::Accel &initialGravity_;
	const ImuData::Gyro &initialAngularVel_;

	gsl_matrix *Q_;
	gsl_matrix *R_;

	gsl_vector *actualMeasurement_;

	gsl_vector *currPos_;  // wrt initial frame
	gsl_vector *prevPos_;  // wrt initial frame
	gsl_vector *currVel_;  // wrt initial frame
	gsl_vector *prevVel_;  // wrt initial frame
	gsl_vector *currQuaternion_;  // wrt initial frame
	gsl_vector *prevQuaternion_;  // wrt initial frame

	gsl_vector *currAccel_;  // wrt initial frame
	gsl_vector *currAngularVel_;  // wrt initial frame

	size_t step_;
};

}  // namespace swl


#endif  // __SWL_GPS_AIDED_IMU_FILTER_APP__GPS_AIDED_IMU_FILTER_RUNNER__H_
