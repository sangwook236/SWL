#if !defined(__SWL_GPS_AIDED_IMU_FILTER_APP__GPS_INTERFACE__H_)
#define __SWL_GPS_AIDED_IMU_FILTER_APP__GPS_INTERFACE__H_ 1


#include "DataDefinition.h"
#include "swl/winutil/WinSerialPort.h"
#include "swl/util/GuardedBuffer.h"
#include <boost/smart_ptr.hpp>
#include <boost/thread.hpp>
#include <string>


namespace swl {

class GpsInterface
{
public:
	//typedef GpsInterface base_type;

public:
#if defined(_UNICODE) || defined(UNICODE)
	GpsInterface(const std::wstring &portName, const unsigned int baudRate);
#else
	GpsInterface(const std::string &portName, const unsigned int baudRate);
#endif
	~GpsInterface();

private:
	GpsInterface(const GpsInterface &rhs);
	GpsInterface & operator=(const GpsInterface &rhs);

public:
	bool isConnected() const  {  return isConnected_;  }

	bool readData(EarthData::Geodetic &pos, EarthData::Speed &speed) const;

	bool setInitialState(const size_t Ninitial, EarthData::ECEF &initialPosition, EarthData::Speed &initialSpeed) const;
	void setInitialState(const std::vector<EarthData::Geodetic> &positions, const std::vector<EarthData::Speed> &speeds, EarthData::ECEF &initialPosition, EarthData::Speed &initialSpeed) const;

private:
	WinSerialPort serialPort_;
	mutable GuardedByteBuffer recvBuffer_;

	boost::scoped_ptr<boost::thread> workerThread_;

#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring portName_;
#else
	const std::string portName_;
#endif
	const unsigned int baudRate_;

	bool isConnected_;
};

}  // namespace swl


#endif  // __SWL_GPS_AIDED_IMU_FILTER_APP__GPS_INTERFACE__H_
