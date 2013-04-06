#include "stdafx.h"
#include "swl/Config.h"
#include "GpsInterface.h"
#include <nmea/nmea.h>
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

namespace {

//-----------------------------------------------------------------------------
//

class NmeaParserImpl
{
public:
	enum NMEA_TYPE { NT_NO_SENTENSE, NT_UNDEFINED, NT_RMC, NT_GGA, NT_GSV };

public:
	NmeaParserImpl()
	{
		nmea_property()->trace_func = &NmeaParserImpl::handleTrace;
		nmea_property()->error_func = &NmeaParserImpl::handleError;

		nmea_parser_init(&parser_);
	}
	~NmeaParserImpl()
	{
		nmea_parser_destroy(&parser_);
	}

public:
	bool parse(double &latitude, double &longitude, double &altitude, double &speed, int &hour, int &min, int &sec, int &hsec, const unsigned char *buf = NULL, const size_t len = 0)
	{
		if (NULL != buf && 0 < len) std::copy(buf, buf + len, std::back_inserter(buf_));

		std::vector<unsigned char> sentence;
		const NMEA_TYPE &type = getNmeaSentence(sentence);
		if (NT_RMC == type)
		//if (NT_RMC == type || NT_GGA == type)
		{
			doParse(&sentence[0], sentence.size(), latitude, longitude, altitude, speed, hour, min, sec, hsec);
			return true;
		}
		else if (NT_NO_SENTENSE == type)
		{
			//latitude = longitude = 0.0;
			return false;
		}
		else return true;
	}

private:
	static void handleTrace(const char *str, int len)
	{
		//printf("Trace: ");
		//write(1, str, len);
		//printf("\n");
	}
	static void handleError(const char *str, int len)
	{
		//printf("Error: ");
		//write(1, str, len);
		//printf("\n");
	}

	void doParse(const unsigned char *buf, const size_t len, double &latitude, double &longitude, double &altitude, double &speed, int &hour, int &min, int &sec, int &hsec)
	{
		nmeaINFO info;
		nmea_zero_INFO(&info);  // initialization

		nmea_parse(&parser_, (const char *)buf, len, &info);
		nmeaPOS dpos;
		nmea_info2pos(&info, &dpos);

		//std::cout << "latitude: " << dpos.lat << ", longitude: " << dpos.lon << ", sig: " << info.sig << ", fix: " << info.fix << std::endl;

		latitude = dpos.lat;
		longitude = dpos.lon;

		hour = info.utc.hour;
		min = info.utc.min;
		sec = info.utc.sec;
		hsec = info.utc.hsec;

		//sig = info.sig;
		//fix = info.fix;

		speed = info.speed;
		altitude = info.elv;
		//direction = info.direction;
		//declination = info.declination;

		//PDOP = info.PDOP;
		//HDOP = info.HDOP;
		//VDOP = info.VDOP;
	}

	NMEA_TYPE getNmeaSentence(std::vector<unsigned char> &sentence)
	{
		if (buf_.empty()) return NT_NO_SENTENSE;

		//std::cout << "1***** ";
		//std::copy(buf_.begin(), buf_.end(), std::ostream_iterator<char>(std::cout, ""));
		//std::cout << std::endl;

		for (std::deque<unsigned char>::iterator it = buf_.begin(); it != buf_.end(); )
		{
			if ('$' == *it)
			{
				++it;
				break;
			}
			else it = buf_.erase(it);
		}

		if (buf_.empty()) return NT_NO_SENTENSE;

		std::deque<unsigned char> sent;
		sent.push_back('$');

		std::deque<unsigned char>::iterator it = buf_.begin();
		++it;
		for (; it != buf_.end(); ++it)
		{
			if ('$' == *it)
			{
				it = buf_.erase(buf_.begin(), it);

				sent.clear();
				sent.push_back('$');
			}
			else if ('\r' == *it)
			{
				sent.push_back('\r');

				std::deque<unsigned char>::iterator it2 = it + 1;
				if (it2 != buf_.end() && '\n' == *it2)
				{
					++it2;
					buf_.erase(buf_.begin(), it2);

					sent.push_back('\n');
					sentence.assign(sent.begin(), sent.end());
					//std::cout << "3***** ";
					//std::copy(sentence.begin(), sentence.end(), std::ostream_iterator<char>(std::cout, ""));
					//std::cout << std::endl;
					//std::cout << "2***** ";
					//std::copy(buf_.begin(), buf_.end(), std::ostream_iterator<char>(std::cout, ""));
					//std::cout << std::endl;
					return getNmeaSentenceType(sentence);
				}
			}
			else sent.push_back(*it);
		}

		//std::cout << "2***** ";
		//std::copy(buf_.begin(), buf_.end(), std::ostream_iterator<char>(std::cout, ""));
		//std::cout << std::endl;
		return NT_NO_SENTENSE;
	}

	NMEA_TYPE getNmeaSentenceType(const std::vector<unsigned char> &sentence) const
	{
		const std::string str(sentence.begin(), sentence.end());
		const std::string ss = str.substr(1, 5);
		if (stricmp("GPRMC", ss.c_str()) == 0)
			return NT_RMC;
		else if (stricmp("GPGGA", ss.c_str()) == 0)
			return NT_GGA;
		else if (stricmp("GPGSV", ss.c_str()) == 0)
			return NT_GSV;
		else return NT_UNDEFINED;
	}

private:
	nmeaPARSER parser_;
	std::deque<unsigned char> buf_;
};

//-----------------------------------------------------------------------------
//

boost::condition_variable gps_data_cond_var;
boost::mutex gps_data_mutex;
bool gps_data_ready = false;
bool gps_data_required = false;
bool gps_worker_thread_is_running = false;
double gps_data_lat, gps_data_lon, gps_data_alt, gps_data_speed;
int gps_data_hour, gps_data_min, gps_data_sec, gps_data_hsec;

struct WinSerialPortThreadFunctor
{
	WinSerialPortThreadFunctor(WinSerialPort &serialPort, GuardedByteBuffer &recvBuffer)
	: serialPort_(serialPort), recvBuffer_(recvBuffer)
	{}
	~WinSerialPortThreadFunctor()
	{}

public:
	void operator()()
	{
		std::cout << "win serial port worker thread is started" << std::endl;

		NmeaParserImpl nmeaParser;

		const size_t msgLen = 4095;
		unsigned char msg[msgLen + 1] = { '\0', };

		const unsigned long timeoutInterval_msec = 10;
		const size_t bufferLen = 0;
		double lat, lon, alt, speed;
		int hour, min, sec, hsec;

		gps_worker_thread_is_running = true;
		while (gps_worker_thread_is_running)
		{
			serialPort_.receive(recvBuffer_, timeoutInterval_msec, bufferLen);

			//
			if (!recvBuffer_.isEmpty() && gps_worker_thread_is_running)
			{
				memset(msg, 0, msgLen + 1);

				const size_t len = recvBuffer_.getSize();
				const size_t len2 = len > msgLen ? msgLen : len;
				recvBuffer_.top(msg, len2);
				recvBuffer_.pop(len2);

				//std::cout << msg << std::endl;

				lat = lon = alt = speed = 0.0;
				hour = min = sec = hsec = 0;
				if (nmeaParser.parse(lat, lon, alt, speed, hour, min, sec, hsec, msg, len2))
				{
					//std::cout << "latitude : " << lat << " [rad], longitude : " << lon << " [rad], altitude : " << alt << " [m], speed : " << speed << " [km/h]" << std::endl;
					//std::cout << "latitude : " << nmea_radian2degree(lat) << " [deg], longitude : " << nmea_radian2degree(lon) << " [deg], altitude : " << alt << " [m], speed : " << speed << " [km/h]" << std::endl;

					//lat = lon = alt = speed = 0.0;
					//hour = min = sec = hsec = 0;
					while (nmeaParser.parse(lat, lon, alt, speed, hour, min, sec, hsec))
					{
						//std::cout << "latitude : " << lat << " [rad], longitude : " << lon << " [rad], altitude : " << alt << " [m], speed : " << speed << " [km/h]" << std::endl;
						//std::cout << "latitude : " << nmea_radian2degree(lat) << " [deg], longitude : " << nmea_radian2degree(lon) << " [deg], altitude : " << alt << " [m], speed : " << speed << " [km/h]" << std::endl;
					}

					// extract the last GPS data
					if (gps_data_required)
					{
						{
							boost::lock_guard<boost::mutex> lock(gps_data_mutex);

							gps_data_lat = lat;
							gps_data_lon = lon;
							gps_data_alt = alt;
							gps_data_speed = speed;
							gps_data_hour = hour;
							gps_data_min = min;
							gps_data_sec = sec;
							gps_data_hsec = hsec;

							gps_data_ready = true;
						}

						gps_data_required = false;
						gps_data_cond_var.notify_one();
					}
				}
			}

			boost::this_thread::yield();
		}

		std::cout << "win serial port worker thread is terminated" << std::endl;
	}

private:
	WinSerialPort &serialPort_;
	GuardedByteBuffer &recvBuffer_;
};

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#if defined(_UNICODE) || defined(UNICODE)
GpsInterface::GpsInterface(const std::wstring &portName, const unsigned int baudRate)
#else
GpsInterface::GpsInterface(const std::string &portName, const unsigned int baudRate)
#endif
: serialPort_(), recvBuffer_(), 
  portName_(portName), baudRate_(baudRate), isConnected_(false)
{
	const size_t inQueueSize = 8192, outQueueSize = 8192;
	isConnected_ = serialPort_.connect(portName_.c_str(), baudRate_, inQueueSize, outQueueSize);
	if (isConnected_)
	{
		// create serial port worker thread
		workerThread_.reset(new boost::thread(WinSerialPortThreadFunctor(serialPort_, recvBuffer_)));
	}
	else
		throw std::runtime_error("fail to connect a serial port for GPS");
}

GpsInterface::~GpsInterface()
{
	gps_worker_thread_is_running = false;
	workerThread_.reset();
	if (isConnected_)
	{
		serialPort_.disconnect();
		isConnected_ = false;
	}
}

bool GpsInterface::readData(EarthData::Geodetic &pos, EarthData::Speed &speed, EarthData::Time &utc) const
{
	if (!isConnected_) return false;

	//
	gps_data_ready = false;
	gps_data_required = true;
    boost::unique_lock<boost::mutex> lock(gps_data_mutex);
    while (!gps_data_ready)
    {
        gps_data_cond_var.wait(lock);
    }

	pos.lat = gps_data_lat;
	pos.lon = gps_data_lon;
	pos.alt = gps_data_alt;
	speed.val = gps_data_speed;

	utc.hour = gps_data_hour;
	utc.min = gps_data_min;
	utc.sec = gps_data_sec;
	// FIXME [check] >>
	utc.msec = gps_data_hsec;  //* 10;

	//Sleep(0);

	return true;
}

bool GpsInterface::setInitialState(const size_t Ninitial, EarthData::ECEF &initialPosition, EarthData::Speed &initialSpeed) const
{
	EarthData::ECEF sumECEF(0.0, 0.0, 0.0);
	EarthData::Speed sumSpeed(0.0);

	EarthData::Geodetic measuredGeodetic(0.0, 0.0, 0.0);
	EarthData::ECEF measuredECEF(0.0, 0.0, 0.0);
	EarthData::Speed measuredSpeed(0.0);
	EarthData::Time gpsUtc(0, 0, 0, 0);
	for (size_t i = 0; i < Ninitial; ++i)
	{
		if (!readData(measuredGeodetic, measuredSpeed, gpsUtc))
			return false;

		EarthData::geodetic_to_ecef(measuredGeodetic, measuredECEF);

		sumECEF.x += measuredECEF.x;
		sumECEF.y += measuredECEF.y;
		sumECEF.z += measuredECEF.z;
		sumSpeed.val += measuredSpeed.val;
	}

#if 0
	// use the average
	initialPosition.x = sumECEF.x / Ninitial;
	initialPosition.y = sumECEF.y / Ninitial;
	initialPosition.z = sumECEF.z / Ninitial;
	initialSpeed.val = sumSpeed.val / Ninitial;
#else
	// use the last value
	initialPosition.x = measuredECEF.x;
	initialPosition.y = measuredECEF.y;
	initialPosition.z = measuredECEF.z;
	initialSpeed.val = sumSpeed.val / Ninitial;
#endif

	return true;
}

void GpsInterface::setInitialState(const std::vector<EarthData::Geodetic> &positions, const std::vector<EarthData::Speed> &speeds, EarthData::ECEF &initialPosition, EarthData::Speed &initialSpeed) const
{
	const size_t Npos = positions.size();

	EarthData::ECEF sumECEF(0.0, 0.0, 0.0);
	EarthData::ECEF measuredECEF(0.0, 0.0, 0.0);
	for (std::vector<EarthData::Geodetic>::const_iterator it = positions.begin(); it != positions.end(); ++it)
	{
		EarthData::geodetic_to_ecef(*it, measuredECEF);

		sumECEF.x += measuredECEF.x;
		sumECEF.y += measuredECEF.y;
		sumECEF.z += measuredECEF.z;
	}

	//
	const size_t Nspeed = speeds.size();

	EarthData::Speed sumSpeed(0.0);
	for (std::vector<EarthData::Speed>::const_iterator it = speeds.begin(); it != speeds.end(); ++it)
	{
		sumSpeed.val += it->val;
	}

	//
	initialPosition.x = sumECEF.x / Npos;
	initialPosition.y = sumECEF.y / Npos;
	initialPosition.z = sumECEF.z / Npos;
	initialSpeed.val = sumSpeed.val / Nspeed;
}

}  // namespace swl
