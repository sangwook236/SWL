#if !defined(__SWL_GPS_AIDED_IMU_FILTER_APP__DATA_DEFINITION__H_)
#define __SWL_GPS_AIDED_IMU_FILTER_APP__DATA_DEFINITION__H_ 1


#if defined(min)
#undef min
#endif

namespace swl {

//-----------------------------------------------------------------------------
//

struct EarthData
{
	static const double REF_GRAVITY;
	static const double REF_ANGULAR_VEL;

	struct Geodetic
	{
		Geodetic(const double &_lat, const double &_lon, const double &_alt)
		: lat(_lat), lon(_lon), alt(_alt)
		{}
		Geodetic(const Geodetic &rhs)
		: lat(rhs.lat), lon(rhs.lon), alt(rhs.alt)
		{}

		double lat, lon, alt;
	};

	struct ECEF
	{
		ECEF(const double &_x, const double &_y, const double &_z)
		: x(_x), y(_y), z(_z)
		{}
		ECEF(const ECEF &rhs)
		: x(rhs.x), y(rhs.y), z(rhs.z)
		{}

		double x, y, z;
	};

	struct Speed
	{
		Speed(const double &_val)
		: val(_val)
		{}
		Speed(const Speed &rhs)
		: val(rhs.val)
		{}

		double val;
	};

	struct Time
	{
		Time(const int &_hour, const int &_min, const int &_sec, const int &_msec)
		: hour(_hour), min(_min), sec(_sec), msec(_msec)
		{}
		Time(const Time &rhs)
		: hour(rhs.hour), min(rhs.min), sec(rhs.sec), msec(rhs.msec)
		{}

		int hour, min, sec, msec;
	};

	// transform from geodetic coordinates to rectangular ECEF coordinates
	static void geodetic_to_ecef(const Geodetic &geodetic, ECEF &pos);
};

//-----------------------------------------------------------------------------
//

struct ImuData
{
	struct Accel
	{
		Accel(const double &_x, const double &_y, const double &_z)
		: x(_x), y(_y), z(_z)
		{}
		Accel(const Accel &rhs)
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
};

}  // namespace swl


#endif  // __SWL_GPS_AIDED_IMU_FILTER_APP__DATA_DEFINITION__H_
