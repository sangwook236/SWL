#include "stdafx.h"
#include "swl/Config.h"
#include "DataDefinition.h"
#include <boost/math/constants/constants.hpp>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

namespace {

const double deg2rad = boost::math::constants::pi<double>() / 180.0;
const double lambda = 36.368 * deg2rad;  // latitude [rad]
const double phi = 127.364 * deg2rad;  // longitude [rad]
const double h = 71.0;  // altitude: 71 ~ 82 [m]
const double sin_lambda = std::sin(lambda);
const double sin_2lambda = std::sin(2 * lambda);

}  // unnamed namespace

// [ref] wikipedia: Gravity of Earth
// (latitude, longitude, altitude) = (lambda, phi, h) = (36.368, 127.364, 71.0)
// g(lambda, h) = 9.780327 * (1 + 0.0053024 * sin(lambda)^2 - 0.0000058 * sin(2 * lambda)^2) - 3.086 * 10^-6 * h
/*static*/ const double EarthData::REF_GRAVITY = 9.780327 * (1.0 + 0.0053024 * sin_lambda*sin_lambda - 0.0000058 * sin_2lambda*sin_2lambda) - 3.086e-6 * h;  // [m/sec^2]

// [ref] "The Global Positioning System and Inertial Navigation", Jay Farrell & Matthew Barth, pp. 22
/*static*/ const double EarthData::REF_ANGULAR_VEL = 7.292115e-5;  // [rad/sec]

// [ref] "The Global Positioning System and Inertial Navigation", Jay Farrell & Matthew Barth, pp. 27
/*static*/ void EarthData::geodetic_to_ecef(const Geodetic &geodetic, ECEF &pos)
{
	// semi-major axis length
	const double a = 6378137.0;  // [m]
	// semi-minor axis length
	const double b = 6356752.3142;  // [m]

	// the flatness of the ellipsoid
	const double f = (a - b) / a;
	// the eccentricity of the ellipsoid
	const double e = std::sqrt(f * (2.0 - f));

	const double sin_lambda = std::sin(geodetic.lat);
	const double cos_lambda = std::cos(geodetic.lat);
	const double sin_phi = std::sin(geodetic.lon);
	const double cos_phi = std::cos(geodetic.lon);

	// the lenght of the normal to the ellipsoid
	const double N = a / std::sqrt(1.0 - e*e*sin_lambda*sin_lambda);

	pos.x = (N + geodetic.alt) * cos_lambda * cos_phi;
	pos.y = (N + geodetic.alt) * cos_lambda * sin_phi;
	pos.z = (N * (1.0 - e*e) + geodetic.alt) * sin_lambda;
}

}  // namespace swl
