#include "swl/math/Coordinates.h"
#include "swl/math/MathExt.h"
#include "swl/math/MathUtil.h"
#include <cmath>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(PI)  //  for djgpp
#undef PI
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// class PolarCoord

PolarCoord::PolarCoord(double x /*= 0.0*/, double y /*= 0.0*/)
{
	toPolar(x, y);
}

PolarCoord::PolarCoord(const PolarCoord& rhs)
: r_(rhs.r_), theta_(rhs.theta_)
{}

PolarCoord::~PolarCoord()  {}

PolarCoord& PolarCoord::operator=(const PolarCoord& rhs)
{
	if (this == &rhs) return *this;
	r_ = rhs.r_;
	theta_ = rhs.theta_;
	return *this;
}

void PolarCoord::toCartesian(double& x, double& y)
{
	 x = r_ * cos(theta_);
	 y = r_ * sin(theta_);
}

void PolarCoord::toPolar(double x, double y)
{
	r_ = sqrt(x*x + y*y);
	if (MathUtil::isZero(r_))
	{
		theta_ = 0.0;
		return;
	}

	if (MathUtil::isZero(x))
		theta_ = y >= 0 ? MathConstant::PI_2 : -MathConstant::PI_2;
	else theta_ = atan2(y, x);
}


//-----------------------------------------------------------------------------------------
// class CylindricalCoord

CylindricalCoord::CylindricalCoord(double x /*= 0.0*/, double y /*= 0.0*/, double z /*= 0.0*/)
{
	toCylindrical(x, y, z);
}

CylindricalCoord::CylindricalCoord(const CylindricalCoord& rhs)
: r_(rhs.r_), theta_(rhs.theta_), z_(rhs.z_)
{}

CylindricalCoord::~CylindricalCoord()  {}

CylindricalCoord& CylindricalCoord::operator=(const CylindricalCoord& rhs)
{
	if (this == &rhs) return *this;
	r_ = rhs.r_;
	theta_ = rhs.theta_;
	z_ = rhs.z_;
	return *this;
}

void CylindricalCoord::toCartesian(double& x, double& y, double& z)
{
	 x = r_ * cos(theta_);
	 y = r_ * sin(theta_);
	 z = z_;
}

void CylindricalCoord::toCylindrical(double x, double y, double z)
{
	z_ = z;

	r_ = sqrt(x*x + y*y);
	if (MathUtil::isZero(r_))
	{
		theta_ = 0.0;
		return;
	}

	if (MathUtil::isZero(x))
		theta_ = y >= 0 ? MathConstant::PI_2 : -MathConstant::PI_2;
	else theta_ = atan2(y, x);
}


//-----------------------------------------------------------------------------------------
// class SphericalCoord

SphericalCoord::SphericalCoord(double x /*= 0.0*/, double y /*= 0.0*/, double z /*= 0.0*/)
{
	toSpherical(x, y, z);
}

SphericalCoord::SphericalCoord(const SphericalCoord& rhs)
: r_(rhs.r_), theta_(rhs.theta_), phi_(rhs.phi_)
{}

SphericalCoord::~SphericalCoord()  {}

SphericalCoord& SphericalCoord::operator=(const SphericalCoord& rhs)
{
	if (this == &rhs) return *this;
	r_ = rhs.r_;
	theta_ = rhs.theta_;
	phi_ = rhs.phi_;
	return *this;
}

void SphericalCoord::toCartesian(double& x, double& y, double& z)
{
	 x = r_ * cos(theta_) * sin(phi_);
	 y = r_ * sin(theta_) * sin(phi_);
	 z = r_ * cos(phi_);
}

void SphericalCoord::toSpherical(double x, double y, double z)
{
	r_ = sqrt(x*x + y*y + z*z);
	if (MathUtil::isZero(r_))
	{
		theta_ = phi_ = 0.0;
		return;
	}

	phi_ = MathUtil::isZero(z) ? MathConstant::PI_2 : atan2(sqrt(x*x + y*y), z);

	//  when sin(phi) = 0
	if (MathUtil::isZero(phi_) && MathUtil::isZero(phi_ - MathConstant::PI))
		theta_ = 0.0;
	else if (MathUtil::isZero(x))
		theta_ = y*sin(phi_) >= 0 ? MathConstant::PI_2 : -MathConstant::PI_2;
	else theta_ = atan2(y, x);
}

}  // namespace swl
