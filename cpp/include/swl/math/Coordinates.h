#if !defined(__SWL_MATH__COORDINATES__H_)
#define __SWL_MATH__COORDINATES__H_ 1


#include "swl/math/ExportMath.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class PolarCoord

struct SWL_MATH_API PolarCoord
{
public:
	PolarCoord(double x = 0.0, double y = 0.0);
	PolarCoord(const PolarCoord& rhs);
	~PolarCoord();

	PolarCoord& operator=(const PolarCoord& rhs);

public:
	/// accessor & mutator
	double& r()  {  return r_;  }
	double r() const  {  return r_;  }
	double& theta()  {  return theta_;  }
	double theta() const  {  return theta_;  }

	///
	void toCartesian(double& x, double& y);

private:
	///
	void toPolar(double x, double y);

private:
	/// r >= 0
	double r_;
	/// -pi <= theta <= pi
	double theta_;
};


//-----------------------------------------------------------------------------------------
// class CylindricalCoord

struct SWL_MATH_API CylindricalCoord
{
public:
	CylindricalCoord(double x = 0.0, double y = 0.0, double z = 0.0);
	CylindricalCoord(const CylindricalCoord& rhs);
	~CylindricalCoord();

	CylindricalCoord& operator=(const CylindricalCoord& rhs);

public:
	/// accessor & mutator
	double& r()  {  return r_;  }
	double r() const  {  return r_;  }
	double& theta()  {  return theta_;  }
	double theta() const  {  return theta_;  }
	double& z()  {  return z_;  }
	double z() const  {  return z_;  }

	///
	void toCartesian(double& x, double& y, double& z);

private:
	///
	void toCylindrical(double x, double y, double z);

private:
	/// r >= 0
	double r_;
	/// -pi <= theta <= pi
	/// theta measures the angle from positive x-axis
	double theta_;
	double z_;
};


//-----------------------------------------------------------------------------------------
// class SphericalCoord

struct SWL_MATH_API SphericalCoord
{
public:
	SphericalCoord(double x = 0.0, double y = 0.0, double z = 0.0);
	SphericalCoord(const SphericalCoord& rhs);
	~SphericalCoord();

	SphericalCoord& operator=(const SphericalCoord& rhs);

public:
	/// accessor & mutator
	double& r()  {  return r_;  }
	double r() const  {  return r_;  }
	double& theta()  {  return theta_;  }
	double theta() const  {  return theta_;  }
	double& phi()  {  return phi_;  }
	double phi() const  {  return phi_;  }

	///
	void toCartesian(double& x, double& y, double& z);

private:
	///
	void toSpherical(double x, double y, double z);

private:
	/// r >= 0
	double r_;
	/// -pi <= theta <= pi
	/// theta measures the angle from positive x-axis
	double theta_;
	/// 0 <= phi <= pi
	/// phi measures the angle from positive z-axis
	double phi_;
};

}  // namespace swl


#endif  // __SWL_MATH__COORDINATES__H_
