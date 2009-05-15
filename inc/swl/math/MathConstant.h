#if !defined(__SWL_MATH__MATH_CONSTANT__H_)
#define __SWL_MATH__MATH_CONSTANT__H_ 1


#include "swl/math/ExportMath.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// struct MathConstant

struct SWL_MATH_API MathConstant
{
public :
	/// axis & plane
	enum AXIS { AXIS_NULL = 0x0, AXIS_X = 0x1, AXIS_Y = 0x2, AXIS_Z = 0x4 };
	enum PLANE { PLANE_NULL = 0x0, PLANE_XY = 0x1, PLANE_YZ = 0x2, PLANE_ZX = 0x4, PLANE_NORMAL = 0x8 };

	/// constants
	static const double INF;
	static const double NaN;

	static const double E;

	static const double PI;  // pi
	static const double _2_PI;  // 2 * pi
	static const double _4_PI;  // 4 * pi
	static const double PI_2;  // pi / 2
	static const double PI_3;  // pi / 3
	static const double PI_4;  // pi / 4
	static const double PI_6;  // pi / 6

	static const double TO_RAD;
	static const double TO_DEG;

	static const double SIN_30;  // sin(30)
	static const double SIN_45;  // sin(45)
	static const double SIN_60;  // sin(60)
	static const double COS_30;  // cos(30)
	static const double COS_45;  // cos(45)
	static const double COS_60;  // cos(60)
	static const double TAN_30;  // tan(30)
	static const double TAN_45;  // tan(45)
	static const double TAN_60;  // tan(60)

	static const double SQRT_2;  // sqrt(2)
	static const double SQRT_3;  // sqrt(3)
	static const double SQRT_5;  // sqrt(5)

	///
	static const double TOL_1;  // 1.0e-1
	static const double TOL_2;  // 1.0e-2
	static const double TOL_3;  // 1.0e-3
	static const double TOL_4;  // 1.0e-4
	static const double TOL_5;  // 1.0e-5
	static const double TOL_6;  // 1.0e-6
	static const double TOL_7;  // 1.0e-7
	static const double TOL_8;  // 1.0e-8
	static const double TOL_9;  // 1.0e-9
	static const double TOL_10;  // 1.0e-10

	///  non-constant
	static double EPS;

	static int ITERATION_LIMIT;
};

}  // namespace swl


#endif  // __SWL_MATH__MATH_CONSTANT__H_
