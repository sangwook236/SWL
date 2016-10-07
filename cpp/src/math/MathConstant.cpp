#include "swl/Config.h"
#include "swl/math/MathConstant.h"
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <limits>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(PI)  // for djgpp
#undef PI
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// Math Constant.

///*static*/ const double MathConstant::INF = std::numeric_limits<double>::max();  // 1.79769e+308.
/*static*/ const double MathConstant::INF = std::numeric_limits<double>::infinity();  // 1.#INF.
/*static*/ const double MathConstant::NaN = std::numeric_limits<double>::quiet_NaN();  // -1.#IND.
///*static*/ const double MathConstant::NaN = std::numeric_limits<double>::signaling_NaN();  // -1.#INF.

///*static*/ const double MathConstant::E = std::exp(1.0);
/*static*/ const double MathConstant::E = boost::math::constants::e<double>();

///*static*/ const double MathConstant::PI = std::atan(1.0) * 4.0;
/*static*/ const double MathConstant::PI = boost::math::constants::pi<double>();
/*static*/ const double MathConstant::_2_PI = MathConstant::PI * 2.0;
/*static*/ const double MathConstant::_4_PI = MathConstant::PI * 4.0;
/*static*/ const double MathConstant::PI_2 = MathConstant::PI / 2.0;
/*static*/ const double MathConstant::PI_3 = MathConstant::PI / 3.0;
/*static*/ const double MathConstant::PI_4 = MathConstant::PI / 4.0;
/*static*/ const double MathConstant::PI_6 = MathConstant::PI / 6.0;

/*static*/ const double MathConstant::TO_RAD = MathConstant::PI / 180.0;
/*static*/ const double MathConstant::TO_DEG = 180.0 / MathConstant::PI;

/*static*/ const double MathConstant::SIN_30 = std::sin(MathConstant::PI_6);
/*static*/ const double MathConstant::SIN_45 = std::sin(MathConstant::PI_4);
/*static*/ const double MathConstant::SIN_60 = std::sin(MathConstant::PI_3);
/*static*/ const double MathConstant::COS_30 = std::cos(MathConstant::PI_6);
/*static*/ const double MathConstant::COS_45 = std::cos(MathConstant::PI_4);
/*static*/ const double MathConstant::COS_60 = std::cos(MathConstant::PI_3);
/*static*/ const double MathConstant::TAN_30 = std::tan(MathConstant::PI_6);
/*static*/ const double MathConstant::TAN_45 = std::tan(MathConstant::PI_4);
/*static*/ const double MathConstant::TAN_60 = std::tan(MathConstant::PI_3);

/*static*/ const double MathConstant::SQRT_2 = std::sqrt(2.0);
/*static*/ const double MathConstant::SQRT_3 = std::sqrt(3.0);
/*static*/ const double MathConstant::SQRT_5 = std::sqrt(5.0);

/*static*/ const double MathConstant::TOL_1 = 1.0e-1;
/*static*/ const double MathConstant::TOL_2 = 1.0e-2;
/*static*/ const double MathConstant::TOL_3 = 1.0e-3;
/*static*/ const double MathConstant::TOL_4 = 1.0e-4;
/*static*/ const double MathConstant::TOL_5 = 1.0e-5;
/*static*/ const double MathConstant::TOL_6 = 1.0e-6;
/*static*/ const double MathConstant::TOL_7 = 1.0e-7;
/*static*/ const double MathConstant::TOL_8 = 1.0e-8;
/*static*/ const double MathConstant::TOL_9 = 1.0e-9;
/*static*/ const double MathConstant::TOL_10 = 1.0e-10;

///*static*/ double MathConstant::EPS = MathConstant::TOL_5;
/*static*/ double MathConstant::EPS = std::numeric_limits<double>::epsilon();  // 2.22045e-016.

/*static*/ int MathConstant::ITERATION_LIMIT = 1000;
		
}  // namespace swl
