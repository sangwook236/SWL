#include "swl/Config.h"
#include "swl/kinematics/DHParam.h"
#include "swl/base/LogException.h"
#include "swl/math/MathUtil.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class DHParam

DHParam & DHParam::operator=(const DHParam &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	d_ = rhs.d_;
	theta_ = rhs.theta_;
	a_ = rhs.a_;
	alpha_ = rhs.alpha_;
	return *this;
}

/*static*/ TMatrix3<double> DHParam::toDHMatrix(const DHParam &dhParam, const bool bIsCraig /*= true*/)
{
	TMatrix3<double> tmat;
	const double st = std::sin(dhParam.theta_), ct = std::cos(dhParam.theta_);
	const double sa = std::sin(dhParam.alpha_), ca = std::cos(dhParam.alpha_);

	if (bIsCraig)  // DH Notation is using in Craig's Book
	{
		// x
		tmat.X().x() = ct;
		tmat.X().y() = st * ca;
		tmat.X().z() = st * sa;

		// y
		tmat.Y().x() = -st;
		tmat.Y().y() = ct * ca;
		tmat.Y().z() = ct * sa;

		// z
		tmat.Z().x() = 0.0;
		tmat.Z().y() = -sa;
		tmat.Z().z() = ca;

		// p
		tmat.T().x() = dhParam.a_;
		tmat.T().y() = -dhParam.d_ * sa;
		tmat.T().z() = dhParam.d_ * ca;
	}
	else  // DH Notation is using in Paul, Fu, Asada's Books
	{
		// x
		tmat.X().x() = ct;
		tmat.X().y() = st;
		tmat.X().z() = 0.0;

		// y
		tmat.Y().x() = -st * ca;
		tmat.Y().y() = ct * ca;
		tmat.Y().z() = sa;

		// z
		tmat.Z().x() = st * sa;
		tmat.Z().y() = -ct * sa;
		tmat.Z().z() = ca;

		// p
		tmat.T().x() = dhParam.a_ * ct;
		tmat.T().y() = dhParam.a_ * st;
		tmat.T().z() = dhParam.d_;
	}
	return tmat;
}

/*static*/ DHParam DHParam::toDHParam(const TMatrix3<double> &tmat, const bool bIsCraig /*= true*/)
{
	if (bIsCraig)  // DH Notation is using in Craig's Book
	{
		const double a = tmat.T().x();
		const double d = std::sqrt(tmat.T().y()*tmat.T().y() + tmat.T().z()*tmat.T().z());
		//const double d = -std::sqrt(tmat.T().y()*tmat.T().y() + tmat.T().z()*tmat.T().z());

		double tmp = tmat.X().x();
		const double theta = MathUtil::isZero(tmp) ? (tmat.Y().x() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2) : std::atan2(tmat.Y().x(), tmp);

		tmp = tmat.Z().z();
		const double alpha = MathUtil::isZero(tmp) ? (tmat.Z().y() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2) : std::atan2(-tmat.Z().y(), tmp);

		return DHParam(d, theta, a, alpha);
	}
	else  // DH Notation is using in Paul, Fu, Asada's Books
	{
		const double a = std::sqrt(tmat.T().x()*tmat.T().x() + tmat.T().y()*tmat.T().y());
		const double d = tmat.T().z();

		double tmp = tmat.X().x();
		const double theta = MathUtil::isZero(tmp) ? (tmat.X().y() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2) : std::atan2(tmat.X().y(), tmp);

		tmp = tmat.Z().z();
		const double alpha = MathUtil::isZero(tmp) ? (tmat.Y().z() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2) : std::atan2(tmat.Y().z(), tmp);

		return DHParam(d, theta, a, alpha);
	}
}

}  // namespace swl
