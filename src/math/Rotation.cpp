#include "swl/math/Rotation.h"
#include "swl/base/LogException.h"
#include <cmath>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#define SLD_ROTATION_ORDER__USE_LEFT_TO_RIGHT 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct RotationOrder

unsigned int RotationOrder::genOrder(bool isFixed, MathConstant::AXIS first, MathConstant::AXIS second, MathConstant::AXIS third)
{
	unsigned int order = isFixed ? 0x0000 : 0x8000;
#if defined(SLD_ROTATION_ORDER__USE_LEFT_TO_RIGHT)
	if (first) order |= first << 8;
	if (second) order |= second << 4;
	if (third) order |= third;
#else
	if (first) order |= first;
	if (second) order |= second << 4;
	if (third) order |= third << 8;
#endif
	return order;
}

bool RotationOrder::parseOrder(unsigned int order, MathConstant::AXIS& first, MathConstant::AXIS& second, MathConstant::AXIS& third)
{
	// if order == 0, identity rotation
#if defined(SLD_ROTATION_ORDER__USE_LEFT_TO_RIGHT)
	first = (MathConstant::AXIS)((order & 0x0F00) >> 8);
	second = (MathConstant::AXIS)((order & 0x00F0) >> 4);
	third = (MathConstant::AXIS)(order & 0x000F);
#else
	first = (MathConstant::AXIS)(order & 0x000F);
	second = (MathConstant::AXIS)((order & 0x00F0) >> 4);
	third = (MathConstant::AXIS)((order & 0x0F00) >> 8);
#endif
	
	// check validity
	return !((first ^ MathConstant::AXIS_NULL) && (first ^ MathConstant::AXIS_X) && (first ^ MathConstant::AXIS_Y) && (first ^ MathConstant::AXIS_Z)) &&
		!((second ^ MathConstant::AXIS_NULL) && (second ^ MathConstant::AXIS_X) && (second ^ MathConstant::AXIS_Y) && (second ^ MathConstant::AXIS_Z)) &&
		!((third ^ MathConstant::AXIS_NULL) && (third ^ MathConstant::AXIS_X) && (third ^ MathConstant::AXIS_Y) && (third ^ MathConstant::AXIS_Z));
}



//-----------------------------------------------------------------------------------------
// struct RotationAngle

RotationAngle RotationAngle::calc(unsigned int order, const RMatrix3<double>& mat)
{
	// if order == 0, identity rotation
	if (!order) return RotationAngle();

	MathConstant::AXIS first, second, third;
	if (!RotationOrder::parseOrder(order, first, second, third))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return RotationAngle();
	}

	if (!first && !second && !third) return RotationAngle();

	bool isFixedRotation = RotationOrder::isFixed(order);
	if (isFixedRotation) std::swap(first, third);

	// simplify rotation info
	if (second)
	{
		if (!first)
		{
			first = second;
			second = MathConstant::AXIS_NULL;
		}
		else if (!(first ^ second))  // first == second
			second = MathConstant::AXIS_NULL;
	}
	if (third)
	{
		if (second)
		{
			if (!(second ^ third))  // second == third
				third = MathConstant::AXIS_NULL;
		}
		else
		{
			if (!first)
			{
				first = third;
				third = MathConstant::AXIS_NULL;
			}
			else
			{
				if (!(first ^ third))  // first == third
					third = MathConstant::AXIS_NULL;
				else
				{
					second = third;
					third = MathConstant::AXIS_NULL;
				}
			}
		}
	}

	//
	RotationAngle rotAngle;
	switch (first)
	{
	//
	case MathConstant::AXIS_NULL:
		throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
		//break;
	//
	case MathConstant::AXIS_X:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return RotationAngle::calcX(mat);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RotationAngle();
			}
			break;
		case MathConstant::AXIS_Y:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				if (isFixedRotation) RotationAngle::calcRelativeXY(mat, rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeXY(mat, rotAngle.alpha(), rotAngle.beta());
				return rotAngle;
			case MathConstant::AXIS_Z:
				if (isFixedRotation) RotationAngle::calcRelativeXYZ(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeXYZ(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			case MathConstant::AXIS_X: 
				if (isFixedRotation) RotationAngle::calcRelativeXYX(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeXYX(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RotationAngle();
			}
			break;
		case MathConstant::AXIS_Z:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				if (isFixedRotation) RotationAngle::calcRelativeXZ(mat, rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeXZ(mat, rotAngle.alpha(), rotAngle.beta());
				return rotAngle;
			case MathConstant::AXIS_X:
				if (isFixedRotation) RotationAngle::calcRelativeXZX(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeXZX(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			case MathConstant::AXIS_Y:
				if (isFixedRotation) RotationAngle::calcRelativeXZY(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeXZY(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RotationAngle();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
			//return RotationAngle();
		}
		break;
	//
	case MathConstant::AXIS_Y:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return RotationAngle::calcY(mat);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RotationAngle();
			}
			break;
		case MathConstant::AXIS_Z:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				if (isFixedRotation) RotationAngle::calcRelativeYZ(mat, rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeYZ(mat, rotAngle.alpha(), rotAngle.beta());
				return rotAngle;
			case MathConstant::AXIS_X:
				if (isFixedRotation) RotationAngle::calcRelativeYZX(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeYZX(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			case MathConstant::AXIS_Y:
				if (isFixedRotation) RotationAngle::calcRelativeYZY(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeYZY(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RotationAngle();
			}
			break;
		case MathConstant::AXIS_X:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				if (isFixedRotation) RotationAngle::calcRelativeYX(mat, rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeYX(mat, rotAngle.alpha(), rotAngle.beta());
				return rotAngle;
			case MathConstant::AXIS_Y:
				if (isFixedRotation) RotationAngle::calcRelativeYXY(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeYXY(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			case MathConstant::AXIS_Z:
				if (isFixedRotation) RotationAngle::calcRelativeYXZ(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeYXZ(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RotationAngle();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
			//return RotationAngle();
		}
		break;
	//
	case MathConstant::AXIS_Z:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return RotationAngle::calcZ(mat);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RotationAngle();
			}
			break;
		case MathConstant::AXIS_X:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				if (isFixedRotation) RotationAngle::calcRelativeZX(mat, rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeZX(mat, rotAngle.alpha(), rotAngle.beta());
				return rotAngle;
			case MathConstant::AXIS_Y:
				if (isFixedRotation) RotationAngle::calcRelativeZXY(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeZXY(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			case MathConstant::AXIS_Z:
				if (isFixedRotation) RotationAngle::calcRelativeZXZ(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeZXZ(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RotationAngle();
			}
			break;
		case MathConstant::AXIS_Y:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				if (isFixedRotation) RotationAngle::calcRelativeZY(mat, rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeZY(mat, rotAngle.alpha(), rotAngle.beta());
				return rotAngle;
			case MathConstant::AXIS_Z:
				if (isFixedRotation) RotationAngle::calcRelativeZYZ(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeZYZ(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			case MathConstant::AXIS_X:
				if (isFixedRotation) RotationAngle::calcRelativeZYX(mat, rotAngle.gamma(), rotAngle.beta(), rotAngle.alpha());
				else RotationAngle::calcRelativeZYX(mat, rotAngle.alpha(), rotAngle.beta(), rotAngle.gamma());
				return rotAngle;
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RotationAngle();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
			//return RotationAngle();
		}
		break;
	//
	default:
		throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
		//return RotationAngle();
	}

	return rotAngle;
}

RotationAngle RotationAngle::calcX(const RMatrix3<double>& mat)
{
	if (!MathUtil::isZero(mat.X().x() - 1.0) || !MathUtil::isZero(mat.X().y()) || !MathUtil::isZero(mat.X().z()) ||
		!MathUtil::isZero(mat.Y().x()) || !MathUtil::isZero(mat.Z().x()))
		return RotationAngle();

	if (MathUtil::isZero(mat.Y().y()))
		return RotationAngle(mat.Y().z() > 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2, 0.0, 0.0);
	else return RotationAngle(::atan2(mat.Y().z(), mat.Y().y()), 0.0, 0.0);
}

RotationAngle RotationAngle::calcY(const RMatrix3<double>& mat)
{
	if (!MathUtil::isZero(mat.Y().x()) || !MathUtil::isZero(mat.Y().y() - 1.0) || !MathUtil::isZero(mat.Y().z()) ||
		!MathUtil::isZero(mat.X().y()) || !MathUtil::isZero(mat.Z().y()))
		return RotationAngle();

	if (MathUtil::isZero(mat.X().x()))
		return RotationAngle(mat.Z().x() > 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2, 0.0, 0.0);
	else return RotationAngle(::atan2(mat.Z().x(), mat.X().x()), 0.0, 0.0);
}

RotationAngle RotationAngle::calcZ(const RMatrix3<double>& mat)
{
	if (!MathUtil::isZero(mat.Z().x()) || !MathUtil::isZero(mat.Z().y()) || !MathUtil::isZero(mat.Z().z() - 1.0) ||
		!MathUtil::isZero(mat.X().z()) || !MathUtil::isZero(mat.Y().z()))
		return RotationAngle();

	if (MathUtil::isZero(mat.X().x()))
		return RotationAngle(mat.X().y() > 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2, 0.0, 0.0);
	else return RotationAngle(::atan2(mat.X().y(), mat.X().x()), 0.0, 0.0);
}

void RotationAngle::calcRelativeXY(const RMatrix3<double>& mat, double& alpha, double& beta)
{
	if (!MathUtil::isZero(mat.Y().x()))
	{
		alpha = beta = 0.0;
		return;
	}

	// alpha
	if (MathUtil::isZero(mat.Y().y()))
		alpha = mat.Y().z() > 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
	else alpha = ::atan2(mat.Y().z(), mat.Y().y());

	// beta
	if (MathUtil::isZero(mat.X().x()))
		beta = mat.Z().x() > 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
	else beta = ::atan2(mat.Z().x(), mat.X().x());
}

void RotationAngle::calcRelativeXZ(const RMatrix3<double>& mat, double& alpha, double& beta)
{
	if (!MathUtil::isZero(mat.Z().x()))
	{
		alpha = beta = 0.0;
		return;
	}

	// alpha
	if (MathUtil::isZero(mat.Z().z()))
		alpha = mat.Z().y() > 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
	else alpha = ::atan2(-mat.Z().y(), mat.Z().z());

	// beta
	if (MathUtil::isZero(mat.X().x()))
		beta = mat.Y().x() > 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
	else beta = ::atan2(-mat.Y().x(), mat.X().x());
}

void RotationAngle::calcRelativeYZ(const RMatrix3<double>& mat, double& alpha, double& beta)
{
	if (!MathUtil::isZero(mat.Z().y()))
	{
		alpha = beta = 0.0;
		return;
	}

	// alpha
	if (MathUtil::isZero(mat.Z().z()))
		alpha = mat.Z().x() > 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
	else alpha = ::atan2(mat.Z().x(), mat.Z().z());

	// beta
	if (MathUtil::isZero(mat.Y().y()))
		beta = mat.X().y() > 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
	else beta = ::atan2(mat.X().y(), mat.Y().y());
}

void RotationAngle::calcRelativeYX(const RMatrix3<double>& mat, double& alpha, double& beta)
{
	if (!MathUtil::isZero(mat.X().y()))
	{
		alpha = beta = 0.0;
		return;
	}

	// alpha
	if (MathUtil::isZero(mat.X().x()))
		alpha = mat.X().z() > 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
	else alpha = ::atan2(-mat.X().z(), mat.X().x());

	// beta
	if (MathUtil::isZero(mat.Y().y()))
		beta = mat.Z().y() > 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
	else beta = ::atan2(-mat.Z().y(), mat.Y().y());
}

void RotationAngle::calcRelativeZX(const RMatrix3<double>& mat, double& alpha, double& beta)
{
	if (!MathUtil::isZero(mat.X().z()))
	{
		alpha = beta = 0.0;
		return;
	}

	// alpha
	if (MathUtil::isZero(mat.X().x()))
		alpha = mat.X().y() > 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
	else alpha = ::atan2(mat.X().y(), mat.X().x());

	// beta
	if (MathUtil::isZero(mat.Z().z()))
		beta = mat.Y().z() > 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
	else beta = ::atan2(mat.Y().z(), mat.Z().z());
}

void RotationAngle::calcRelativeZY(const RMatrix3<double>& mat, double& alpha, double& beta)
{
	if (!MathUtil::isZero(mat.Y().z()))
	{
		alpha = beta = 0.0;
		return;
	}

	// alpha
	if (MathUtil::isZero(mat.Y().y()))
		alpha = mat.Y().x() > 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
	else alpha = ::atan2(-mat.Y().x(), mat.Y().y());

	// beta
	if (MathUtil::isZero(mat.Z().z()))
		beta = mat.X().z() > 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
	else beta = ::atan2(-mat.X().z(), mat.Z().z());
}

void RotationAngle::calcRelativeXYZ(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double cb = sqrt(mat.X().x() * mat.X().x() + mat.Y().x() * mat.Y().x());
	//const double cb = -sqrt(mat.X().x() * mat.X().x() + mat.Y().x() * mat.Y().x());
	if (MathUtil::isZero(cb))  // degenerating case
	{
		beta = mat.Z().x() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;

		gamma = 0.0;
		if (MathUtil::isZero(mat.Y().y()))
			alpha = mat.Y().z() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Y().z(), mat.Y().y());
	}
	else
	{
		beta = atan2(mat.Z().x(), cb);

		if (MathUtil::isZero(mat.Z().z()))
			alpha = mat.Z().y() * cb >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else alpha = atan2(-mat.Z().y(), mat.Z().z());

		if (MathUtil::isZero(mat.X().x()))
			gamma = mat.Y().x() * cb >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else gamma = atan2(-mat.Y().x(), mat.X().x());
	}
}

void RotationAngle::calcRelativeXZY(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double cb = sqrt(mat.Y().y() * mat.Y().y() + mat.Y().z() * mat.Y().z());
	//const double cb = -sqrt(mat.Y().y() * mat.Y().y() + mat.Y().z() * mat.Y().z());
	if (MathUtil::isZero(cb))  // degenerating case
	{
		beta = mat.Y().x() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;

		gamma = 0.0;
		if (MathUtil::isZero(mat.Z().z()))
			alpha = mat.Z().y() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else alpha = atan2(-mat.Z().y(), mat.Z().z());
	}
	else
	{
		beta = atan2(-mat.Y().x(), cb);

		if (MathUtil::isZero(mat.Y().y()))
			alpha = mat.Y().z() * cb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Y().z(), mat.Y().y());

		if (MathUtil::isZero(mat.X().x()))
			gamma = mat.Z().x() * cb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else gamma = atan2(mat.Z().x(), mat.X().x());
	}
}

void RotationAngle::calcRelativeYZX(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double cb = sqrt(mat.X().x() * mat.X().x() + mat.X().z() * mat.X().z());
	//const double cb = -sqrt(mat.X().x() * mat.X().x() + mat.X().z() * mat.X().z());
	if (MathUtil::isZero(cb))  // degenerating case
	{
		beta = mat.X().y() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;

		gamma = 0.0;
		if (MathUtil::isZero(mat.Z().z()))
			alpha = mat.Z().x() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Z().x(), mat.Z().z());
	}
	else
	{
		beta = atan2(mat.X().y(), cb);

		if (MathUtil::isZero(mat.X().x()))
			alpha = mat.X().z() * cb >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else alpha = atan2(-mat.X().z(), mat.X().x());

		if (MathUtil::isZero(mat.Y().y()))
			gamma = mat.Z().y() * cb >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else gamma = atan2(-mat.Z().y(), mat.Y().y());
	}
}

void RotationAngle::calcRelativeYXZ(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double cb = sqrt(mat.Z().x() * mat.Z().x() + mat.Z().z() * mat.Z().z());
	//const double cb = -sqrt(mat.Z().x() * mat.Z().x() + mat.Z().z() * mat.Z().z());
	if (MathUtil::isZero(cb))  // degenerating case
	{
		beta = mat.Z().y() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;

		gamma = 0.0;
		if (MathUtil::isZero(mat.X().x()))
			alpha = mat.X().z() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else alpha = atan2(-mat.X().z(), mat.X().x());
	}
	else
	{
		beta = atan2(-mat.Z().y(), cb);

		if (MathUtil::isZero(mat.Z().z()))
			alpha = mat.Z().x() * cb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Z().x(), mat.Z().z());

		if (MathUtil::isZero(mat.Y().y()))
			gamma = mat.X().y() * cb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else gamma = atan2(mat.X().y(), mat.Y().y());
	}
}

void RotationAngle::calcRelativeZXY(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double cb = sqrt(mat.Y().x() * mat.Y().x() + mat.Y().y() * mat.Y().y());
	//const double cb = -sqrt(mat.Y().x() * mat.Y().x() + mat.Y().y() * mat.Y().y());
	if (MathUtil::isZero(cb))  // degenerating case
	{
		beta = mat.Y().z() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;

		gamma = 0.0;
		if (MathUtil::isZero(mat.X().x()))
			alpha = mat.X().y() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.X().y(), mat.X().x());
	}
	else
	{
		beta = atan2(mat.Y().z(), cb);

		if (MathUtil::isZero(mat.Y().y()))
			alpha = mat.Y().x() * cb >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else alpha = atan2(-mat.Y().x(), mat.Y().y());

		if (MathUtil::isZero(mat.Z().z()))
			gamma = mat.X().z() * cb >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else gamma = atan2(-mat.X().z(), mat.Z().z());
	}
}

void RotationAngle::calcRelativeZYX(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double cb = sqrt(mat.X().x() * mat.X().x() + mat.X().y() * mat.X().y());
	//const double cb = -sqrt(mat.X().x() * mat.X().x() + mat.X().y() * mat.X().y());
	if (MathUtil::isZero(cb))  // degenerating case
	{
		beta = mat.X().z() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;

		gamma = 0.0;
		if (MathUtil::isZero(mat.Y().y()))
			alpha = mat.Y().x() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else alpha = atan2(-mat.Y().x(), mat.Y().y());
	}
	else
	{
		beta = atan2(-mat.X().z(), cb);

		if (MathUtil::isZero(mat.X().x()))
			alpha = mat.X().y() * cb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.X().y(), mat.X().x());

		if (MathUtil::isZero(mat.Z().z()))
			gamma = mat.Y().z() * cb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else gamma = atan2(mat.Y().z(), mat.Z().z());
	}
}

void RotationAngle::calcRelativeXYX(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double sb = sqrt(mat.X().y() * mat.X().y() + mat.X().z() * mat.X().z());
	//const double sb = -sqrt(mat.X().y() * mat.X().y() + mat.X().z() * mat.X().z());
	if (MathUtil::isZero(sb))  // degenerating case
	{
		beta = mat.X().x() >= 0.0 ? 0 : MathConstant::PI;
		//beta = mat.X().x() >= 0.0 ? 0 : -MathConstant::PI;

		gamma = 0.0;
		if (MathUtil::isZero(mat.Y().y()))
			alpha = mat.Y().z() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Y().z(), mat.Y().y());
	}
	else
	{
		if (MathUtil::isZero(mat.X().x()))
			beta = sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else beta = atan2(sb, mat.X().x());

		if (MathUtil::isZero(mat.X().z()))
			alpha = mat.X().y() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.X().y(), -mat.X().z());

		if (MathUtil::isZero(mat.Z().x()))
			gamma = mat.Y().x() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else gamma = atan2(mat.Y().x(), mat.Z().x());
	}
}

void RotationAngle::calcRelativeXZX(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double sb = sqrt(mat.X().y() * mat.X().y() + mat.X().z() * mat.X().z());
	//const double sb = -sqrt(mat.X().y() * mat.X().y() + mat.X().z() * mat.X().z());
	if (MathUtil::isZero(sb))  // degenerating case
	{
		beta = mat.X().x() >= 0.0 ? 0 : MathConstant::PI;
		//beta = mat.X().x() >= 0.0 ? 0 : -MathConstant::PI;

		gamma = 0.0;
		if (MathUtil::isZero(mat.Z().z()))
			alpha = mat.Z().y() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else alpha = atan2(-mat.Z().y(), mat.Z().z());
	}
	else
	{
		if (MathUtil::isZero(mat.X().x()))
			beta = sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else beta = atan2(sb, mat.X().x());

		if (MathUtil::isZero(mat.X().y()))
			alpha = mat.X().z() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.X().z(), mat.X().y());

		if (MathUtil::isZero(mat.Y().x()))
			gamma = mat.Z().x() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else gamma = atan2(mat.Z().x(), -mat.Y().x());
	}
}

void RotationAngle::calcRelativeYZY(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double sb = sqrt(mat.Y().x() * mat.Y().x() + mat.Y().z() * mat.Y().z());
	//const double sb = -sqrt(mat.Y().x() * mat.Y().x() + mat.Y().z() * mat.Y().z());
	if (MathUtil::isZero(sb))  // degenerating case
	{
		beta = mat.Y().y() >= 0.0 ? 0 : MathConstant::PI;
		//beta = mat.Y().y() >= 0.0 ? 0 : -MathConstant::PI;

		gamma = 0.0;
		if (MathUtil::isZero(mat.Z().z()))
			alpha = mat.Z().x() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Z().x(), mat.Z().z());
	}
	else
	{
		if (MathUtil::isZero(mat.Y().y()))
			beta = sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else beta = atan2(sb, mat.Y().y());

		if (MathUtil::isZero(mat.Y().x()))
			alpha = mat.Y().z() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Y().z(), -mat.Y().x());

		if (MathUtil::isZero(mat.X().y()))
			gamma = mat.Z().y() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else gamma = atan2(mat.Z().y(), mat.X().y());
	}
}

void RotationAngle::calcRelativeYXY(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double sb = sqrt(mat.Y().x() * mat.Y().x() + mat.Y().z() * mat.Y().z());
	//const double sb = -sqrt(mat.Y().x() * mat.Y().x() + mat.Y().z() * mat.Y().z());
	if (MathUtil::isZero(sb))  // degenerating case
	{
		beta = mat.Y().y() >= 0.0 ? 0 : MathConstant::PI;
		//beta = mat.Y().y() >= 0.0 ? 0 : -MathConstant::PI;

		gamma = 0.0;
		if (MathUtil::isZero(mat.X().x()))
			alpha = mat.X().z() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else alpha = atan2(-mat.X().z(), mat.X().x());
	}
	else
	{
		if (MathUtil::isZero(mat.Y().y()))
			beta = sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else beta = atan2(sb, mat.Y().y());

		if (MathUtil::isZero(mat.Y().z()))
			alpha = mat.Y().x() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Y().x(), mat.Y().z());

		if (MathUtil::isZero(mat.Z().y()))
			gamma = mat.X().y() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else gamma = atan2(mat.X().y(), -mat.Z().y());
	}
}

void RotationAngle::calcRelativeZXZ(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double sb = sqrt(mat.Z().x() * mat.Z().x() + mat.Z().y() * mat.Z().y());
	//const double sb = -sqrt(mat.Z().x() * mat.Z().x() + mat.Z().y() * mat.Z().y());
	if (MathUtil::isZero(sb))  // degenerating case
	{
		beta = mat.Z().z() >= 0.0 ? 0 : MathConstant::PI;
		//beta = mat.Z().z() >= 0.0 ? 0 : -MathConstant::PI;

		gamma = 0.0;
		if (MathUtil::isZero(mat.X().x()))
			alpha = mat.X().y() >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.X().y(), mat.X().x());
	}
	else
	{
		if (MathUtil::isZero(mat.Z().z()))
			beta = sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else beta = atan2(sb, mat.Z().z());

		if (MathUtil::isZero(mat.Z().y()))
			alpha = mat.Z().x() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Z().x(), -mat.Z().y());

		if (MathUtil::isZero(mat.Y().z()))
			gamma = mat.X().z() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else gamma = atan2(mat.X().z(), mat.Y().z());
	}
}

void RotationAngle::calcRelativeZYZ(const RMatrix3<double>& mat, double& alpha, double& beta, double& gamma)
{
	const double sb = sqrt(mat.Z().x() * mat.Z().x() + mat.Z().y() * mat.Z().y());
	//const double sb = -sqrt(mat.Z().x() * mat.Z().x() + mat.Z().y() * mat.Z().y());
	if (MathUtil::isZero(sb))  // degenerating case
	{
		beta = mat.Z().z() >= 0.0 ? 0 : MathConstant::PI;
		//beta = mat.Z().z() >= 0.0 ? 0 : -MathConstant::PI;

		gamma = 0.0;
		if (MathUtil::isZero(mat.Y().y()))
			alpha = mat.Y().x() >= 0.0 ? -MathConstant::PI_2 : MathConstant::PI_2;
		else alpha = atan2(-mat.Y().x(), mat.Y().y());
	}
	else
	{
		if (MathUtil::isZero(mat.Z().z()))
			beta = sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else beta = atan2(sb, mat.Z().z());

		if (MathUtil::isZero(mat.Z().x()))
			alpha = mat.Z().y() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else alpha = atan2(mat.Z().y(), mat.Z().x());

		if (MathUtil::isZero(mat.X().z()))
			gamma = mat.Y().z() * sb >= 0.0 ? MathConstant::PI_2 : -MathConstant::PI_2;
		else gamma = atan2(mat.Y().z(), -mat.X().z());
	}
}


//-----------------------------------------------------------------------------------------
//  struct Rotation

RMatrix3<double> Rotation::rotateX(double radian)
{
	const double s = sin(radian), c = cos(radian);

	RMatrix3<double> m;
	m.X().x() = 1.0;
	m.X().y() = 0.0;
	m.X().z() = 0.0;

	m.Y().x() = 0.0;
	m.Y().y() = c;
	m.Y().z() = s;

	m.Z().x() = 0.0;
	m.Z().y() = -s;
	m.Z().z() = c;

	//if (!m.isValid()) m.orthonormalize();
	return m;
}

RMatrix3<double> Rotation::rotateY(double radian)
{
	const double s = sin(radian), c = cos(radian);

	RMatrix3<double> m;
	m.X().x() = c;
	m.X().y() = 0.0;
	m.X().z() = -s;

	m.Y().x() = 0.0;
	m.Y().y() = 1.0;
	m.Y().z() = 0.0;

	m.Z().x() = s;
	m.Z().y() = 0.0;
	m.Z().z() = c;

	//if (!m.isValid()) m.orthonormalize();
	return m;
}

RMatrix3<double> Rotation::rotateZ(double radian)
{
	const double s = sin(radian), c = cos(radian);

	RMatrix3<double> m;
	m.X().x() = c;
	m.X().y() = s;
	m.X().z() = 0.0;

	m.Y().x() = -s;
	m.Y().y() = c;
	m.Y().z() = 0.0;

	m.Z().x() = 0.0;
	m.Z().y() = 0.0;
	m.Z().z() = 1.0;

	//if (!m.isValid()) m.orthonormalize();
	return m;
}

RMatrix3<double> Rotation::rotate(double rad, const TVector3<double>& axis)
{
	// for making a unit vector
	const double norm = sqrt(axis.x()*axis.x() + axis.y()*axis.y() + axis.z()*axis.z());
	if (MathUtil::isZero(norm))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return RMatrix3<double>();
	}

	RMatrix3<double> m;
	const double x(axis.x() / norm), y(axis.y() / norm), z(axis.z() / norm);
	const double s = sin(rad);
	const double c = cos(rad);
	const double v = 1.0 - c;

	m.X().x() = x * x * v + c;
	m.X().y() = x * y * v + z * s;
	m.X().z() = x * z * v - y * s;

	m.Y().x() = x * y * v - z * s;
	m.Y().y() = y * y * v + c;
	m.Y().z() = y * z * v + x * s;

	m.Z().x() = x * z * v + y * s;
	m.Z().y() = y * z * v - x * s;
	m.Z().z() = z * z * v + c;

	//if (!m.isValid()) m.orthonormalize();
	return m;
}

RMatrix3<double> Rotation::rotate(unsigned int order, const RotationAngle& angle)
{
	// if order == 0, identity rotation
	if (!order) return RMatrix3<double>();

	MathConstant::AXIS first, second, third;
	if (!RotationOrder::parseOrder(order, first, second, third))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return RMatrix3<double>();
	}

	double alpha(angle.alpha()), beta(angle.beta()), gamma(angle.gamma());
	if (!first && !MathUtil::isZero(alpha))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//alpha = 0.0;
	}
	if (!second && !MathUtil::isZero(beta))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//beta = 0.0;
	}
	if (!third && !MathUtil::isZero(gamma))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//gamma = 0.0;
	}

	//
	if ((!first && !second && !third) || angle.isZero())
		return RMatrix3<double>();

	if (RotationOrder::isFixed(order))
	{
		std::swap(alpha, gamma);
		std::swap(first, third);
	}
/*
	// simplify rotation info
	if (second)
	{
		if (!(first ^ second))  // first == second
		{
			first = MathConstant::AXIS_NULL;
			beta += alpha;
			alpha = 0.0;
		}
		if (!(second ^ third))  // second == third
		{
			third = MathConstant::AXIS_NULL;
			beta += gamma;
			gamma = 0.0;
		}
	}
	else
	{
		if (third && !(first ^ third))  // first == third
		{
			first = MathConstant::AXIS_NULL;
			gamma += alpha;
			alpha = 0.0;
		}
	}

	//
	switch (first)
	{
	//
	case MathConstant::AXIS_NULL:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			case MathConstant::AXIS_X:
				return Rotation::rotateX(gamma);
			case MathConstant::AXIS_Y:
				return Rotation::rotateY(gamma);
			case MathConstant::AXIS_Z:
				return Rotation::rotateZ(gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_X:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateX(beta);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeZXY(0.0, beta, gamma);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeYXZ(0.0, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Y:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateY(beta);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeXYZ(0.0, beta, gamma);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeZYX(0.0, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Z:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateZ(beta);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeYZX(0.0, beta, gamma);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeXZY(0.0, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
			//return RMatrix3<double>();
		}
		break;
	//
	case MathConstant::AXIS_X:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateX(alpha);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeXZY(alpha, 0.0, gamma);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeXYZ(alpha, 0.0, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Y:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeXYZ(alpha, beta, 0.0);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeXYZ(alpha, beta, gamma);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeXYX(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Z:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeXZY(alpha, beta, 0.0);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeXZX(alpha, beta, gamma);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeXZY(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
			//return RMatrix3<double>();
		}
		break;
	//
	case MathConstant::AXIS_Y:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateY(alpha);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeYXZ(alpha, 0.0, gamma);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeYZX(alpha, 0.0, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Z:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeYZX(alpha, beta, 0.0);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeYZX(alpha, beta, gamma);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeYZY(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_X:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeYXZ(alpha, beta, 0.0);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeYXY(alpha, beta, gamma);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeYXZ(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
			//return RMatrix3<double>();
		}
		break;
	//
	case MathConstant::AXIS_Z:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateZ(alpha);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeZYX(alpha, 0.0, gamma);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeZXY(alpha, 0.0, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_X:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeZXY(alpha, beta, 0.0);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeZXY(alpha, beta, gamma);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeZXZ(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Y:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeZYX(alpha, beta, 0.0);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeZYZ(alpha, beta, gamma);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeZYX(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
			//return RMatrix3<double>();
		}
		break;
	//
	default:
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return RMatrix3<double>();
	}
*/
	// simplify rotation info
	if (second && !MathUtil::isZero(beta))
	{
		if (!first || MathUtil::isZero(alpha))
		{
			first = second;
			second = MathConstant::AXIS_NULL;
			alpha = beta;
			beta = 0.0;
		}
		else if (!(first ^ second))  // first == second
		{
			second = MathConstant::AXIS_NULL;
			alpha += beta;
			beta = 0.0;
		}
	}
	if (third && !MathUtil::isZero(gamma))
	{
		if (second && !MathUtil::isZero(beta))
		{
			if (!(second ^ third))  // second == third
			{
				third = MathConstant::AXIS_NULL;
				beta += gamma;
				gamma = 0.0;
			}
		}
		else
		{
			if (!first || MathUtil::isZero(alpha))
			{
				first = third;
				third = MathConstant::AXIS_NULL;
				alpha = gamma;
				gamma = 0.0;
			}
			else
			{
				if (!(first ^ third))  // first == third
				{
					third = MathConstant::AXIS_NULL;
					alpha += gamma;
					gamma = 0.0;
				}
				else
				{
					second = third;
					third = MathConstant::AXIS_NULL;
					beta = gamma;
					gamma = 0.0;
				}
			}
		}
	}

	//
	switch (first)
	{
	//
	case MathConstant::AXIS_NULL:
		throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
		//break;
	//
	case MathConstant::AXIS_X:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateX(alpha);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Y:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeXYZ(alpha, beta, 0.0);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeXYZ(alpha, beta, gamma);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeXYX(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Z:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeXZY(alpha, beta, 0.0);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeXZX(alpha, beta, gamma);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeXZY(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
			//return RMatrix3<double>();
		}
		break;
	//
	case MathConstant::AXIS_Y:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateY(alpha);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Z:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeYZX(alpha, beta, 0.0);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeYZX(alpha, beta, gamma);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeYZY(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_X:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeYXZ(alpha, beta, 0.0);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeYXY(alpha, beta, gamma);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeYXZ(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
			//return RMatrix3<double>();
		}
		break;
	//
	case MathConstant::AXIS_Z:
		switch (second)
		{
		case MathConstant::AXIS_NULL:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateZ(alpha);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_X:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeZXY(alpha, beta, 0.0);
			case MathConstant::AXIS_Y:
				return Rotation::rotateRelativeZXY(alpha, beta, gamma);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeZXZ(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		case MathConstant::AXIS_Y:
			switch (third)
			{
			case MathConstant::AXIS_NULL:
				return Rotation::rotateRelativeZYX(alpha, beta, 0.0);
			case MathConstant::AXIS_Z:
				return Rotation::rotateRelativeZYZ(alpha, beta, gamma);
			case MathConstant::AXIS_X:
				return Rotation::rotateRelativeZYX(alpha, beta, gamma);
			default:
				throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
				//return RMatrix3<double>();
			}
			break;
		default:
			throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
			//return RMatrix3<double>();
		}
		break;
	//
	default:
		throw LogException(LogException::L_ERROR, "unexpected result", __FILE__, __LINE__, __FUNCTION__);
		//return RMatrix3<double>();
	}

	return RMatrix3<double>();
}

RMatrix3<double> Rotation::rotateRelativeXYZ(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = cb * cr;
	m.X().y() = sa * sb * cr + ca * sr;
	m.X().z() = -ca * sb * cr + sa * sr;

	m.Y().x() = -cb * sr;
	m.Y().y() = -sa * sb * sr + ca * cr;
	m.Y().z() = ca * sb * sr + sa * cr;

	m.Z().x() = sb;
	m.Z().y() = -sa * cb;
	m.Z().z() = ca * cb;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeXZY(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = cb * cr;
	m.X().y() = ca * sb * cr + sa * sr;
	m.X().z() = sa * sb * cr - ca * sr;

	m.Y().x() = -sb;
	m.Y().y() = ca * cb;
	m.Y().z() = sa * cb;

	m.Z().x() = cb * sr;
	m.Z().y() = ca * sb * sr - sa * cr;
	m.Z().z() = sa * sb * sr + ca * cr;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeYZX(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = ca * cb;
	m.X().y() = sb;
	m.X().z() = -sa * cb;

	m.Y().x() = -ca * sb * cr + sa * sr;
	m.Y().y() = cb * cr;
	m.Y().z() = sa * sb * cr + ca * sr;

	m.Z().x() = ca * sb * sr + sa * cr;
	m.Z().y() = -cb * sr;
	m.Z().z() = -sa * sb * sr + ca * cr;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeYXZ(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = sa * sb * sr + ca * cr;
	m.X().y() = cb * sr;
	m.X().z() = ca * sb * sr - sa * cr;

	m.Y().x() = sa * sb * cr - ca * sr;
	m.Y().y() = cb * cr;
	m.Y().z() = ca * sb * cr + sa * sr;

	m.Z().x() = sa * cb;
	m.Z().y() = -sb;
	m.Z().z() = ca * cb;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeZXY(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = -sa * sb * sr + ca * cr;
	m.X().y() = ca * sb * sr + sa * cr;
	m.X().z() = -cb * sr;

	m.Y().x() = -sa * cb;
	m.Y().y() = ca * cb;
	m.Y().z() = sb;

	m.Z().x() = sa * sb * cr + ca * sr;
	m.Z().y() = -ca * sb * cr + sa * sr;
	m.Z().z() = cb * cr;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeZYX(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = ca * cb;
	m.X().y() = sa * cb;
	m.X().z() = -sb;

	m.Y().x() = ca * sb * sr - sa * cr;
	m.Y().y() = sa * sb * sr + ca * cr;
	m.Y().z() = cb * sr;

	m.Z().x() = ca * sb * cr + sa * sr;
	m.Z().y() = sa * sb * cr - ca * sr;
	m.Z().z() = cb * cr;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeXYX(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = cb;
	m.X().y() = sa * sb;
	m.X().z() = -ca * sb;

	m.Y().x() = sb * sr;
	m.Y().y() = -sa * cb * sr + ca * cr;
	m.Y().z() = ca * cb * sr + sa * cr;

	m.Z().x() = sb * cr;
	m.Z().y() = -sa * cb * cr - ca * sr;
	m.Z().z() = ca * cb * cr - sa * sr;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeXZX(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = cb;
	m.X().y() = ca * sb;
	m.X().z() = sa * sb;

	m.Y().x() = -sb * cr;
	m.Y().y() = ca * cb * cr - sa * sr;
	m.Y().z() = sa * cb * cr + ca * sr;

	m.Z().x() = sb * sr;
	m.Z().y() = -ca * cb * sr - sa * cr;
	m.Z().z() = -sa * cb * sr + ca * cr;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeYZY(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = ca * cb * cr - sa * sr;
	m.X().y() = sb * cr;
	m.X().z() = -sa * cb * cr - ca * sr;

	m.Y().x() = -ca * sb;
	m.Y().y() = cb;
	m.Y().z() = sa * sb;

	m.Z().x() = ca * cb * sr + sa * cr;
	m.Z().y() = sb * sr;
	m.Z().z() = -sa * cb * sr + ca * cr;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeYXY(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = -sa * cb * sr + ca * cr;
	m.X().y() = sb * sr;
	m.X().z() = -ca * cb * sr - sa * cr;

	m.Y().x() = sa * sb;
	m.Y().y() = cb;
	m.Y().z() = ca * sb;

	m.Z().x() = sa * cb * cr + ca * sr;
	m.Z().y() = -sb * cr;
	m.Z().z() = ca * cb * cr - sa * sr;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeZXZ(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = -sa * cb * sr + ca * cr;
	m.X().y() = ca * cb * sr + sa * cr;
	m.X().z() = sb * sr;

	m.Y().x() = -sa * cb * cr - ca * sr;
	m.Y().y() = ca * cb * cr - sa * sr;
	m.Y().z() = sb * cr;

	m.Z().x() = sa * sb;
	m.Z().y() = -ca * sb;
	m.Z().z() = cb;

	return m;
}

RMatrix3<double> Rotation::rotateRelativeZYZ(double alpha, double beta, double gamma)
{
	const double sa = sin(alpha), ca = cos(alpha);
	const double sb = sin(beta), cb = cos(beta);
	const double sr = sin(gamma), cr = cos(gamma);

	RMatrix3<double> m;
	m.X().x() = ca * cb * cr - sa * sr;
	m.X().y() = sa * cb * cr + ca * sr;
	m.X().z() = -sb * cr;

	m.Y().x() = -ca * cb * sr - sa * cr;
	m.Y().y() = -sa * cb * sr + ca * cr;
	m.Y().z() = sb * sr;

	m.Z().x() = ca * sb;
	m.Z().y() = sa * sb;
	m.Z().z() = cb;

	return m;
}

}  // namespace swl
