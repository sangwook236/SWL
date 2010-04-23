#if !defined(__SWL_MATH__ROTATION__H_)
#define __SWL_MATH__ROTATION__H_ 1


#include "swl/math/RMatrix.h"
#include "swl/math/MathUtil.h"
#include "swl/math/MathConstant.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// struct RotationOrder

struct SWL_MATH_API RotationOrder
{
public:
	/// the order of a rotation
	static unsigned int genOrder(const bool isFixed, const MathConstant::AXIS first, const MathConstant::AXIS second, const MathConstant::AXIS third);
	static bool isFixed(const unsigned int order)  {  return !(order & 0xF000);  }
	static bool parseOrder(const unsigned int order, MathConstant::AXIS &first, MathConstant::AXIS &second, MathConstant::AXIS &third);
};


//-----------------------------------------------------------------------------------------
// struct RotationAngle

struct SWL_MATH_API RotationAngle
{
public:
	RotationAngle(const double alpha = 0.0, const double beta = 0.0, const double gamma = 0.0)
	: alpha_(alpha), beta_(beta), gamma_(gamma)
	{}
	explicit RotationAngle(const double rhs[3])
	: alpha_(rhs[0]), beta_(rhs[1]), gamma_(rhs[2])
	{}
	RotationAngle(const RotationAngle &rhs)
	: alpha_(rhs.alpha_), beta_(rhs.beta_), gamma_(rhs.gamma_)
	{}
	~RotationAngle()  {}

	RotationAngle & operator=(const RotationAngle &rhs)
	{
	    if (this == &rhs) return *this;
	    alpha_ = rhs.alpha_;  beta_ = rhs.beta_;  gamma_ = rhs.gamma_;
	    return *this;
	}

public:
	/// calculate a rotational angle
	static RotationAngle calc(const unsigned int order, const RMatrix3<double> &mat);

	///
	static RotationAngle calcX(const RMatrix3<double> &mat);
	static RotationAngle calcY(const RMatrix3<double> &mat);
	static RotationAngle calcZ(const RMatrix3<double> &mat);

	/// fixed rotation
	static RotationAngle calcFixedXY(const RMatrix3<double> &mat)
	{
		RotationAngle angle;
		calcRelativeYX(mat, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedXZ(const RMatrix3<double> &mat)
	{
		RotationAngle angle;
		calcRelativeZX(mat, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedYZ(const RMatrix3<double> &mat)
	{
		RotationAngle angle;
		calcRelativeZY(mat, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedYX(const RMatrix3<double> &mat)
	{
		RotationAngle angle;
		calcRelativeXY(mat, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedZX(const RMatrix3<double> &mat)
	{
		RotationAngle angle;
		calcRelativeXZ(mat, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedZY(const RMatrix3<double> &mat)
	{
		RotationAngle angle;
		calcRelativeYZ(mat, angle.beta_, angle.alpha_);
		return angle;
	}

	static RotationAngle calcFixedXYZ(const RMatrix3<double> &mat)
	{
		RotationAngle angle;
		calcRelativeZYX(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedXZY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYZX(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedYZX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXZY(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedYXZ(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeZXY(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedZXY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYXZ(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedZYX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXYZ(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}

	static RotationAngle calcFixedXYX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXYX(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedXZX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXZX(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedYZY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYZY(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedYXY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYXY(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedZXZ(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeZXZ(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}
	static RotationAngle calcFixedZYZ(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeZYZ(mat, angle.gamma_, angle.beta_, angle.alpha_);
		return angle;
	}

	/// relative rotation: Euler angle
	static RotationAngle calcRelativeXY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXY(mat, angle.alpha_, angle.beta_);
		return angle;
	}
	static RotationAngle calcRelativeXZ(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXZ(mat, angle.alpha_, angle.beta_);
		return angle;
	}
	static RotationAngle calcRelativeYZ(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYZ(mat, angle.alpha_, angle.beta_);
		return angle;
	}
	static RotationAngle calcRelativeYX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYX(mat, angle.alpha_, angle.beta_);
		return angle;
	}
	static RotationAngle calcRelativeZX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeZX(mat, angle.alpha_, angle.beta_);
		return angle;
	}
	static RotationAngle calcRelativeZY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeZY(mat, angle.alpha_, angle.beta_);
		return angle;
	}

	static RotationAngle calcRelativeXYZ(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXYZ(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeXZY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXZY(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeYZX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYZX(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeYXZ(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYXZ(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeZXY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeZXY(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeZYX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeZYX(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}

	static RotationAngle calcRelativeXYX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXYX(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeXZX(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeXZX(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeYZY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYZY(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeYXY(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeYXY(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeZXZ(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeZXZ(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}
	static RotationAngle calcRelativeZYZ(const RMatrix3<double> &mat)
	{  
		RotationAngle angle;
		calcRelativeZYZ(mat, angle.alpha_, angle.beta_, angle.gamma_);
		return angle;
	}

public:
	/// accessor & mutator
	double & alpha()  {  return alpha_;  }
	double alpha() const  {  return alpha_;  }
	double & beta()  {  return beta_;  }
	double beta() const  {  return beta_;  }
	double & gamma()  {  return gamma_;  }
	double gamma() const  {  return gamma_;  }

	///
	bool isZero(const double tol = MathConstant::EPS) const
	{  return MathUtil::isZero(alpha_, tol) && MathUtil::isZero(beta_, tol) && MathUtil::isZero(gamma_, tol);  }
	bool isEqual(const RotationAngle &rhs, const double tol = MathConstant::EPS) const
	{  return (*this - rhs).isZero(tol);  }
	bool isBounded() const
	{  return isBounded(alpha_) && isBounded(beta_) && isBounded(gamma_);  }

	/// comparison operator
    bool operator==(const RotationAngle &rhs) const  {  return isEqual(rhs);  }
    bool operator!=(const RotationAngle &rhs) const  {  return !isEqual(rhs);  }

	///
	RotationAngle & operator+()  {  return *this;  }
	const RotationAngle & operator+() const  {  return *this;  }
	RotationAngle operator+(const RotationAngle &rhs) const
	{  return RotationAngle(alpha_+rhs.alpha_, beta_+rhs.beta_, gamma_+rhs.gamma_);  }
	RotationAngle & operator+=(const RotationAngle &rhs)
	{  alpha_ += rhs.alpha_;  beta_ += rhs.beta_;  gamma_ += rhs.gamma_;  return *this;  }
	RotationAngle operator-() const  {  return RotationAngle(-alpha_, -beta_, -gamma_);  }
	RotationAngle operator-(const RotationAngle &rhs) const
	{  return RotationAngle(alpha_-rhs.alpha_, beta_-rhs.beta_, gamma_-rhs.gamma_);  }
	RotationAngle & operator-=(const RotationAngle &rhs)
	{  alpha_ -= rhs.alpha_;  beta_ -= rhs.beta_;  gamma_ -= rhs.gamma_;  return *this;  }

	/// scalar operation
	RotationAngle operator*(const double S) const
	{  return RotationAngle(alpha_*S, beta_*S, gamma_*S);  }
	RotationAngle & operator*=(const double S)
	{  alpha_ *= S;  beta_ *= S;  gamma_ *= S;  return *this;  }
	RotationAngle operator/(const double S) const
	{
		if (MathUtil::isZero(S))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
	    return RotationAngle(alpha_/S, beta_/S, gamma_/S);
	}
	RotationAngle & operator/=(const double S)
	{
		if (MathUtil::isZero(S))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
	    alpha_ /= S;  beta_ /= S;  gamma_ /= S;
	    return *this;
	}

	///
	void bound()
	{
		bound(alpha_);
		bound(beta_);
		bound(gamma_);
	}

private:
	/// relative rotation: Euler angle
	static void calcRelativeXY(const RMatrix3<double> &mat, double &alpha, double &beta);
	static void calcRelativeXZ(const RMatrix3<double> &mat, double &alpha, double &beta);
	static void calcRelativeYZ(const RMatrix3<double> &mat, double &alpha, double &beta);
	static void calcRelativeYX(const RMatrix3<double> &mat, double &alpha, double &beta);
	static void calcRelativeZX(const RMatrix3<double> &mat, double &alpha, double &beta);
	static void calcRelativeZY(const RMatrix3<double> &mat, double &alpha, double &beta);

	static void calcRelativeXYZ(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeXZY(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeYZX(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeYXZ(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeZXY(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeZYX(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);

	static void calcRelativeXYX(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeXZX(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeYZY(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeYXY(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeZXZ(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);
	static void calcRelativeZYZ(const RMatrix3<double> &mat, double &alpha, double &beta, double &gamma);

private:
	///
	bool isBounded(const double angle) const
	{  return -MathConstant::PI < angle && angle <= MathConstant::PI;  }
	//{  return MathUtil::isBounded_oc(angle, -MathConstant::PI, MathConstant::PI);  }
	void bound(double & angle) const
	{
		if (angle <= -MathConstant::PI)
		{
			do
			{
				angle += MathConstant::_2_PI;
			} while (angle <= -MathConstant::PI);
		}
		else if (angle > MathConstant::PI)
		{
			do
			{
				angle -= MathConstant::_2_PI;
			} while (angle > MathConstant::PI);
		}
	}

private:
	/// rotational angle: (-pi, pi] or [-pi, pi]
	double alpha_, beta_, gamma_;
};


//-----------------------------------------------------------------------------------------
// struct Rotation

struct SWL_MATH_API Rotation
{
public:
	///
	static RMatrix3<double> rotateX(const double radian);
	static RMatrix3<double> rotateY(const double radian);
	static RMatrix3<double> rotateZ(const double radian);
	static RMatrix3<double> rotate(const double radian, const Vector3<double> &axis);

	/// calculate a rotation matrix
	static RMatrix3<double> rotate(const unsigned int order, const RotationAngle &angle);

	/// fixed rotation
	static RMatrix3<double> rotateFixedXYZ(const RotationAngle &angle)  {  return rotateRelativeZYX(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedXZY(const RotationAngle &angle)  {  return rotateRelativeYZX(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedYZX(const RotationAngle &angle)  {  return rotateRelativeXZY(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedYXZ(const RotationAngle &angle)  {  return rotateRelativeZXY(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedZXY(const RotationAngle &angle)  {  return rotateRelativeYXZ(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedZYX(const RotationAngle &angle)  {  return rotateRelativeXYZ(angle.gamma(), angle.beta(), angle.alpha());  }

	static RMatrix3<double> rotateFixedXYX(const RotationAngle &angle)  {  return rotateRelativeXYX(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedXZX(const RotationAngle &angle)  {  return rotateRelativeXZX(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedYZY(const RotationAngle &angle)  {  return rotateRelativeYZY(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedYXY(const RotationAngle &angle)  {  return rotateRelativeYXY(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedZXZ(const RotationAngle &angle)  {  return rotateRelativeZXZ(angle.gamma(), angle.beta(), angle.alpha());  }
	static RMatrix3<double> rotateFixedZYZ(const RotationAngle &angle)  {  return rotateRelativeZYZ(angle.gamma(), angle.beta(), angle.alpha());  }

	/// relative rotation: Euler angle
	static RMatrix3<double> rotateRelativeXYZ(const RotationAngle &angle)  {  return rotateRelativeXYZ(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeXZY(const RotationAngle &angle)  {  return rotateRelativeXZY(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeYZX(const RotationAngle &angle)  {  return rotateRelativeYZX(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeYXZ(const RotationAngle &angle)  {  return rotateRelativeYXZ(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeZXY(const RotationAngle &angle)  {  return rotateRelativeZXY(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeZYX(const RotationAngle &angle)  {  return rotateRelativeZYX(angle.alpha(), angle.beta(), angle.gamma());  }

	static RMatrix3<double> rotateRelativeXYX(const RotationAngle &angle)  {  return rotateRelativeXYX(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeXZX(const RotationAngle &angle)  {  return rotateRelativeXZX(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeYZY(const RotationAngle &angle)  {  return rotateRelativeYZY(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeYXY(const RotationAngle &angle)  {  return rotateRelativeYXY(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeZXZ(const RotationAngle &angle)  {  return rotateRelativeZXZ(angle.alpha(), angle.beta(), angle.gamma());  }
	static RMatrix3<double> rotateRelativeZYZ(const RotationAngle &angle)  {  return rotateRelativeZYZ(angle.alpha(), angle.beta(), angle.gamma());  }

private:
	/// relative rotation:Euler angle
	static RMatrix3<double> rotateRelativeXYZ(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeXZY(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeYZX(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeYXZ(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeZXY(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeZYX(const double alpha, const double beta, const double gamma);

	static RMatrix3<double> rotateRelativeXYX(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeXZX(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeYZY(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeYXY(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeZXZ(const double alpha, const double beta, const double gamma);
	static RMatrix3<double> rotateRelativeZYZ(const double alpha, const double beta, const double gamma);
};

}  // namespace swl


#endif  // __SWL_MATH__ROTATION__H_
