#if !defined(__SWL_KINEMATICS__DENAVIT_HARTENBERG_PARAMETER__H_)
#define __SWL_KINEMATICS__DENAVIT_HARTENBERG_PARAMETER__H_ 1


#include "swl/kinematics/JointParam.h"
#include "swl/math/TMatrix.h"


namespace swl {

//--------------------------------------------------------------------------------
// class DHParam: Denavit-Hartenberg Parameter

class SWL_KINEMATICS_API DHParam
{
public:
	//typedef DHParam base_type;

public:
	///
	DHParam(const double d, const double theta, const double a, const double alpha)
	: //base_type(),
	  d_(d), theta_(theta), a_(a), alpha_(alpha)
	{}
	DHParam(const DHParam &rhs)
	: //base_type(rhs),
	  d_(rhs.d_), theta_(rhs.theta_), a_(rhs.a_), alpha_(rhs.alpha_)
	{}
	virtual ~DHParam()  {}

	DHParam & operator=(const DHParam &rhs);

public:
	/// accessor & mutator
	void setDH(const double d, const double theta, const double a, const double alpha)
	{  d_ = d;  theta_ = theta;  a_ = a;  alpha_ = alpha;  }

	double getD() const  {  return d_;  }
	double getTheta() const  {  return theta_;  }
	double getA() const  {  return a_;  }
	double getAlpha() const  {  return alpha_;  }

	///
	static TMatrix3<double> toDHMatrix(const DHParam &dhParam, const bool bIsCraig = true);
	static DHParam toDHParam(const TMatrix3<double> &tmat, const bool bIsCraig = true);

protected:
	/// Denavit-Hartenberg parameters: d, theta, a & alpha mean values of each row of DH table
	///  -. if Craig's notation is applied, d = d(i), theta = theta(i), a = a(i-1) & alpha = alpha(i-1)
	///  -. if Paul's notation is applied, d = d(i), theta = theta(i), a = a(i) & alpha = alpha(i)
	double d_, theta_, a_, alpha_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__DENAVIT_HARTENBERG_PARAMETER__H_
