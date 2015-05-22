#if !defined(__SWL_MATH__STATISTIC__H_)
#define __SWL_MATH__STATISTIC__H_ 1


#include "swl/math/ExportMath.h"
#include <Eigen/Core>


namespace swl {

//-----------------------------------------------------------------------------------------
// struct Statistic

struct SWL_MATH_API Statistic
{
public:
	///
	static double sampleVariance(const Eigen::VectorXd &D);
	static Eigen::VectorXd sampleVariance(const Eigen::MatrixXd &D);
	static Eigen::MatrixXd sampleCovarianceMatrix(const Eigen::MatrixXd &D);
};

}  // namespace swl


#endif  // __SWL_MATH__STATISTIC__H_
