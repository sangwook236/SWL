#if !defined(__SWL_MATH__DATA_NORMALIZATION__H_)
#define __SWL_MATH__DATA_NORMALIZATION__H_ 1


#include "swl/math/MathConstant.h"
#include <Eigen/Core>


namespace swl {

//-----------------------------------------------------------------------------------------
// Data Normalization.

struct SWL_MATH_API DataNormalization
{
public:
	///
	static bool normalizeDataByCentering(Eigen::MatrixXd &D);
	static bool normalizeDataByAverageDistance(Eigen::MatrixXd &D, const double averageDistance, const double tol = MathConstant::EPS);
	static bool normalizeDataByRange(Eigen::MatrixXd &D, const double minBound, const double maxBound, const double tol = MathConstant::EPS);
	static bool normalizeDataByLinearTransformation(Eigen::MatrixXd &D, const Eigen::MatrixXd &T);
	static bool normalizeDataByHomogeneousTransformation(Eigen::MatrixXd &D, const Eigen::MatrixXd &H);
	static bool normalizeDataByZScore(Eigen::MatrixXd &D, const double sigma);
	static bool normalizeDataByTStatistic(Eigen::MatrixXd &D, const double tol = MathConstant::EPS);

};

}  // namespace swl


#endif  // __SWL_MATH__DATA_NORMALIZATION__H_
