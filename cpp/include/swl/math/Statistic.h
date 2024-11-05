#if !defined(__SWL_MATH__STATISTIC__H_)
#define __SWL_MATH__STATISTIC__H_ 1


#include "swl/math/ExportMath.h"
#include <Eigen/Core>
#include <vector>


namespace swl {

//-----------------------------------------------------------------------------------------
// Statistic.

struct SWL_MATH_API Statistic
{
public:
	///
	static double mean(const std::vector<double> &sample);
	static double variance(const std::vector<double> &sample, const double mean);
	static double sampleVariance(const std::vector<double> &sample, const double mean);
	static double standardDeviation(const std::vector<double> &sample, const double mean);
	static double sampleStandardDeviation(const std::vector<double> &sample, const double mean);
	static double skewness(const std::vector<double> &sample, const double mean, const double sd);
	static double kurtosis(const std::vector<double> &sample, const double mean, const double sd);

	static double rms(const std::vector<double> &sample, const double mean);
	static double peak(const std::vector<double> &sample);
	static double crestFactor(const std::vector<double> &sample, const double mean);

	///
	static double sampleVariance(const Eigen::VectorXd &D);
	static Eigen::VectorXd sampleVariance(const Eigen::MatrixXd &D);
	static Eigen::MatrixXd sampleCovarianceMatrix(const Eigen::MatrixXd &D);

	///
	static double multivariateNormalDistibutionPdf(const Eigen::VectorXd &x, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov);
	static double multivariateNormalDistibutionLogPdf(const Eigen::VectorXd &x, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov);
};

}  // namespace swl


#endif  // __SWL_MATH__STATISTIC__H_
