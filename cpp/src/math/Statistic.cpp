#include "swl/Config.h"
#include "swl/math/Statistic.h"
#include <numeric>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(PI)  // For djgpp.
#	undef PI
#endif


namespace swl {

/*static*/ double Statistic::mean(const std::vector<double> &sample)
{
	if (sample.empty()) return 0.0;
	
	const size_t &num = sample.size();
	if (1 == num) return sample[0];

	return std::accumulate(std::begin(sample), std::end(sample), 0.0) / num;
}

/*static*/ double Statistic::standardDeviation(const std::vector<double> &sample, const double mean /*= 0.0*/)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return 0.0;

	//const double &moment = std::accumulate(std::begin(sample), std::end(sample), 0.0, [&](const double lhs, const double rhs) { return lhs + (rhs - mean) * (rhs - mean); });
	//return std::sqrt(moment / double(num - 1));
	double accum = 0.0;
	std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += (val - mean) * (val - mean); });
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { const double delta = val - mean; accum += delta * delta; });
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += std::pow(val - mean, 2); });

	return std::sqrt(accum / (double)num);
}

/*static*/ double Statistic::sampleStandardDeviation(const std::vector<double> &sample, const double mean /*= 0.0*/)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return 0.0;

	//const double &moment = std::accumulate(std::begin(sample), std::end(sample), 0.0, [&](const double lhs, const double rhs) { return lhs + (rhs - mean) * (rhs - mean); });
	//return std::sqrt(moment / double(num - 1));
	double accum = 0.0;
	std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += (val - mean) * (val - mean); });
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { const double delta = val - mean; accum += delta * delta; });
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += std::pow(val - mean, 2); });

	return std::sqrt(accum / double(num - 1));
}

/*static*/ double Statistic::skewness(const std::vector<double> &sample, const double mean /*= 0.0*/, const double sd /*= 1.0*/)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return 0.0;

	double accum = 0.0;
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { const double delta = val - mean; accum += delta * delta * delta; });
	std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += std::pow(val - mean, 3); });

	//return accum / ((double)num * sd * sd * sd);
	return accum / ((double)num * std::pow(sd, 3));
}

/*static*/ double Statistic::kurtosis(const std::vector<double> &sample, const double mean /*= 0.0*/, const double sd /*= 1.0*/)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return 0.0;

	double accum = 0.0;
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { const double delta = val - mean; accum += delta * delta * delta * delta; });
	std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += std::pow(val - mean, 4); });

	//return accum / ((double)num * sd * sd * sd * sd);
	return accum / ((double)num * std::pow(sd, 4));
}

/*static*/ double Statistic::sampleVariance(const Eigen::VectorXd &D)
// Sample variance.
{
	if (D.size() <= 1) return 0.0;

	// Centered data.
	const Eigen::VectorXd centered(D.array() - D.mean());
	return centered.dot(centered) / (D.size() - 1);
}

/*static*/ Eigen::VectorXd Statistic::sampleVariance(const Eigen::MatrixXd &D)
// Sample variances of each row.
// row : The dimension of data.
// col : The number of data.
{
	if (D.cols() <= 1) return Eigen::VectorXd::Zero(D.rows());

	// Centered data.
	const Eigen::MatrixXd centered(D.colwise() - D.rowwise().mean());
	return Eigen::VectorXd(centered.cwiseProduct(centered).rowwise().sum().array() / double(D.cols() - 1));
}

/*static*/ Eigen::MatrixXd Statistic::sampleCovarianceMatrix(const Eigen::MatrixXd &D)
// Sample covariance matrix.
// row : The dimension of data.
// col : The number of data.
{
	if (D.cols() <= 1) return Eigen::MatrixXd::Zero(D.rows(), D.rows());

	// Centered data.
	const Eigen::MatrixXd centered(D.colwise() - D.rowwise().mean());
	return (centered * centered.adjoint()) / double(D.cols() - 1);
}

}  //  namespace swl
