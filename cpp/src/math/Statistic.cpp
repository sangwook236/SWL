#include "swl/Config.h"
#include "swl/math/Statistic.h"
#include "swl/math/MathConstant.h"
#include <Eigen/Cholesky>
#include <algorithm>
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

	return std::accumulate(std::begin(sample), std::end(sample), 0.0) / double(num);
}

/*static*/ double Statistic::variance(const std::vector<double> &sample, const double mean)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return 0.0;

#if 0
	const double &moment = std::accumulate(std::begin(sample), std::end(sample), 0.0, [&](const double lhs, const double rhs) { return lhs + (rhs - mean) * (rhs - mean); });
	return std::sqrt(moment / double(num));
#else
	double accum = 0.0;
	std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += (val - mean) * (val - mean); });
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { const double delta = val - mean; accum += delta * delta; });
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += std::pow(val - mean, 2); });

	return accum / double(num);
#endif
}

/*static*/ double Statistic::sampleVariance(const std::vector<double> &sample, const double mean)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return 0.0;

#if 0
	const double &moment = std::accumulate(std::begin(sample), std::end(sample), 0.0, [&](const double lhs, const double rhs) { return lhs + (rhs - mean) * (rhs - mean); });
	return std::sqrt(moment / double(num - 1));
#else
	double accum = 0.0;
	std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += (val - mean) * (val - mean); });
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { const double delta = val - mean; accum += delta * delta; });
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += std::pow(val - mean, 2); });

	return accum / double(num - 1);
#endif
}

/*static*/ double Statistic::standardDeviation(const std::vector<double> &sample, const double mean)
{
	return std::sqrt(Statistic::variance(sample, mean));
}

/*static*/ double Statistic::sampleStandardDeviation(const std::vector<double> &sample, const double mean)
{
	return std::sqrt(Statistic::sampleVariance(sample, mean));
}

/*static*/ double Statistic::skewness(const std::vector<double> &sample, const double mean, const double sd)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return 0.0;

	double accum = 0.0;
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { const double delta = val - mean; accum += delta * delta * delta; });
	std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += std::pow(val - mean, 3); });

	//return accum / (double(num) * sd * sd * sd);
	return accum / (double(num) * std::pow(sd, 3));
}

/*static*/ double Statistic::kurtosis(const std::vector<double> &sample, const double mean, const double sd)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return 0.0;

	double accum = 0.0;
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { const double delta = val - mean; accum += delta * delta * delta * delta; });
	std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += std::pow(val - mean, 4); });

	//return accum / (double(num) * sd * sd * sd * sd);
	return accum / (double(num) * std::pow(sd, 4));
}

/*static*/ double Statistic::rms(const std::vector<double> &sample, const double mean)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return std::abs(sample[0] - mean);

#if 0
	const double &sum = std::accumulate(std::begin(sample), std::end(sample), 0.0, [&](const double lhs, const double rhs) { return lhs + (rhs - mean) * (rhs - mean); });
	return std::sqrt(sum / double(num));
#else
	double accum = 0.0;
	std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += (val - mean) * (val - mean); });
	//std::for_each(std::begin(sample), std::end(sample), [&](const double val) { accum += std::pow(val - mean, 2); });

	return std::sqrt(accum / double(num));
#endif
}

/*static*/ double Statistic::peak(const std::vector<double> &sample)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return std::abs(sample[0]);

	//return *std::max_element(sample.begin(), sample.end(), [](const double lhs, const double rhs) -> bool { return std::abs(lhs) < std::abs(rhs); });
	return std::abs(*std::max_element(sample.begin(), sample.end(), [](const double lhs, const double rhs) -> bool { return std::abs(lhs) < std::abs(rhs); }));
}

/*static*/ double Statistic::crestFactor(const std::vector<double> &sample, const double mean)
{
	if (sample.empty()) return 0.0;

	const size_t &num = sample.size();
	if (1 == num) return 1.0;
	//if (1 == num) return std::abs(sample[0]) / (sample[0] - mean);

	const double &rms = Statistic::rms(sample, mean);
	return rms > std::numeric_limits<double>::epsilon() ? Statistic::peak(sample) / rms : std::numeric_limits<double>::infinity();
}

/*static*/ double Statistic::sampleVariance(const Eigen::VectorXd &D)
// Sample variance.
{
	if (D.size() <= 1) return 0.0;

	// Centered data.
	const Eigen::VectorXd centered(D.array() - D.mean());
	return centered.dot(centered) / double(D.size() - 1);
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

/*static*/ double Statistic::multivariateNormalDistibutionPdf(const Eigen::VectorXd &x, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov)
// Compute p(x) at the point x using mean vector and variance-covariance matrix.
{
	if (x.size() != mean.size() || x.size() != cov.rows() || cov.rows() != cov.cols())
		throw std::runtime_error("Invalid vector or matrix size");

	Eigen::LLT<Eigen::MatrixXd> chol(cov);
	if (Eigen::Success != chol.info())
		throw std::runtime_error("Cholesky decompoistion failed");

	const Eigen::LLT<Eigen::MatrixXd>::Traits::MatrixL& L = chol.matrixL();
	const double numer = std::exp(-0.5 * L.solve(x - mean).squaredNorm());
	const double denom = std::pow(MathConstant::_2_PI, L.rows() * 0.5) * L.determinant();
	return numer / denom;
}

/*static*/ double Statistic::multivariateNormalDistibutionLogPdf(const Eigen::VectorXd &x, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov)
// Compute log(p(x)) at the point x using mean vector and variance-covariance matrix.
{
	if (x.size() != mean.size() || x.size() != cov.rows() || cov.rows() != cov.cols())
		throw std::runtime_error("Invalid vector or matrix size");

	Eigen::LLT<Eigen::MatrixXd> chol(cov);
	if (Eigen::Success != chol.info())
		throw std::runtime_error("Cholesky decompoistion failed");

	const Eigen::LLT<Eigen::MatrixXd>::Traits::MatrixL& L = chol.matrixL();
	return -0.5 * L.solve(x - mean).squaredNorm() - 0.5 * L.rows() * std::log(MathConstant::_2_PI) - std::log(L.determinant());
}

}  //  namespace swl
