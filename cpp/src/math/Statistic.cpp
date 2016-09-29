#include "swl/Config.h"
#include "swl/math/Statistic.h"
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(PI)  // For djgpp.
#	undef PI
#endif


namespace swl {

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
