#include "swl/Config.h"
#include "swl/math/DataNormalization.h"
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(PI)  // For djgpp.
#	undef PI
#endif


namespace swl {

/*static*/ bool DataNormalization::normalizeDataByCentering(Eigen::MatrixXd &D)
// goal : Mean of each row = 0.
// row : The dimension of data.
// col : The number of data.
{
	// Centered data.
	D.colwise() -= D.rowwise().mean();
	return true;
}

/*static*/ bool DataNormalization::normalizeDataByAverageDistance(Eigen::MatrixXd &D, const double averageDistance, const double tol /*= MathConstant::EPS*/)
// goal : 2-norm of each row = averageDistance.
// row : The dimension of data.
// col : The number of data.
{
	const std::size_t rows = D.rows();

	const Eigen::VectorXd normVec(D.rowwise().norm());  // 2-norm.
#if 0
	D.array() *= averageDistance;
	D.colwise() /= normVec;  // Error : THIS_METHOD_IS_ONLY_FOR_ARRAYS_NOT_MATRICES.
	//D.colwise() *= normVec.cwiseInverse();  // Error : THIS_METHOD_IS_ONLY_FOR_ARRAYS_NOT_MATRICES.
#else
	for (std::size_t i = 0; i < rows; ++i)
	{
		// TODO [check] >>
		if (normVec(i) >= tol)
			D.row(i) *= averageDistance / normVec(i);
	}
#endif

	return true;
}

/*static*/ bool DataNormalization::normalizeDataByRange(Eigen::MatrixXd &D, const double minBound, const double maxBound, const double tol /*= MathConstant::EPS*/)
// goal : Min & max of each row = [minBound, maxBound].
// row : The dimension of data.
// col : The number of data.
{
	const std::size_t rows = D.rows();
	const std::size_t cols = D.cols();

	const Eigen::VectorXd minVec(D.rowwise().minCoeff());
	const Eigen::VectorXd maxVec(D.rowwise().maxCoeff());

#if 0
	// FIXME [modify] >> Have to consider the case that the max. and min. values are equal.
	const Eigen::VectorXd factor((maxVec - minVec) / (maxBound - minBound));
	D = ((D.colwise() - minVec).colwise() / factor).array() + minBound;  // Error : THIS_METHOD_IS_ONLY_FOR_ARRAYS_NOT_MATRICES.
#elif 0
	// FIXME [modify] >> Have to consider the case that the max. and min. values are equal.
	const Eigen::VectorXd factor((maxVec - minVec) / (maxBound - minBound));
	D.colwise() -= minVec;
	D.colwise() /= factor;  // Error : THIS_METHOD_IS_ONLY_FOR_ARRAYS_NOT_MATRICES.
	D.array() += minBound;
#elif 1
	for (std::size_t i = 0; i < rows; ++i)
	{
		if (std::abs(maxVec(i) - minVec(i)) < tol)
		{
			// TODO [check] >>
			//D.row(i).array() = (maxBound - minBound) * 0.5;
			D.row(i).setConstant((maxBound - minBound) * 0.5);
		}
		else
		{
			const double factor = (maxBound - minBound) / (maxVec(i) - minVec(i));
			D.row(i) = ((D.row(i).array() - minVec(i)) * factor).array() + minBound;
		}
	}
#else
	for (std::size_t i = 0; i < rows; ++i)
	{
		if (std::abs(maxVec(i) - minVec(i)) < tol)
		{
			// TODO [check] >>
			//D.row(i).array() = (maxBound - minBound) * 0.5;
			D.row(i).setConstant((maxBound - minBound) * 0.5);
		}
		else
		{
			const double factor = (maxBound - minBound) / (maxVec(i) - minVec(i));
			for (std::size_t j = 0; j < cols; ++j)
				D(i, j) = (D(i, j) - minVec(i)) * factor + minBound;
		}
	}
#endif

	return true;
}

/*static*/ bool DataNormalization::normalizeDataByLinearTransformation(Eigen::MatrixXd &D, const Eigen::MatrixXd &T)
// goal : D = T * D where T = row * row.
// row : The dimension of data.
// col : The number of data.
{
	const std::size_t rows = D.rows();
	if (T.rows() != rows || T.cols() != rows)
		return false;

	D = T * D;

	return true;
}

/*static*/ bool DataNormalization::normalizeDataByHomogeneousTransformation(Eigen::MatrixXd &D, const Eigen::MatrixXd &H)
// goal : D = H * D where H = row * (row + 1).
// row : The dimension of data.
// col : The number of data.
{
	const std::size_t rows = D.rows();
	if (H.rows() != rows || H.cols() != (rows + 1))
		return false;

#if 0
	D = (H.leftCols(rows) * D).colwise() + H.rightCols(1);  // Error : YOU_TRIED_CALLING_A_VECTOR_METHOD_ON_A_MATRIX.
#else
	D = (H.leftCols(rows) * D).colwise() + Eigen::VectorXd(H.rightCols(1));
#endif

	return true;
}

/*static*/ bool DataNormalization::normalizeDataByZScore(Eigen::MatrixXd &D, const double sigma)
// goal : z = (x - sample-mean) / sigma for each row.
// row : The dimension of data.
// col : The number of data.
{
#if 1
	D = (D.colwise() - D.rowwise().mean()).array() / sigma;
#else
	// centered data.
	D.colwise() -= D.rowwise().mean();
	D.array() /= sigma;
#endif

	return true;
}

/*static*/ bool DataNormalization::normalizeDataByTStatistic(Eigen::MatrixXd &D, const double tol /*= MathConstant::EPS*/)
// goal : t = (x - sample-mean) / sqrt(sample-variance / N) for each row.
// row : The dimension of data.
// col : The number of data.
{
	const std::size_t rows = D.rows();
	const std::size_t cols = D.cols();

	// Centered data.
	D.colwise() -= D.rowwise().mean();

	// sqrt(sample variance) = sample standard deviation.
	Eigen::VectorXd stdDev((D.cwiseProduct(D).rowwise().sum().array() / double(cols - 1)).cwiseSqrt());

	stdDev.array() /= std::sqrt((double)cols);
#if 0
	D.colwise() /= stdDev;  // Error : THIS_METHOD_IS_ONLY_FOR_ARRAYS_NOT_MATRICES.
	//D.colwise() *= stdDev.cwiseInverse();  // Error : THIS_METHOD_IS_ONLY_FOR_ARRAYS_NOT_MATRICES.
#else
	for (std::size_t i = 0; i < rows; ++i)
	{
		// TODO [check] >>
		if (stdDev(i) >= tol)
			D.row(i) /= stdDev(i);
	}
#endif

	return true;
}

}  //  namespace swl
