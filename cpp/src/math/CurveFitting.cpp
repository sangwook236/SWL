#include "swl/Config.h"
#include "swl/math/CurveFitting.h"
#include <Eigen/Dense>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

// Line equation: a * x + b * y + c = 0.
/*static*/ bool CurveFitting::estimateLineByLeastSquares(const std::list<std::array<double, 2> >& points, double& a, double& b, double& c)
{
	const size_t numPoints = points.size();

#if 1
	const size_t dim = 3;
	Eigen::MatrixXd AA(numPoints, dim);
	{
		size_t k = 0;
		for (const auto& pt : points)
		{
			AA(k, 0) = pt[0]; AA(k, 1) = pt[1]; AA(k, 2) = 1.0;
			++k;
		}
	}

	// Use SVD for linear least squares.
	// MxN matrix, K=min(M,N), M>=N.
	//const Eigen::SVD<Eigen::MatrixXd> svd(AA);
	const Eigen::JacobiSVD<Eigen::MatrixXd>& svd = AA.jacobiSvd(Eigen::ComputeThinV);
	// Right singular vectors: KxN matrix.
	const Eigen::JacobiSVD<Eigen::MatrixXd>::MatrixVType& V = svd.matrixV();
	assert(dim == V.rows());

	const size_t last = V.rows() - 1;
	a = V(last, 0);
	b = V(last, 1);
	c = V(last, 2);
#else
	const size_t dim = 2;
	if (numPoints < dim) return false;

	Eigen::MatrixXd AA(numPoints, dim);
	Eigen::VectorXd bb(numPoints);
	{
		size_t k = 0;
		for (const auto& pt : points)
		{
			AA(k, 0) = pt[0]; AA(k, 1) = 1.0;
			bb(k) = pt[1];
			++k;
		}
	}

#if 1
	// Use SVD for linear least squares.
	const Eigen::VectorXd& sol = AA.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bb);
#else
	// Use normal matrix for linear least squares.
	//const Eigen::VectorXd& sol = (AA.transpose() * AA).inverse() * AA.transpose() * bb;  // Slow.
	const Eigen::VectorXd& sol = (AA.transpose() * AA).ldlt().solve(AA.transpose() * bb);
#endif
	assert(dim == sol.size());

	// NOTICE [caution] >> How to handle a case where a line is nearly vertical (b ~= 0, infinite slope).
	a = sol(0);
	b = -1.0;
	c = sol(1);
#endif

	return true;
}

// Quadratic equation: a * x^2 + b * x + c * y + d = 0.
// Line equation: b * x + c * y + d = 0 if a = 0.
/*static*/ bool CurveFitting::estimateQuadraticByLeastSquares(const std::list<std::array<double, 2> >& points, double& a, double& b, double& c, double& d)
{
	const size_t numPoints = points.size();

#if 1
	const size_t dim = 4;
	Eigen::MatrixXd AA(numPoints, dim);
	{
		size_t k = 0;
		for (const auto& pt : points)
		{
			AA(k, 0) = pt[0] * pt[0]; AA(k, 1) = pt[0]; AA(k, 2) = pt[1]; AA(k, 3) = 1.0;
			++k;
		}
	}

	// Use SVD for linear least squares.
	// MxN matrix, K=min(M,N), M>=N.
	//const Eigen::SVD<Eigen::MatrixXd> svd(AA);
	const Eigen::JacobiSVD<Eigen::MatrixXd>& svd = AA.jacobiSvd(Eigen::ComputeThinV);
	// Right singular vectors: KxN matrix.
	const Eigen::JacobiSVD<Eigen::MatrixXd>::MatrixVType& V = svd.matrixV();
	assert(dim == V.rows());

	const size_t last = V.rows() - 1;
	a = V(last, 0);
	b = V(last, 1);
	c = V(last, 2);
	d = V(last, 3);
#else
	const size_t dim = 3;
	if (numPoints < dim) return false;

	Eigen::MatrixXd AA(numPoints, dim);
	Eigen::VectorXd bb(numPoints);
	{
		size_t k = 0;
		for (const auto& pt : points)
		{
			AA(k, 0) = pt[0] * pt[0]; AA(k, 1) = pt[0]; AA(k, 2) = 1.0;
			bb(k) = pt[1];
			++k;
		}
	}

#if 1
	// Use SVD for linear least squares.
	const Eigen::VectorXd& sol = AA.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bb);
#else
	// Use normal matrix for linear least squares.
	//const Eigen::VectorXd& sol = (AA.transpose() * AA).inverse() * AA.transpose() * bb;  // Slow.
	const Eigen::VectorXd& sol = (AA.transpose() * AA).ldlt().solve(AA.transpose() * bb);
#endif
	assert(dim == sol.size());

	// NOTICE [caution] >> How to handle the case where c = 0.
	a = sol(0);
	b = sol(1);
	c = -1.0;
	d = sol(2);
#endif

	return true;
}

}  //  namespace swl
