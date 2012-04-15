//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/math/MathConstant.h"
#include "swl/rnd_util/Ransac.h"
#include <iostream>
#include <ctime>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

struct Point3
{
	Point3(const double _x, const double _y, const double _z)
	: x(_x), y(_y), z(_z)
	{}
	Point3(const Point3 &rhs)
	: x(rhs.x), y(rhs.y), z(rhs.z)
	{}

	double x, y, z;
};

class Plane3RansacEstimator: public swl::Ransac
{
public:
	typedef swl::Ransac base_type;

public:
	Plane3RansacEstimator(const std::vector<Point3> &samples, const size_t minimalSampleSetSize)
	: base_type(samples.size(), minimalSampleSetSize), samples_(samples)
	{
	}
	Plane3RansacEstimator(const std::vector<Point3> &samples, const size_t minimalSampleSetSize, const std::vector<double> &scores)
	: base_type(samples.size(), minimalSampleSetSize, scores), samples_(samples)
	{
	}

public:
	double getA() const  {  return a_;  }
	double getB() const  {  return b_;  }
	double getC() const  {  return c_;  }
	double getD() const  {  return d_;  }

private:
	/*virtual*/ bool estimateModel(const std::vector<size_t> &indices);
	/*virtual*/ bool verifyModel() const;
	/*virtual*/ bool estimateModelFromInliers();

	// for RANSAC
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const;
	// for MLESAC
	/*virtual*/ void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const;
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const;

	bool calculateNormal(const double vx1, const double vy1, const double vz1, const double vx2, const double vy2, const double vz2, double &nx, double &ny, double &nz) const
	{
		nx = vy1 * vz2 - vz1 * vy2;
		ny = vz1 * vx2 - vx1 * vz2;
		nz = vx1 * vy2 - vy1 * vx2;

		const double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
		const double eps = 1.0e-20;
		if (norm < eps) return false;

		nx /= norm;
		ny /= norm;
		nz /= norm;
		return true;
	}

private:
	const std::vector<Point3> &samples_;

	// plane equation: a * x + b * y + c * z + d = 0
	double a_, b_, c_, d_;
};

bool Plane3RansacEstimator::estimateModel(const std::vector<size_t> &indices)
{
	if (indices.size() < minimalSampleSetSize_) return false;

	const Point3 &pt1 = samples_[indices[0]];
	const Point3 &pt2 = samples_[indices[1]];
	const Point3 &pt3 = samples_[indices[2]];

	if (calculateNormal(pt2.x - pt1.x, pt2.y - pt1.y, pt2.z - pt1.z, pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z, a_, b_, c_))
	{
		d_ = -(a_ * pt1.x + b_ * pt1.y + c_ * pt1.z);
		return true;
	}
	else return false;
}

bool Plane3RansacEstimator::verifyModel() const
{
	return true;
}

bool Plane3RansacEstimator::estimateModelFromInliers()
{
	// TODO [improve] >> (e.g.) can find the least squares solution from inliers
	return true;
}

size_t Plane3RansacEstimator::lookForInliers(std::vector<bool> &inliers, const double threshold) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = std::fabs(a_ * it->x + b_ * it->y + c_ * it->z + d_) < threshold;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

void Plane3RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double factor = 1.0 / std::sqrt(2.0 * swl::MathConstant::PI * inlierSquaredStandardDeviation);

	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// compute errors
		const double err = a_ * it->x + b_ * it->y + c_ * it->z + d_;

		// compute inliers' probabilities
		inlierProbs[k] = factor * std::exp(-0.5 * err * err / inlierSquaredStandardDeviation);
	}
}

size_t Plane3RansacEstimator::lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = inlierProbs[k] >= outlierUniformProbability;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

}  // namespace local
}  // unnamed namespace

void estimate_3d_plane_using_ransac()
{
	std::srand((unsigned int)time(NULL));

	const size_t N_plane = 30;
	const size_t N_noise = 100;

	// generate random points
	std::vector<local::Point3> samples;
	{
		const double PLANE_EQ[4] = { 1, -1, 1, -2 };  // { 0.5774, -0.5774, 0.5774, -1.1547 }

		for (size_t i = 0; i < N_plane; ++i)
		{
			const double x = std::rand() % 10001 * 0.0006 - 3.0;  // [-3, 3]
			const double y = std::rand() % 10001 * 0.0006 - 3.0;  // [-3, 3]
			const double z = -(PLANE_EQ[3] + PLANE_EQ[0] * x + PLANE_EQ[1] * y) / PLANE_EQ[2];
			samples.push_back(local::Point3(x, y, z));
		}

		for (size_t i = 0; i < N_noise; ++i)
		{
			const double x = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5]
			const double y = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5]
			const double z = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5]
			samples.push_back(local::Point3(x, y, z));
		}
	}

	const size_t minimalSampleSetSize = 3;
	local::Plane3RansacEstimator ransac(samples, minimalSampleSetSize);

	const size_t maxIterationCount = 500;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = false;

	std::cout << "********* RANSAC" << std::endl;
	{
		const double threshold = 0.2;

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, threshold);

		std::cout << "the number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "the number of inliers: " << inlierCount << std::endl;
		std::cout << "indices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}

	std::cout << "********* MLESAC" << std::endl;
	{
		const double inlierSquaredStandardDeviation = 0.15;
		const double outlierUniformProbability = 0.1;
		const size_t maxEMIterationCount = 10;

		const size_t inlierCount = ransac.runMLESAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, inlierSquaredStandardDeviation, outlierUniformProbability, maxEMIterationCount);

		std::cout << "the number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "the number of inliers: " << inlierCount << std::endl;
		std::cout << "indices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}
}
