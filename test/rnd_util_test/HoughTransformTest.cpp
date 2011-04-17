#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/HoughTransform.h"
#include "swl/math/MathConstant.h"
#include <boost/tuple/tuple.hpp>
#include <boost/multi_array.hpp>
#include <list>
#include <deque>
#include <vector>
#include <iostream>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(max)
#undef max
#endif
#if defined(min)
#undef min
#endif


namespace {

struct ShapeInfo
{
	ShapeInfo(const double _x, const double _y, const double _tangentAngle)
	: x(_x), y(_y), tangentAngle(_tangentAngle)
	{}
	ShapeInfo(const ShapeInfo &rhs)
	: x(rhs.x), y(rhs.y), tangentAngle(rhs.tangentAngle)
	{}

	double x, y;
	// 0 <= a tangent angle < 2 * pi
	double tangentAngle;
};

struct RTableEntry
{
	RTableEntry(const double _distance, const double _angle)
	: distance(_distance), angle(_angle)
	{}
	RTableEntry(const RTableEntry &rhs)
	: distance(rhs.distance), angle(rhs.angle)
	{}

	double distance;
	// -pi <= an angle < pi
	double angle;
};

struct ParameterSpaceInfo
{
	ParameterSpaceInfo(const double _min, const double _max, const size_t _resolution)
	: min(_min), max(_max), resolution(_resolution)
	{}
	ParameterSpaceInfo(const ParameterSpaceInfo &rhs)
	: min(rhs.min), max(rhs.max), resolution(rhs.resolution)
	{}

	double min, max;
	size_t resolution;
};

class RectangleHoughTransform: public swl::GeneralizedHoughTransform
{
public:
	typedef swl::GeneralizedHoughTransform base_type;

public:
	RectangleHoughTransform(const size_t tangentAngleCount)
	: base_type(tangentAngleCount), parameterSpace_(), parameterSpaceInfos_(), entries_()
	{
	}

public:
	bool constructParameterSpace(const std::vector<ParameterSpaceInfo> &space);
	bool constructRTable(const std::vector<ShapeInfo> &reference);
	void vote(const std::vector<ShapeInfo> &input);
	void findLocalMaxima(std::list<boost::tuple<size_t, size_t, size_t, size_t, size_t> > &localMaxima, const size_t minVotingCount);

	boost::tuple<double, double, double, double, double, size_t> getParameter(const size_t xcIdx, const size_t ycIdx, const size_t thetaIdx, const size_t sxIdx, const size_t syIdx) const;

private:
	bool isLocalMaximum(const size_t xcIdx0, const size_t ycIdx0, const size_t thetaIdx0, const size_t sxIdx0, const size_t syIdx0, const size_t minVotingCount) const;

private:
	// the parameter space: { xc, yc, theta, sx, sy }
	// (xc, yc) : x & y coordinates of the center
	// theta : rotational angle about z axis 
	// (sx, sy) : scale factors along x & y axes
	boost::multi_array<size_t, 5> parameterSpace_;
	std::vector<ParameterSpaceInfo> parameterSpaceInfos_;

	// a set of pairs which represents (distance,angle) or (dx,dy) from a reference point (x,y) to the centroid of the shape
	std::deque<RTableEntry> entries_;
};

bool RectangleHoughTransform::constructParameterSpace(const std::vector<ParameterSpaceInfo> &space)
{
	if (space.size() != 5) return false;

	parameterSpaceInfos_.assign(space.begin(), space.end());

	const size_t size0 = 0 == parameterSpaceInfos_[0].resolution ? 1 : parameterSpaceInfos_[0].resolution;
	const size_t size1 = 0 == parameterSpaceInfos_[1].resolution ? 1 : parameterSpaceInfos_[1].resolution;
	const size_t size2 = 0 == parameterSpaceInfos_[2].resolution ? 1 : parameterSpaceInfos_[2].resolution;
	const size_t size3 = 0 == parameterSpaceInfos_[3].resolution ? 1 : parameterSpaceInfos_[3].resolution;
	const size_t size4 = 0 == parameterSpaceInfos_[4].resolution ? 1 : parameterSpaceInfos_[4].resolution;

#if defined(NDEBUG) || defined(_STLPORT_VERSION)
	parameterSpace_.resize(boost::extents[size0][size1][size2][size3][size4]);
#else
	// MSVC: compile-time error in debug build: i don't know why
	//parameterSpace_.resize(boost::extents[size0][size1][size2][size3][size4]);
#	error MSVC: compile-time error in debug build
#endif

	return true;
}

bool RectangleHoughTransform::constructRTable(const std::vector<ShapeInfo> &reference)
{
	rTable_.clear();
	entries_.clear();

	const size_t pointCount = reference.size();

	// calculate the center of the reference shape
	double xc = 0.0, yc = 0.0;
	for (std::vector<ShapeInfo>::const_iterator it = reference.begin(); it != reference.end(); ++it)
	{
		xc += it->x;
		yc += it->y;
	}
	xc /= pointCount;
	yc /= pointCount;

	// construct R table
	for (std::vector<ShapeInfo>::const_iterator it = reference.begin(); it != reference.end(); ++it)
	{
		double tangentAngle = it->tangentAngle;
		while (tangentAngle < 0.0) tangentAngle += swl::MathConstant::_2_PI;
		tangentAngle = std::fmod(tangentAngle, swl::MathConstant::_2_PI);
		const size_t tangentAngleIdx = (size_t)std::floor((tangentAngle / swl::MathConstant::_2_PI) * tangentAngleCount_);

		const double dx = xc - it->x, dy = yc - it->y;
		const double r = std::sqrt(dx * dx + dy * dy);
		const double alpha = std::atan2(dy, dx);

		rtable_type::iterator itRTableEntry = rTable_.find(tangentAngleIdx);
		if (rTable_.end() == itRTableEntry)
		{
			entries_.push_back(RTableEntry(r, alpha));
			
			rtable_entry_type entryIndexSet;
			entryIndexSet.insert(entries_.size() - 1);
			
			rTable_.insert(std::make_pair(tangentAngleIdx, entryIndexSet));
		}
		else
		{
			entries_.push_back(RTableEntry(r, alpha));

			rtable_entry_type &entryIndexSet = itRTableEntry->second;
			entryIndexSet.insert(entries_.size() - 1);
		}
	}

	return true;
}

void RectangleHoughTransform::vote(const std::vector<ShapeInfo> &input)
{
	const size_t size0 = 0 == parameterSpaceInfos_[0].resolution ? 1 : parameterSpaceInfos_[0].resolution;
	const size_t size1 = 0 == parameterSpaceInfos_[1].resolution ? 1 : parameterSpaceInfos_[1].resolution;
	const size_t size2 = 0 == parameterSpaceInfos_[2].resolution ? 1 : parameterSpaceInfos_[2].resolution;
	const size_t size3 = 0 == parameterSpaceInfos_[3].resolution ? 1 : parameterSpaceInfos_[3].resolution;
	const size_t size4 = 0 == parameterSpaceInfos_[4].resolution ? 1 : parameterSpaceInfos_[4].resolution;

	memset(parameterSpace_.data(), 0, sizeof(size_t) * size0 * size1 * size2 * size3 * size4);

	const double eps = 1.0e-20;

	const double min0 = parameterSpaceInfos_[0].min;
	const double min1 = parameterSpaceInfos_[1].min;
	const double min2 = parameterSpaceInfos_[2].min;
	const double min3 = parameterSpaceInfos_[3].min;
	const double min4 = parameterSpaceInfos_[4].min;
	const double factor0 = std::fabs(parameterSpaceInfos_[0].max - min0) < eps ? 1.0 : (parameterSpaceInfos_[0].max - min0) / size0;
	const double factor1 = std::fabs(parameterSpaceInfos_[1].max - min1) < eps ? 1.0 : (parameterSpaceInfos_[1].max - min1) / size1;
	const double factor2 = std::fabs(parameterSpaceInfos_[2].max - min2) < eps ? 1.0 : (parameterSpaceInfos_[2].max - min2) / size2;
	const double factor3 = std::fabs(parameterSpaceInfos_[3].max - min3) < eps ? 1.0 : (parameterSpaceInfos_[3].max - min3) / size3;
	const double factor4 = std::fabs(parameterSpaceInfos_[4].max - min4) < eps ? 1.0 : (parameterSpaceInfos_[4].max - min4) / size4;
	for (std::vector<ShapeInfo>::const_iterator it = input.begin(); it != input.end(); ++it)
	{
		const double x = it->x;
		const double y = it->y;

		double tangentAngle = it->tangentAngle;
		while (tangentAngle < 0.0) tangentAngle += swl::MathConstant::_2_PI;
		tangentAngle = std::fmod(tangentAngle, swl::MathConstant::_2_PI);

		for (rtable_type::iterator itRTableEntry = rTable_.begin(); itRTableEntry != rTable_.end(); ++itRTableEntry)
		{
			const double phi_k = itRTableEntry->first * swl::MathConstant::_2_PI / tangentAngleCount_;
			const rtable_entry_type &entryIndexSet = itRTableEntry->second;
			for (rtable_entry_type::const_iterator itEntryIndex = entryIndexSet.begin(); itEntryIndex != entryIndexSet.end(); ++itEntryIndex)
			{
				const double r = entries_[*itEntryIndex].distance;
				const double alpha = entries_[*itEntryIndex].angle;

				// theta : rotational angle about z axis 
				double theta = 0 == parameterSpaceInfos_[2].resolution ? 0.0 : tangentAngle - phi_k;
				while (theta < 0.0) theta += swl::MathConstant::_2_PI;
				theta = std::fmod(theta, swl::MathConstant::_2_PI) - swl::MathConstant::PI;
				const size_t thetaIdx = (size_t)std::floor((theta - min2) / factor2);
				if (thetaIdx >= size2)
					std::cout << "error: invalid index of a rotational angle about z axis !!!" << std::endl;
				const double cos_at = std::cos(alpha + theta);
				const double sin_at = std::sin(alpha + theta);

				// sx : scale factor along x axis: 2^n
				for (size_t sxIdx = 0; sxIdx < size3; ++sxIdx)
				{
					// TODO [check] >>
					//const double sx = 0 == parameterSpaceInfos_[3].resolution ? 1.0 : min3 + sxIdx * factor3;
					const double sx = 0 == parameterSpaceInfos_[3].resolution ? 1.0 : std::pow(2.0, min3 + sxIdx * factor3);

					// sy : scale factor along y axis: 2^n
					for (size_t syIdx = 0; syIdx < size4; ++syIdx)
					{
						// TODO [check] >>
						//const double sy = 0 == parameterSpaceInfos_[4].resolution ? 1.0 : min4 + syIdx * factor4;
						const double sy = 0 == parameterSpaceInfos_[4].resolution ? 1.0 : std::pow(2.0, min4 + syIdx * factor4);

						const double xc = x + r * sx * cos_at;
						const double yc = y + r * sy * sin_at;

						const size_t xcIdx = (size_t)std::floor((xc - min0) / factor0);
						const size_t ycIdx = (size_t)std::floor((yc - min1) / factor1);
						if (xcIdx >= size0 || ycIdx >= size1)
						{
							// FIXME [uncommnet] >>
							//std::cout << "error: invalid index of center's coordiates !!!" << std::endl;
						}
						else
							++parameterSpace_[xcIdx][ycIdx][thetaIdx][sxIdx][syIdx];
					}
				}
			}
		}
	}
}

bool RectangleHoughTransform::isLocalMaximum(const size_t xcIdx0, const size_t ycIdx0, const size_t thetaIdx0, const size_t sxIdx0, const size_t syIdx0, const size_t minVotingCount) const
{
	const size_t votingCount = parameterSpace_[xcIdx0][ycIdx0][thetaIdx0][sxIdx0][syIdx0];
	if (0 == votingCount || votingCount < minVotingCount) return false;

	const size_t size0 = 0 == parameterSpaceInfos_[0].resolution ? 1 : parameterSpaceInfos_[0].resolution;
	const size_t size1 = 0 == parameterSpaceInfos_[1].resolution ? 1 : parameterSpaceInfos_[1].resolution;
	const size_t size2 = 0 == parameterSpaceInfos_[2].resolution ? 1 : parameterSpaceInfos_[2].resolution;
	const size_t size3 = 0 == parameterSpaceInfos_[3].resolution ? 1 : parameterSpaceInfos_[3].resolution;
	const size_t size4 = 0 == parameterSpaceInfos_[4].resolution ? 1 : parameterSpaceInfos_[4].resolution;

	// xc : x coordinate of the center
	for (size_t xc = 0; xc <= 2; ++xc)
	{
		const size_t xcIdx = xcIdx0 + xc - 1;
		if (xcIdx == -1) continue;
		else if (xcIdx >= size0) break;

		// yc : y coordinate of the center
		for (size_t yc = 0; yc <= 2; ++yc)
		{
			const size_t ycIdx = ycIdx0 + yc - 1;
			if (ycIdx == -1) continue;
			else if (ycIdx >= size1) break;

			// theta : rotational angle about z axis
			for (size_t theta = 0; theta <= 2; ++theta)
			{
				const size_t thetaIdx = thetaIdx0 + theta - 1;
				if (thetaIdx == -1) continue;
				else if (thetaIdx >= size2) break;

				// sx : scale factor along x axis
				for (size_t sx = 0; sx <= 2; ++sx)
				{
					const size_t sxIdx = sxIdx0 + sx - 1;
					if (sxIdx == -1) continue;
					else if (sxIdx >= size3) break;

					// sy : scale factor along y axis
					for (size_t sy = 0; sy <= 2; ++sy)
					{
						const size_t syIdx = syIdx0 + sy - 1;
						if (syIdx == -1) continue;
						else if (syIdx >= size4) break;

						if (xcIdx0 == xcIdx && ycIdx0 == ycIdx && thetaIdx0 == thetaIdx && sxIdx0 == sxIdx && syIdx0 == syIdx) continue;

						// FIXME [improve] >>
						if (parameterSpace_[xcIdx][ycIdx][thetaIdx][sxIdx][syIdx] > votingCount)
							return false;
					}
				}
			}
		}
	}

	return true;
}

void RectangleHoughTransform::findLocalMaxima(std::list<boost::tuple<size_t, size_t, size_t, size_t, size_t> > &localMaxima, const size_t minVotingCount)
{
	const size_t size0 = 0 == parameterSpaceInfos_[0].resolution ? 1 : parameterSpaceInfos_[0].resolution;
	const size_t size1 = 0 == parameterSpaceInfos_[1].resolution ? 1 : parameterSpaceInfos_[1].resolution;
	const size_t size2 = 0 == parameterSpaceInfos_[2].resolution ? 1 : parameterSpaceInfos_[2].resolution;
	const size_t size3 = 0 == parameterSpaceInfos_[3].resolution ? 1 : parameterSpaceInfos_[3].resolution;
	const size_t size4 = 0 == parameterSpaceInfos_[4].resolution ? 1 : parameterSpaceInfos_[4].resolution;

	// xc : x coordinate of the center
	for (size_t xcIdx = 0; xcIdx < size0; ++xcIdx)
	{
		// yc : y coordinate of the center
		for (size_t ycIdx = 0; ycIdx < size1; ++ycIdx)
		{
			// theta : rotational angle about z axis
			for (size_t thetaIdx = 0; thetaIdx < size2; ++thetaIdx)
			{
				// sx : scale factor along x axis
				for (size_t sxIdx = 0; sxIdx < size3; ++sxIdx)
				{
					// sy : scale factor along y axis
					for (size_t syIdx = 0; syIdx < size4; ++syIdx)
					{
						if (isLocalMaximum(xcIdx, ycIdx, thetaIdx, sxIdx, syIdx, minVotingCount))
							localMaxima.push_back(boost::make_tuple(xcIdx, ycIdx, thetaIdx, sxIdx, syIdx));
					}
				}
			}
		}
	}
}

boost::tuple<double, double, double, double, double, size_t> RectangleHoughTransform::getParameter(const size_t xcIdx, const size_t ycIdx, const size_t thetaIdx, const size_t sxIdx, const size_t syIdx) const
{
	const size_t size0 = 0 == parameterSpaceInfos_[0].resolution ? 1 : parameterSpaceInfos_[0].resolution;
	const size_t size1 = 0 == parameterSpaceInfos_[1].resolution ? 1 : parameterSpaceInfos_[1].resolution;
	const size_t size2 = 0 == parameterSpaceInfos_[2].resolution ? 1 : parameterSpaceInfos_[2].resolution;
	const size_t size3 = 0 == parameterSpaceInfos_[3].resolution ? 1 : parameterSpaceInfos_[3].resolution;
	const size_t size4 = 0 == parameterSpaceInfos_[4].resolution ? 1 : parameterSpaceInfos_[4].resolution;

	const double eps = 1.0e-20;

	const double min0 = parameterSpaceInfos_[0].min;
	const double min1 = parameterSpaceInfos_[1].min;
	const double min2 = parameterSpaceInfos_[2].min;
	const double min3 = parameterSpaceInfos_[3].min;
	const double min4 = parameterSpaceInfos_[4].min;
	const double factor0 = std::fabs(parameterSpaceInfos_[0].max - min0) < eps ? 1.0 : (parameterSpaceInfos_[0].max - min0) / size0;
	const double factor1 = std::fabs(parameterSpaceInfos_[1].max - min1) < eps ? 1.0 : (parameterSpaceInfos_[1].max - min1) / size1;
	const double factor2 = std::fabs(parameterSpaceInfos_[2].max - min2) < eps ? 1.0 : (parameterSpaceInfos_[2].max - min2) / size2;
	const double factor3 = std::fabs(parameterSpaceInfos_[3].max - min3) < eps ? 1.0 : (parameterSpaceInfos_[3].max - min3) / size3;
	const double factor4 = std::fabs(parameterSpaceInfos_[4].max - min4) < eps ? 1.0 : (parameterSpaceInfos_[4].max - min4) / size4;

	const double xc = 0 == parameterSpaceInfos_[0].resolution ? 0.0 : min0 + xcIdx * factor0;
	const double yc = 0 == parameterSpaceInfos_[1].resolution ? 0.0 : min1 + ycIdx * factor1;
	const double theta = 0 == parameterSpaceInfos_[2].resolution ? 0.0 : min2 + thetaIdx * factor2;
	// TODO [check] >>
	//const double sx = 0 == parameterSpaceInfos_[3].resolution ? 1.0 : min3 + sxIdx * factor3;
	const double sx = 0 == parameterSpaceInfos_[3].resolution ? 1.0 : std::pow(2.0, min3 + sxIdx * factor3);
	//const double sy = 0 == parameterSpaceInfos_[4].resolution ? 1.0 : min4 + syIdx * factor4;
	const double sy = 0 == parameterSpaceInfos_[4].resolution ? 1.0 : std::pow(2.0, min4 + syIdx * factor4);

	return boost::make_tuple(xc, yc, theta, sx, sy, parameterSpace_[xcIdx][ycIdx][thetaIdx][sxIdx][syIdx]);
}

void extract_points_in_rectangle(const double xi, const double  yi, const double xLength, const double yLength, const double rotationAngle, const size_t dataCountPerSide, std::vector<ShapeInfo> &shape)
{
	const double c = std::cos(rotationAngle), s = std::sin(rotationAngle);

	shape.reserve(4 * (dataCountPerSide + 1));

	double x, y, angle;
	for (size_t i = 0; i <= dataCountPerSide; ++i)
	{
		const double xx = i * xLength / dataCountPerSide;
		const double yy = 0.0;

		x = xi + xx * c - yy * s;
		y = yi + xx * s + yy * c;
		angle = rotationAngle + 0.0;
		while (angle < 0.0) angle += swl::MathConstant::_2_PI;
		angle = std::fmod(angle, swl::MathConstant::_2_PI);

		shape.push_back(ShapeInfo(x, y, angle));
	}
	for (size_t i = 0; i <= dataCountPerSide; ++i)
	{
		const double xx = xLength;
		const double yy = i * yLength / dataCountPerSide;

		x = xi + xx * c - yy * s;
		y = yi + xx * s + yy * c;
		angle = rotationAngle + swl::MathConstant::PI_2;
		while (angle < 0.0) angle += swl::MathConstant::_2_PI;
		angle = std::fmod(angle, swl::MathConstant::_2_PI);

		shape.push_back(ShapeInfo(x, y, angle));
	}
	for (size_t i = 0; i <= dataCountPerSide; ++i)
	{
		const double xx = xLength - i * xLength / dataCountPerSide;
		const double yy = yLength;

		x = xi + xx * c - yy * s;
		y = yi + xx * s + yy * c;
		angle = rotationAngle + swl::MathConstant::PI;
		//angle = rotationAngle + 0.0;
		while (angle < 0.0) angle += swl::MathConstant::_2_PI;
		angle = std::fmod(angle, swl::MathConstant::_2_PI);

		shape.push_back(ShapeInfo(x, y, angle));
	}
	for (size_t i = 0; i <= dataCountPerSide; ++i)
	{
		const double xx = 0.0;
		const double yy = yLength - i * yLength / dataCountPerSide;

		x = xi + xx * c - yy * s;
		y = yi + xx * s + yy * c;
		angle = rotationAngle - swl::MathConstant::PI_2;
		//angle = rotationAngle + swl::MathConstant::PI_2;
		while (angle < 0.0) angle += swl::MathConstant::_2_PI;
		angle = std::fmod(angle, swl::MathConstant::_2_PI);

		shape.push_back(ShapeInfo(x, y, angle));
	}
}

void generalized_hough_transform_1()
{
	std::vector<ShapeInfo> referenceShape;
	std::vector<ShapeInfo> inputShape;

	// reference shape(object)
	{
		const double xi = 0.0, yi = 0.0;
		const double xLength = 1.0, yLength = 2.0;
		const double rotationAngle = 15.0 * swl::MathConstant::TO_RAD;
		const size_t dataCountPerSide = 20;

		extract_points_in_rectangle(xi, yi, xLength, yLength, rotationAngle, dataCountPerSide, referenceShape);
	}

	// input shape(object)
	{
		const double xi = 10.0, yi = 5.0;
		const double xLength = 1.0, yLength = 2.0;
		const double rotationAngle = 123.0 * swl::MathConstant::TO_RAD;
		const size_t dataCountPerSide = 20;

		extract_points_in_rectangle(xi, yi, xLength, yLength, rotationAngle, dataCountPerSide, inputShape);
	}

	//
	const size_t tangentAngleCount = 360;  // determine a resolution of tangent angles: 1 [deg]
	RectangleHoughTransform houghTransform(tangentAngleCount);

	std::vector<ParameterSpaceInfo> paramSpace;
	paramSpace.reserve(5);
	paramSpace.push_back(ParameterSpaceInfo(-100.0, 100.0, 200));  // x coordinate of the center
	paramSpace.push_back(ParameterSpaceInfo(-100.0, 100.0, 200));  // y coordinate of the center
	paramSpace.push_back(ParameterSpaceInfo(-swl::MathConstant::PI, swl::MathConstant::PI, 360));  // rotational angle about z axis 
	paramSpace.push_back(ParameterSpaceInfo(-10.0, 10.0, 0));  // scale factor along x axis: 2^n
	paramSpace.push_back(ParameterSpaceInfo(-10.0, 10.0, 0));  // scale factor along y axis: 2^n
	if (!houghTransform.constructParameterSpace(paramSpace))
	{
		std::cout << "parameter space fails to be constructed !!!" << std::endl;
		return;
	}
	if (!houghTransform.constructRTable(referenceShape))
	{
		std::cout << "R-Table fails to be constructed !!!" << std::endl;
		return;
	}

	houghTransform.vote(inputShape);

	size_t sum = 0;
	const size_t minVotingCount = 1;
	std::list<boost::tuple<size_t, size_t, size_t, size_t, size_t> > localMaxima;
	houghTransform.findLocalMaxima(localMaxima, minVotingCount);
	for (std::list<boost::tuple<size_t, size_t, size_t, size_t, size_t> >::iterator it = localMaxima.begin(); it != localMaxima.end(); ++it)
	{
		std::cout << it->get<0>() << ", " << it->get<1>() << ", " << it->get<2>() << ", " << it->get<3>() << ", " << it->get<4>() << " : ";

		const boost::tuple<double, double, double, double, double, size_t> &param = houghTransform.getParameter(it->get<0>(), it->get<1>(), it->get<2>(), it->get<3>(), it->get<4>());
		std::cout << param.get<5>() << " : " << param.get<0>() << ", " << param.get<1>() << ", " << param.get<2>() * swl::MathConstant::TO_DEG << ", " << param.get<3>() << ", " << param.get<4>() << std::endl;

		sum += param.get<5>();
	}
}

void generalized_hough_transform_2()
{
	std::vector<ShapeInfo> referenceShape;
	std::vector<ShapeInfo> inputShape;

	// reference shape(object)
	{
		const double xi = 0.0, yi = 0.0;
		const double xLength = 1.0, yLength = 2.0;
		const double rotationAngle = 123.0 * swl::MathConstant::TO_RAD;
		const size_t dataCountPerSide = 20;

		extract_points_in_rectangle(xi, yi, xLength, yLength, rotationAngle, dataCountPerSide, referenceShape);
	}

	// input shape(object)
	{
		const double xi = -15.0, yi = 10.0;
		const double xLength = 2.0, yLength = 8.0;
		const double rotationAngle = 15.0 * swl::MathConstant::TO_RAD;
		const size_t dataCountPerSide = 20;

		extract_points_in_rectangle(xi, yi, xLength, yLength, rotationAngle, dataCountPerSide, inputShape);
	}

	//
	const size_t tangentAngleCount = 360;  // determine a resolution of tangent angles: 1 [deg]
	RectangleHoughTransform houghTransform(tangentAngleCount);

	std::vector<ParameterSpaceInfo> paramSpace;
	paramSpace.reserve(5);
	paramSpace.push_back(ParameterSpaceInfo(-100.0, 100.0, 25));  // x coordinate of the center
	paramSpace.push_back(ParameterSpaceInfo(-100.0, 100.0, 25));  // y coordinate of the center
	paramSpace.push_back(ParameterSpaceInfo(-swl::MathConstant::PI, swl::MathConstant::PI, 360));  // rotational angle about z axis 
	paramSpace.push_back(ParameterSpaceInfo(-5.0, 5.0, 10));  // scale factor along x axis: 2^n
	paramSpace.push_back(ParameterSpaceInfo(-5.0, 5.0, 10));  // scale factor along y axis: 2^n
	if (!houghTransform.constructParameterSpace(paramSpace))
	{
		std::cout << "parameter space fails to be constructed !!!" << std::endl;
		return;
	}
	if (!houghTransform.constructRTable(referenceShape))
	{
		std::cout << "R-Table fails to be constructed !!!" << std::endl;
		return;
	}

	houghTransform.vote(inputShape);

	size_t sum = 0;
	const size_t minVotingCount = 1;
	std::list<boost::tuple<size_t, size_t, size_t, size_t, size_t> > localMaxima;
	houghTransform.findLocalMaxima(localMaxima, minVotingCount);
	for (std::list<boost::tuple<size_t, size_t, size_t, size_t, size_t> >::iterator it = localMaxima.begin(); it != localMaxima.end(); ++it)
	{
		std::cout << it->get<0>() << ", " << it->get<1>() << ", " << it->get<2>() << ", " << it->get<3>() << ", " << it->get<4>() << " : ";

		const boost::tuple<double, double, double, double, double, size_t> &param = houghTransform.getParameter(it->get<0>(), it->get<1>(), it->get<2>(), it->get<3>(), it->get<4>());
		std::cout << param.get<5>() << " : " << param.get<0>() << ", " << param.get<1>() << ", " << param.get<2>() * swl::MathConstant::TO_DEG << ", " << param.get<3>() << ", " << param.get<4>() << std::endl;

		sum += param.get<5>();
	}
}

}  // unnamed namespace

void hough_transform()
{
	std::cout << "********** generalized Hough transform: a rectangle with the same scale" << std::endl;
	//generalized_hough_transform_1();
	std::cout << "********** generalized Hough transform: a rectangle with different scale" << std::endl;
	generalized_hough_transform_2();
}
