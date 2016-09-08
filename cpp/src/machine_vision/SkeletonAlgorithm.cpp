#include "swl/Config.h"
#include "swl/machine_vision/SkeletonAlgorithm.h"
#include <algorithm>
#include <queue>
#include <vector>
#include <string>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

const size_t NUM_NEIGHBOR_COORDS = 8;
const cv::Point NEIGHBOR_COORDS[] = { cv::Point(1, 0), cv::Point(1, -1), cv::Point(0, -1), cv::Point(-1, -1), cv::Point(-1, 0), cv::Point(-1, 1), cv::Point(0, 1), cv::Point(1, 1) };

std::list<cv::Point> getNeighborsUsingNeighborCoords(const cv::Mat& bw, const cv::Point& pt)
{
	std::list<cv::Point> neighbors;
	for (int i = 0; i < NUM_NEIGHBOR_COORDS; ++i)
	{
		const cv::Point neigh(pt + NEIGHBOR_COORDS[i]);
		if (neigh.x >= 0 && neigh.x < bw.cols && neigh.y >= 0 && neigh.y < bw.rows && (int)bw.at<unsigned char>(neigh.y, neigh.x))
			neighbors.push_back(neigh);
	}

	return neighbors;
}

struct LineSegment
{
public:
	explicit LineSegment(const cv::Point& pt1, const cv::Point& pt2)
	: pt1_(pt1), pt2_(pt2)
	{}

public:
	bool operator<(const LineSegment& rhs) const
	{
		// FIXME [check] >>
		return pt1_.x < rhs.pt1_.x && pt2_.x < rhs.pt2_.x;
	}

public:
	bool isOverlapped(const LineSegment& rhs, const int skeletonOverlapMargin) const
	{
		return !(pt1_.x - skeletonOverlapMargin > rhs.pt2_.x || pt2_.x + skeletonOverlapMargin < rhs.pt1_.x);
	}
	bool isContained(const cv::Point& pt) const
	{
		return std::abs(pt1_.y - pt.y) < 1 && pt1_.x <= pt.x && pt.x <= pt2_.x;
	}

	cv::Point& getPt1() { return pt1_; }
	const cv::Point& getPt1() const { return pt1_; }
	cv::Point& getPt2() { return pt2_; }
	const cv::Point& getPt2() const { return pt2_; }
	cv::Point getMidPoint() const { return (pt1_ + pt2_) / 2; }

private:
	cv::Point pt1_, pt2_;
};

struct PrContainPointInLineSegment
{
public:
	explicit PrContainPointInLineSegment(const LineSegment& line)
	: line_(line)
	{}

	bool operator()(const cv::Point& pt) const
	{
		return line_.isContained(pt);
	}

private:
	const LineSegment& line_;
};

struct PrContainVertexInLineSegment
{
public:
	explicit PrContainVertexInLineSegment(const LineSegment& line, const swl::SkeletonAlgorithm::VertexType type)
		: line_(line), type_(type)
	{}

	bool operator()(const swl::SkeletonAlgorithm::Vertex& vtx) const
	{
		return line_.isContained(vtx.pt) && type_ == vtx.type;
	}

private:
	const LineSegment& line_;
	const swl::SkeletonAlgorithm::VertexType type_;
};

void findLineSegmentsInRow(const cv::Mat& bw, const int row, const int colStart, const int colEnd, std::map<LineSegment, int>& lineSegments)
{
	for (int c = colStart; c <= colEnd; ++c)
	{
		if ((int)bw.at<unsigned char>(row, c) > 0)
		{
			const int segmentStart = c;
			for (++c; c < bw.cols; ++c)
			{
				if ((int)bw.at<unsigned char>(row, c) == 0)
					break;
			}
			lineSegments.insert(std::make_pair(LineSegment(cv::Point(segmentStart, row), cv::Point(c - 1, row)), -1));
		}
	}
}

void showVertices(cv::Mat& rgb, const std::string& windowName, const std::list<swl::SkeletonAlgorithm::Vertex>& vertices)
{
	int count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::ROOT == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << "#Roots = " << count;

	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::LEAF_END == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Leaf-ends = " << count;
	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::LEAF_CONCAVE == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(255, 255, 0), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Leaf-concaves = " << count;
	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::LEAF_CONVEX == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(64, 64, 0), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Leaf-convexes = " << count;

	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::BRANCH == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Branches = " << count;
	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::JOIN == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(255, 0, 255), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Joins = " << count;

	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::CROSS == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Crosses = " << count;
	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::CONTACT == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(0, 64, 64), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Contacts = " << count;

	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::INTERNAL == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(0, 128, 255), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Internals = " << count;

	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::ISOLATED == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(0, 64, 0), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Isolateds = " << count;

	count = 0;
	for (auto vtx : vertices)
		if (swl::SkeletonAlgorithm::UNDEFINED == vtx.type)
		{
			cv::circle(rgb, vtx.pt, 3, cv::Scalar(128, 128, 128), cv::FILLED, cv::LINE_AA);
			++count;
		}
	std::cout << ", #Undefineds = " << count << std::endl;

	cv::imshow(windowName, rgb);
}

}  // namespace local
}  // unnamed namespace

namespace swl {

/*static*/ void SkeletonAlgorithm::findLeafEndVertices(const cv::Mat& bw, std::list<cv::Point>& leafEndVertices)
{
	// DbC [precondition] {required} >> The thickness of skeletons in an input image bw is 1.

	// Create a kernel that we will use for accuting/sharpening our image.
	const cv::Mat kernel = (cv::Mat_<short>(3, 3) <<
		1, 1, 1,
		1, 0, 1,
		1, 1, 1);

	// FIXME [improve] >> Do filtering for non-zero pixels only.
	cv::Mat result;
	cv::filter2D(bw, result, CV_16S, kernel);
	result.setTo(cv::Scalar::all(0), bw == 0);
	result.setTo(cv::Scalar::all(0), result != 255);  // Single point.

	cv::Mat result_uchar;
	result.convertTo(result_uchar, CV_8UC1);

	std::vector<cv::Point> pixels;
	cv::findNonZero(result_uchar, pixels);

	leafEndVertices.assign(pixels.begin(), pixels.end());
}

/*static*/ void SkeletonAlgorithm::findInternalVertices(const cv::Mat& bw, std::list<cv::Point>& internalVertices)
{
	// DbC [precondition] {required} >> The thickness of skeletons in an input image bw is 1.

	// FIXME [correct] >> Not correctly working.

	// Create a kernel.
	const cv::Mat kernel = (cv::Mat_<short>(3, 3) <<
		1, 1, 1,
		1, 0, 1,
		1, 1, 1);

	// FIXME [improve] >> Do filtering for non-zero pixels only.
	cv::Mat result;
	cv::filter2D(bw, result, CV_16S, kernel);
	result.setTo(cv::Scalar::all(0), bw == 0);
	result.setTo(cv::Scalar::all(0), result < 765);  // Greater than 3 points: 255 * 3 = 765. 

	cv::Mat result_uchar;
	result.convertTo(result_uchar, CV_8UC1);

	std::vector<cv::Point> pixels;
	cv::findNonZero(result_uchar, pixels);

	internalVertices.assign(pixels.begin(), pixels.end());
}

/*static*/ size_t SkeletonAlgorithm::constructEdgeGroups(const cv::Mat& bw, const cv::Point& center, std::map<int, int>& edgeGroupIds)
// idx = (neighbor.y - center.y + 1) * 3 + (neighbor.x - center.x + 1).
// If edgeGroupIds[idx] == 0, no edge (black pixel).
// If edgeGroupIds[idx] > 0, edge group ID (white pixel). Use one-based index.
// If edgeGroupIds[idx] < 0, no pixel (boundary).
{
	unsigned char prevVal;
	bool hasPrevPt = false;
	{
		const cv::Point pt(center.x + local::NEIGHBOR_COORDS[local::NUM_NEIGHBOR_COORDS - 1].x, center.y + local::NEIGHBOR_COORDS[local::NUM_NEIGHBOR_COORDS - 1].y);
		if (pt.x >= 0 && pt.x < bw.cols && pt.y >= 0 && pt.y < bw.rows)
		{
			prevVal = bw.at<unsigned char>(pt.y, pt.x);
			hasPrevPt = true;
		}
	}

	size_t numEdgeGroups = 0;
	for (size_t i = 0; i < local::NUM_NEIGHBOR_COORDS; ++i)
	{
		const cv::Point neighbor(center.x + local::NEIGHBOR_COORDS[i].x, center.y + local::NEIGHBOR_COORDS[i].y);
		if (neighbor.x < 0 || neighbor.x >= bw.cols || neighbor.y < 0 || neighbor.y >= bw.rows)
		{
			edgeGroupIds[(neighbor.y - center.y + 1) * 3 + (neighbor.x - center.x + 1)] = -1;  // No pixel.
			hasPrevPt = false;
			continue;
		}

		const unsigned char currVal = bw.at<unsigned char>(neighbor.y, neighbor.x);
		if (currVal)
		{
			if (0 == i) ++numEdgeGroups;  // New edge group.
			else
			{
				if (hasPrevPt)
				{
					if (currVal != prevVal) ++numEdgeGroups;  // New edge group.
				}
				else ++numEdgeGroups;  // New edge group.
			}
			edgeGroupIds[(neighbor.y - center.y + 1) * 3 + (neighbor.x - center.x + 1)] = numEdgeGroups;
		}
		else
			edgeGroupIds[(neighbor.y - center.y + 1) * 3 + (neighbor.x - center.x + 1)] = 0;  // No edge.

		prevVal = currVal;
		hasPrevPt = true;
	}

	const cv::Point first(center.x + local::NEIGHBOR_COORDS[0].x, center.y + local::NEIGHBOR_COORDS[0].y);
	const cv::Point last(center.x + local::NEIGHBOR_COORDS[local::NUM_NEIGHBOR_COORDS - 1].x, center.y + local::NEIGHBOR_COORDS[local::NUM_NEIGHBOR_COORDS - 1].y);
	if (first.x >= 0 && first.x < bw.cols && first.y >= 0 && first.y < bw.rows && last.x >= 0 && last.x < bw.cols && last.y >= 0 && last.y < bw.rows)
	{
		const unsigned char firstVal = bw.at<unsigned char>(first.y, first.x), lastVal = bw.at<unsigned char>(last.y, last.x);
		if (firstVal && firstVal == lastVal)
		{
			for (std::map<int, int>::iterator it = edgeGroupIds.begin(); it != edgeGroupIds.end(); ++it)
				if (it->second == numEdgeGroups) it->second = 1;  // Change into the first edge group.
			--numEdgeGroups;
		}
	}

	return numEdgeGroups;
}

// REF [function] >> constructEdgeGroups().
/*static*/ bool SkeletonAlgorithm::isInTheSameEdgeGroup(const std::map<int, int>& edgeGroupIds, const cv::Point& center, const cv::Point& pt1, const cv::Point& pt2)
{
	std::map<int, int>::const_iterator cit1 = edgeGroupIds.find((pt1.y - center.y + 1) * 3 + (pt1.x - center.x + 1));
	std::map<int, int>::const_iterator cit2 = edgeGroupIds.find((pt2.y - center.y + 1) * 3 + (pt2.x - center.x + 1));
	return edgeGroupIds.end() != cit1 && edgeGroupIds.end() != cit2 && cit1->second > 0 && cit1->second == cit2->second;
}

// REF [function] >> constructEdgeGroups().
/*static*/ bool SkeletonAlgorithm::isOnBorder(const std::map<int, int>& edgeGroupIds)
{
	for (std::map<int, int>::const_iterator cit = edgeGroupIds.begin(); cit != edgeGroupIds.end(); ++cit)
		if (cit->second < 0) return true;
	return false;
}

// REF [function] >> constructEdgeGroups().
/*static*/ bool SkeletonAlgorithm::isSurroundedBySingleEdgeGroup(const std::map<int, int>& edgeGroupIds)
{
	int id = -1;
	for (std::map<int, int>::const_iterator cit = edgeGroupIds.begin(); cit != edgeGroupIds.end(); ++cit)
		if (0 == cit->second) return false;
	//else if (cit->second > 0)
	//{
	//	if (-1 == id) id = cit->second;
	//	else if (id != cit->second) return false;
	//}
	return true;
}

/*static*/ bool SkeletonAlgorithm::checkIfVertex(const cv::Mat& bw, const cv::Point& currPt, SkeletonAlgorithm::VertexType* vertexType)
// If a point has 3 edge groups or more, it is a vertex.
{
#if 0
	// NOTICE [info] >> Too naive implementation.

	// Get the number of edge groups.
	unsigned char prevVal;
	bool hasPrevPt = false;
	{
		const cv::Point pt(currPt.x + local::NEIGHBOR_COORDS[local::NUM_NEIGHBOR_COORDS - 1].x, currPt.y + local::NEIGHBOR_COORDS[local::NUM_NEIGHBOR_COORDS - 1].y);
		if (pt.x >= 0 && pt.x < bw.cols && pt.y >= 0 && pt.y < bw.rows)
		{
			prevVal = bw.at<unsigned char>(pt.y, pt.x);
			hasPrevPt = true;
		}
	}
	size_t numEdgeGroups = 0;
	int sumPixels = 0;
	int numValidPixels = 0;
	for (size_t i = 0; i < local::NUM_NEIGHBOR_COORDS; ++i)
	{
		const cv::Point neighbor(currPt.x + local::NEIGHBOR_COORDS[i].x, currPt.y + local::NEIGHBOR_COORDS[i].y);
		if (neighbor.x < 0 || neighbor.x >= bw.cols || neighbor.y < 0 || neighbor.y >= bw.rows)
		{
			hasPrevPt = false;
			continue;
		}

		const unsigned char val = bw.at<unsigned char>(neighbor.y, neighbor.x);
		sumPixels += val;
		++numValidPixels;
		if (val)
		{
			if (hasPrevPt)
			{
				if (val != prevVal) ++numEdgeGroups;
			}
			else ++numEdgeGroups;
		}

		prevVal = val;
		hasPrevPt = true;
	}

	switch (numEdgeGroups)
	{
	case 2:  // If the point is not a vertex.
		return false;
	case 0:
		if (prevVal == bw.at<unsigned char>(currPt.y, currPt.x))
			return false;
		else  // Isolated vertex.
		{
			if (vertexType) *vertexType = SkeletonAlgorithm::ISOLATED;
			return true;
		}
		break;
	case 1:
		if (numValidPixels < 8)  // If the current point is on the border of the image.
		{
			switch (sumPixels / numValidPixels)
			{
			case 0:  // Surrounded by black pixels.
				if (!bw.at<unsigned char>(currPt.y, currPt.x))
					return false;
				else  // Isolated vertex.
				{
					if (vertexType) *vertexType = SkeletonAlgorithm::ISOLATED;
					return true;
				}
				break;
			case 255:  // Surrounded by white pixels.
				if (bw.at<unsigned char>(currPt.y, currPt.x))
					return false;
				else  // Isolated vertex?
				{
					if (vertexType) *vertexType = SkeletonAlgorithm::ISOLATED;
					return true;
				}
				break;
			default:  // Root or leaf vertex.
				if (vertexType) *vertexType = SkeletonAlgorithm::LEAF_END;
				return true;
			}
		}
		else  // Root or leaf vertex.
		{
			if (vertexType) *vertexType = SkeletonAlgorithm::LEAF_END;
			return true;
		}
		break;
	default:
		if (vertexType) *vertexType = SkeletonAlgorithm::INTERNAL;
		return true;
	}

	return false;
#else
	const unsigned char currVal = bw.at<unsigned char>(currPt.y, currPt.x);
	if (!currVal) return false;

	// Get edge groups.
	std::map<int, int> edgeGroupIds;
	const size_t numEdgeGroupIds = constructEdgeGroups(bw, currPt, edgeGroupIds);  // Use one-based index.

	switch (numEdgeGroupIds)
	{
	case 2:  // If the point is not a vertex.
		return false;
	case 0:  // Isolated vertex surrounded by black pixels.
		if (vertexType) *vertexType = SkeletonAlgorithm::ISOLATED;
		return true;
	case 1:
		if (isOnBorder(edgeGroupIds))  // If the current point is on the border of an image.
		{
			if (isSurroundedBySingleEdgeGroup(edgeGroupIds))  // Isolated(?) vertex surrounded by white pixels.
			{
				//if (vertexType) *vertexType = SkeletonAlgorithm::ISOLATED;
				return false;  // TODO [check] >> Is it normal?
			}
			else  // Root or leaf vertex.
			{
				//if (vertexType) *vertexType = SkeletonAlgorithm::ROOT;
				if (vertexType) *vertexType = SkeletonAlgorithm::LEAF_END;
				return true;
			}
		}
		else  // Root or leaf vertex.
		{
			//if (vertexType) *vertexType = SkeletonAlgorithm::ROOT;
			if (vertexType) *vertexType = SkeletonAlgorithm::LEAF_END;
			return true;
		}
		break;
	default:
		if (vertexType) *vertexType = SkeletonAlgorithm::INTERNAL;
		return true;
	}

	return false;
#endif
}

/*static*/ void SkeletonAlgorithm::findVerticesAndEdgesByUpwardLineScanning(const cv::Mat& bw, const int rootVerticesLocation, const int skeletonOverlapMargin, std::list<SkeletonAlgorithm::Vertex>& vertices, std::list<std::pair<const int, const int> >& edges)
{
	const int bottomRow = bw.rows - rootVerticesLocation;
	int vertexCount = 0;

	std::map<local::LineSegment, int> prevLineSegments;  // Pair of line segment & starting vertex ID.
	local::findLineSegmentsInRow(bw, bottomRow, 0, bw.cols - 1, prevLineSegments);
	for (std::map<local::LineSegment, int>::iterator itLineMap = prevLineSegments.begin(); itLineMap != prevLineSegments.end(); ++itLineMap)
	{
		itLineMap->second = vertexCount;
		vertices.push_back(SkeletonAlgorithm::Vertex(itLineMap->first.getMidPoint(), SkeletonAlgorithm::ROOT, vertexCount++));
	}

	// Find leaf-end vertices w/o information on relationship with other vertices (more correct).
	{
		std::list<cv::Point> leafEndVertices;
		findLeafEndVertices(bw, leafEndVertices);
		for (std::list<cv::Point>::const_iterator itLeafEnd = leafEndVertices.begin(); itLeafEnd != leafEndVertices.end(); ++itLeafEnd)
			vertices.push_back(SkeletonAlgorithm::Vertex(*itLeafEnd, SkeletonAlgorithm::LEAF_END, vertexCount++));
	}

	//
	std::map<local::LineSegment, int> currLineSegments, lineSegments, overlappedLineSegments;  // Pair of line segment & starting vertex ID.
	for (int r = bottomRow - 1; r >= 0; --r)
	{
		currLineSegments.clear();
		lineSegments.clear();

		local::findLineSegmentsInRow(bw, r, 0, bw.cols - 1, lineSegments);
		if (prevLineSegments.empty() && lineSegments.empty()) continue;

		for (std::map<local::LineSegment, int>::iterator itPrevLineMap = prevLineSegments.begin(); itPrevLineMap != prevLineSegments.end(); ++itPrevLineMap)
		{
			overlappedLineSegments.clear();

			for (std::map<local::LineSegment, int>::iterator itLineMap = lineSegments.begin(); itLineMap != lineSegments.end(); ++itLineMap)
				if (itPrevLineMap->first.isOverlapped(itLineMap->first, skeletonOverlapMargin))
				{
					itLineMap->second = itPrevLineMap->second;
					overlappedLineSegments.insert(*itLineMap);
				}

			if (overlappedLineSegments.empty())  // Last appeared point.
			{
				std::list<SkeletonAlgorithm::Vertex>::iterator itJoin = std::find_if(vertices.begin(), vertices.end(), local::PrContainVertexInLineSegment(itPrevLineMap->first, SkeletonAlgorithm::JOIN));
				if (vertices.end() != itJoin)  // Join -> leaf-concave.
				{
					itJoin->type = SkeletonAlgorithm::LEAF_CONCAVE;
				}
				else
				{
					std::list<SkeletonAlgorithm::Vertex>::iterator itLeafEnd = std::find_if(vertices.begin(), vertices.end(), local::PrContainVertexInLineSegment(itPrevLineMap->first, SkeletonAlgorithm::LEAF_END));
					if (vertices.end() == itLeafEnd)  // Leaf-concave.
					{
						edges.push_back(std::make_pair(itPrevLineMap->second, vertexCount));
						vertices.push_back(SkeletonAlgorithm::Vertex(itPrevLineMap->first.getMidPoint(), SkeletonAlgorithm::LEAF_CONCAVE, vertexCount));
						itPrevLineMap->second = vertexCount++;
					}
					else  // Leaf-end.
					{
						edges.push_back(std::make_pair(itPrevLineMap->second, itLeafEnd->id));
						itPrevLineMap->second = itLeafEnd->id;
					}
				}
			}
			else if (overlappedLineSegments.size() > 1)  // Branch.
			{
				for (std::map<local::LineSegment, int>::iterator itLineMap = overlappedLineSegments.begin(); itLineMap != overlappedLineSegments.end(); ++itLineMap)
					itLineMap->second = vertexCount;

				edges.push_back(std::make_pair(itPrevLineMap->second, vertexCount));
				vertices.push_back(SkeletonAlgorithm::Vertex(itPrevLineMap->first.getMidPoint(), SkeletonAlgorithm::BRANCH, vertexCount));
				itPrevLineMap->second = vertexCount++;
			}

			currLineSegments.insert(overlappedLineSegments.begin(), overlappedLineSegments.end());
		}

		for (auto line : lineSegments)
		{
			overlappedLineSegments.clear();

			for (auto prevLine : prevLineSegments)
			{
				if (line.first.isOverlapped(prevLine.first, skeletonOverlapMargin))
					overlappedLineSegments.insert(prevLine);
			}

			if (overlappedLineSegments.empty())  // First appeared point.
			{
				std::list<SkeletonAlgorithm::Vertex>::iterator itLeafEnd = std::find_if(vertices.begin(), vertices.end(), local::PrContainVertexInLineSegment(line.first, SkeletonAlgorithm::LEAF_END));
				if (vertices.end() == itLeafEnd)  // Leaf-convex.
				{
					vertices.push_back(SkeletonAlgorithm::Vertex(line.first.getMidPoint(), SkeletonAlgorithm::LEAF_CONVEX, vertexCount));
					currLineSegments[line.first] = vertexCount++;
				}
				else  // Leaf-end.
				{
					currLineSegments[line.first] = itLeafEnd->id;
				}
			}
			else if (overlappedLineSegments.size() > 1)  // Join or leaf-concave points.
			{
				std::list<SkeletonAlgorithm::Vertex>::iterator itLeafConcave = std::find_if(vertices.begin(), vertices.end(), local::PrContainVertexInLineSegment(line.first, SkeletonAlgorithm::LEAF_CONCAVE));
				if (vertices.end() == itLeafConcave)  // Join.
				{
					for (auto overlappedLine : overlappedLineSegments)
						edges.push_back(std::make_pair(overlappedLine.second, vertexCount));
					vertices.push_back(SkeletonAlgorithm::Vertex(line.first.getMidPoint(), SkeletonAlgorithm::JOIN, vertexCount));
					currLineSegments[line.first] = vertexCount++;
				}
				else  // Leaf-concave.
				{
					//itLeafConcave->type = SkeletonAlgorithm::LEAF_JOIN;
				}
			}
		}

#if 0
		// Show line segments.
		{
			//std::cout << "#line segments = " << currLineSegments.size() << ", #leaf-concave vertices = " << leafConcaveVertices.size() << ", #branch vertices = " << branchVertices.size() << ", #join vertices = " << joinVertices.size() << std::endl;

			cv::Mat tmp;
			cv::cvtColor(bw, tmp, cv::COLOR_GRAY2BGR);
			cv::line(tmp, cv::Point(0, r), cv::Point(tmp.cols - 1, r), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
			for (auto line : currLineSegments)
				cv::line(tmp, line.first.getPt1(), line.first.getPt2(), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

			cv::imshow("Skeleton - Line segments", tmp);
			cv::waitKey(0);
		}
#endif

		prevLineSegments.swap(currLineSegments);
	}

#if 0
	// Show vertices: for checking.
	cv::Mat rgb;
	cv::cvtColor(bw, rgb, cv::COLOR_GRAY2BGR);
	showVertices(rgb, "Plant Graph - Vertices by Upward Line Scanning", vertices);
#endif
}

/*static*/ void SkeletonAlgorithm::findAllVerticesByTracingSkeleton(const cv::Mat& bw, const cv::Point& startPt, std::set<int>& visited, int &vertexId, std::list<SkeletonAlgorithm::Vertex>& vertices)
{
	// Breadth-first search.

	std::queue<cv::Point> que;
	que.push(startPt);

	while (!que.empty())
	{
		const cv::Point& u = que.front();
		que.pop();

		if (visited.end() == visited.find(u.y * bw.cols + u.x))  // If unvisited.
		{
			visited.insert(u.y * bw.cols + u.x);  // Set visited.

			SkeletonAlgorithm::VertexType vertexType;
			if (checkIfVertex(bw, u, &vertexType))
				vertices.push_back(SkeletonAlgorithm::Vertex(u, vertexType, vertexId++));

			const std::list<cv::Point>& neighbors = local::getNeighborsUsingNeighborCoords(bw, u);
			for (std::list<cv::Point>::const_iterator cit = neighbors.begin(); cit != neighbors.end(); ++cit)
				if (visited.end() == visited.find(cit->y * bw.cols + cit->x))  // If unvisited.
					que.push(*cit);
		}
	}
}

/*static*/ void SkeletonAlgorithm::findAdjacentVerticesByTracingSkeleton(const cv::Mat& bw, const cv::Point& currPt, std::set<int>& visited, std::list<SkeletonAlgorithm::Vertex>& adjacents)
{
	// Depth-first search.

	// FIXME [improve] >> Two slow implementation.

	if (visited.end() == visited.find(currPt.y * bw.cols + currPt.x))  // If unvisited.
		visited.insert(currPt.y * bw.cols + currPt.x);  // Set visited.

	// Get neighbors.
	const std::list<cv::Point>& neighbors = local::getNeighborsUsingNeighborCoords(bw, currPt);

	// Visit self.
	{
		std::list<SkeletonAlgorithm::Vertex> vertexNeighbors;
		for (std::list<cv::Point>::const_iterator cit = neighbors.begin(); cit != neighbors.end(); ++cit)
			if (visited.end() == visited.find(cit->y * bw.cols + cit->x))  // If unvisited.
			{
				SkeletonAlgorithm::VertexType vertexType;
				if (checkIfVertex(bw, *cit, &vertexType))
					vertexNeighbors.push_back(SkeletonAlgorithm::Vertex(*cit, vertexType, -1));
			}

		// Remove all vertices but the nearest one in the same edge group.
		switch (vertexNeighbors.size())
		{
		case 0:
			break;
		case 1:
			adjacents.insert(adjacents.end(), vertexNeighbors.begin(), vertexNeighbors.end());
			return;
		default:
		{
			// Get edge groups.
			std::map<int, int> edgeGroupIds;
			const size_t numEdgeGroupIds = constructEdgeGroups(bw, currPt, edgeGroupIds);  // Use one-based index.

			std::vector<std::list<SkeletonAlgorithm::Vertex> > edgeGroups(numEdgeGroupIds);
			for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = vertexNeighbors.begin(); cit != vertexNeighbors.end(); ++cit)
			{
				const int edgeGroupId = edgeGroupIds[(cit->pt.y - currPt.y + 1) * 3 + (cit->pt.x - currPt.x + 1)];
				if (edgeGroupId > 0)
					edgeGroups[edgeGroupId - 1].push_back(*cit);
				else assert(false);
			}

			std::list<SkeletonAlgorithm::Vertex> filteredVertexNeighbors;
			for (std::vector<std::list<SkeletonAlgorithm::Vertex> >::const_iterator citEdgeGroup = edgeGroups.begin(); citEdgeGroup != edgeGroups.end(); ++citEdgeGroup)
			{
				if (citEdgeGroup->empty()) continue;

				if (1 == citEdgeGroup->size()) filteredVertexNeighbors.push_back(citEdgeGroup->front());
				else  // When multiple vertices exist in the same edge groups.
				{
					std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = citEdgeGroup->begin();
					// Choose the nearest vertex.
					int minDist = std::abs(cit->pt.y - currPt.y) + std::abs(cit->pt.x - currPt.x);
					std::list<SkeletonAlgorithm::Vertex>::const_iterator citMin = cit;
					for (++cit; cit != citEdgeGroup->end(); ++cit)
					{
						const int dist = std::abs(cit->pt.y - currPt.y) + std::abs(cit->pt.x - currPt.x);
						if (dist < minDist)
						{
							minDist = dist;
							citMin = cit;
						}
					}
					filteredVertexNeighbors.push_back(*citMin);
				}
			}

			assert(!filteredVertexNeighbors.empty());

			adjacents.insert(adjacents.end(), filteredVertexNeighbors.begin(), filteredVertexNeighbors.end());
		}
		return;
		}
	}

	// Traverse neighbors.
	for (std::list<cv::Point>::const_iterator cit = neighbors.begin(); cit != neighbors.end(); ++cit)
		if (visited.end() == visited.find(cit->y * bw.cols + cit->x))  // If unvisited.
			findAdjacentVerticesByTracingSkeleton(bw, *cit, visited, adjacents);
}

/*static*/ void SkeletonAlgorithm::findVerticesAndEdgesByTracingSkeleton(const cv::Mat& bw, const SkeletonAlgorithm::Vertex& curr, std::set<int>& visited, int& vertexId, std::list<SkeletonAlgorithm::Vertex>& vertices, std::list<std::pair<const int, const int> >& edges)
{
#if 0
	// Depth-first search.

	//if (visited.end() == visited.find(curr.y * bw.cols + curr.x))  // If unvisited.
	//	visited.insert(curr.y * bw.cols + curr.x);  // Set visited.

	// Visit self.
	std::list<SkeletonAlgorithm::Vertex> adjacents;
	findAdjacentVerticesByTracingSkeleton(bw, curr.pt, visited, adjacents);
	for (std::list<SkeletonAlgorithm::Vertex>::iterator it = adjacents.begin(); it != adjacents.end(); ++it)
	{
		it->id = vertexId++;
		edges.push_back(std::make_pair(curr.id, it->id));
		vertices.push_back(*it);
	}

	// Traverse adjacent vertices.
	for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = adjacents.begin(); cit != adjacents.end(); ++cit)
		if (visited.end() == visited.find(cit->pt.y * bw.cols + cit->pt.x))  // If unvisited.
			findVerticesAndEdgesByTracingSkeleton(bw, *cit, visited, vertexId, vertices, edges);
#else
	// Breadth-first search.

	std::queue<SkeletonAlgorithm::Vertex> que;
	que.push(curr);

	while (!que.empty())
	{
		const SkeletonAlgorithm::Vertex& u = que.front();
		que.pop();

		if (visited.end() == visited.find(u.pt.y * bw.cols + u.pt.x))  // If unvisited.
		{
			//visited.insert(u.pt.y * bw.cols + u.pt.x);  // Set visited.

			// Visit self.
			std::list<SkeletonAlgorithm::Vertex> adjacents;
			findAdjacentVerticesByTracingSkeleton(bw, u.pt, visited, adjacents);
			for (std::list<SkeletonAlgorithm::Vertex>::iterator it = adjacents.begin(); it != adjacents.end(); ++it)
			{
				it->id = vertexId++;
				edges.push_back(std::make_pair(u.id, it->id));
				vertices.push_back(*it);
			}

			//
			for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = adjacents.begin(); cit != adjacents.end(); ++cit)
				if (visited.end() == visited.find(cit->pt.y * bw.cols + cit->pt.x))  // If unvisited.
					que.push(*cit);
		}
	}
#endif
}

/*static*/ void SkeletonAlgorithm::findVerticesAndEdgesByTracingSkeleton(const cv::Mat& bw, const int rootVerticesLocation, std::list<SkeletonAlgorithm::Vertex>& vertices, std::list<std::pair<const int, const int> >& edges)
{
	const int bottomRow = bw.rows - rootVerticesLocation;
	int vertexCount = 0;

	std::map<local::LineSegment, int> prevLineSegments;  // Pair of line segment & starting vertex ID.
	local::findLineSegmentsInRow(bw, bottomRow, 0, bw.cols - 1, prevLineSegments);
	for (std::map<local::LineSegment, int>::iterator itLineMap = prevLineSegments.begin(); itLineMap != prevLineSegments.end(); ++itLineMap)
	{
		//itLineMap->second = vertexCount;
		vertices.push_back(SkeletonAlgorithm::Vertex(itLineMap->first.getMidPoint(), SkeletonAlgorithm::ROOT, vertexCount++));
	}

/*
	// Find leaf-end vertices w/o information on relationship with other vertices (more correct).
	{
		std::list<cv::Point> leafEndVertices;
		findLeafEndVertices(bw, leafEndVertices);
		for (std::list<cv::Point>::const_iterator itLeafEnd = leafEndVertices.begin(); itLeafEnd != leafEndVertices.end(); ++itLeafEnd)
		vertices.push_back(SkeletonAlgorithm::Vertex(*itLeafEnd, SkeletonAlgorithm::LEAF_END, vertexCount++));
	}
*/

#if 0
	// Find internal vertices w/o information on relationship with other vertices (more correct).
	{
		std::list<cv::Point> internalVertices;
		findInternalVertices(bw, internalVertices);  // FIXME [correct] >> Not correctly working.
		for (std::list<cv::Point>::const_iterator itInternal = internalVertices.begin(); itInternal != internalVertices.end(); ++itInternal)
			// FIXME [improve] >> Change INTERNAL to more specific vertex type.
			vertices.push_back(SkeletonAlgorithm::Vertex(*itInternal, SkeletonAlgorithm::INTERNAL, vertexCount++));
	}
#elif 0
	{
		const std::list<SkeletonAlgorithm::Vertex> rootAndLeafVertices(vertices);
		std::set<int> visited;
		for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = rootAndLeafVertices.begin(); cit != rootAndLeafVertices.end(); ++cit)
			findAllVerticesByTracingSkeleton(bw, cit->pt, visited, vertexCount, vertices);

		// FIXME [implement] >> Find edges from vertices.

		//visited.clear();
		//std::map<int, int> vertexMap;
		//for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = vertices.begin(); cit != vertices.end(); ++cit)
		//	vertexMap.insert(std::make_pair(cit->pt.y * bw.cols + cit->pt.x, cit->id));
		//for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = rootAndLeafVertices.begin(); cit != rootAndLeafVertices.end(); ++cit)
		//	findEdgesByTracingSkeleton(bw, cit->pt, vertexMap, visited, edges);
	}
#else
	{
		const std::list<SkeletonAlgorithm::Vertex> rootAndLeafVertices(vertices);
		for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = rootAndLeafVertices.begin(); cit != rootAndLeafVertices.end(); ++cit)
		{
			std::set<int> visited;
			findVerticesAndEdgesByTracingSkeleton(bw, *cit, visited, vertexCount, vertices, edges);
		}
	}
#endif

#if 0
	// Show vertices: for checking.
	cv::Mat rgb;
	cv::cvtColor(bw, rgb, cv::COLOR_GRAY2BGR);
	showVertices(rgb, "Plant Graph - Vertices by Tracing Skeleton", vertices);
#endif
}

/*static*/ bool SkeletonAlgorithm::findSkeletalPathBetweenTwoVertices(const cv::Mat& bw, const cv::Point& curr, const cv::Point& end, std::set<int>& visited, std::list<cv::Point>& path)
{
	// Depth-first search.

	if (visited.end() == visited.find(curr.y * bw.cols + curr.x))  // If unvisited.
		visited.insert(curr.y * bw.cols + curr.x);  // Set visited.

	// Get neighbors.
	const std::list<cv::Point>& neighbors = local::getNeighborsUsingNeighborCoords(bw, curr);

	// Visit self.
	path.push_back(curr);
	bool hasVertex = false;
	for (std::list<cv::Point>::const_iterator cit = neighbors.begin(); cit != neighbors.end(); ++cit)
	{
		if (visited.end() == visited.find(cit->y * bw.cols + cit->x))  // If unvisited.
		{
			if (checkIfVertex(bw, *cit, NULL))
			{
				if (*cit == end)
				{
					path.push_back(*cit);
					return true;
				}
				hasVertex = true;
			}
		}
	}
	if (hasVertex)
		return false;

	// Traverse neighbors.
	for (std::list<cv::Point>::const_iterator cit = neighbors.begin(); cit != neighbors.end(); ++cit)
		if (visited.end() == visited.find(cit->y * bw.cols + cit->x))  // If unvisited.
		{
			if (findSkeletalPathBetweenTwoVertices(bw, *cit, end, visited, path))
				return true;
			else path.pop_back();
		}

	return false;
}

/*static*/ bool SkeletonAlgorithm::findSkeletalPathBetweenTwoVertices(const cv::Mat& bw, const cv::Point& start, const cv::Point& end, std::list<cv::Point>& path)
{
	if (!bw.at<unsigned char>(start.y, start.x) || !bw.at<unsigned char>(end.y, end.x))
		return false;

	std::set<int> visited;
	findSkeletalPathBetweenTwoVertices(bw, start, end, visited, path);
}

}  // namespace swl
