#include "swl/Config.h"
#include "swl/machine_vision/SkeletonAlgorithm.h"
#include <algorithm>
#include <queue>
#include <string>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

const size_t NUM_NEIGHBORS = 8;
const size_t NEIGHBOR_INDICES[] = { 3, 2, 1, 4, -1, 0, 5, 6, 7 };
const cv::Point NEIGHBOR_COORDINATES[] = { cv::Point(1, 0), cv::Point(1, -1), cv::Point(0, -1), cv::Point(-1, -1), cv::Point(-1, 0), cv::Point(-1, 1), cv::Point(0, 1), cv::Point(1, 1) };
const size_t NEIGHBORHOOD_SIZE = 3;  // 3x3 kernel.
const cv::Mat NEIGHBORHOOD_KERNEL = (cv::Mat_<short>(NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE) <<
	1, 1, 1,
	1, 0, 1,
	1, 1, 1);

bool isNeighbor(const cv::Point& neighbor, const cv::Point& center)
{
	return neighbor != center && std::abs(neighbor.x - center.x) <= 1 && std::abs(neighbor.y - center.y) <= 1;
}

size_t getNeighborId(const cv::Point& neighbor, const cv::Point& center)
{
	return NEIGHBOR_INDICES[(neighbor.y - center.y + 1) * NEIGHBORHOOD_SIZE + (neighbor.x - center.x + 1)];
}

std::list<cv::Point> getNeighborsUsingNeighborCoords(const cv::Mat& bw, const cv::Point& center)
{
	std::list<cv::Point> neighbors;
	for (size_t i = 0; i < NUM_NEIGHBORS; ++i)
	{
		const cv::Point neigh(center + NEIGHBOR_COORDINATES[i]);
		if (neigh.x >= 0 && neigh.x < bw.cols && neigh.y >= 0 && neigh.y < bw.rows && (int)bw.at<unsigned char>(neigh.y, neigh.x))
			neighbors.push_back(neigh);
	}

	return neighbors;
}

void getNeighborsUsingNeighborCoords(const cv::Mat& bw, const cv::Point& center, bool* neighborFlags)
{
	// DbC [precondindion] {required} >> neighbors is an array of size NUM_NEIGHBORS.

	for (size_t i = 0; i < NUM_NEIGHBORS; ++i)
	{
		const cv::Point neigh(center + NEIGHBOR_COORDINATES[i]);
		neighborFlags[i] = (neigh.x >= 0 && neigh.x < bw.cols && neigh.y >= 0 && neigh.y < bw.rows && (int)bw.at<unsigned char>(neigh.y, neigh.x));
	}
}

struct  PrVertexComparator
{
public:
	PrVertexComparator(const swl::SkeletonAlgorithm::Vertex& vtx)
		: vtx_(vtx)
	{}

	bool operator()(const swl::SkeletonAlgorithm::Vertex& rhs) const
	{
		return vtx_.pt == rhs.pt;
	}

private:
	const swl::SkeletonAlgorithm::Vertex& vtx_;
};

struct PrContainPointInLineSegment
{
public:
	explicit PrContainPointInLineSegment(const swl::SkeletonAlgorithm::LineSegment& line)
		: line_(line)
	{}

	bool operator()(const cv::Point& pt) const
	{
		return line_.isContained(pt);
	}

private:
	const swl::SkeletonAlgorithm::LineSegment& line_;
};

struct PrContainVertexInLineSegment
{
public:
	explicit PrContainVertexInLineSegment(const swl::SkeletonAlgorithm::LineSegment& line, const swl::SkeletonAlgorithm::VertexType type)
		: line_(line), type_(type)
	{}

	bool operator()(const swl::SkeletonAlgorithm::Vertex& vtx) const
	{
		return line_.isContained(vtx.pt) && type_ == vtx.type;
	}

private:
	const swl::SkeletonAlgorithm::LineSegment& line_;
	const swl::SkeletonAlgorithm::VertexType type_;
};

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

SkeletonAlgorithm::LineSegment::LineSegment(const cv::Point& pt1, const cv::Point& pt2)
: pt1_(pt1), pt2_(pt2)
{}

bool SkeletonAlgorithm::LineSegment::operator<(const SkeletonAlgorithm::LineSegment& rhs) const
{
	// FIXME [check] >>
	return pt1_.x < rhs.pt1_.x && pt2_.x < rhs.pt2_.x;
}

bool SkeletonAlgorithm::LineSegment::isOverlapped(const SkeletonAlgorithm::LineSegment& rhs, const int skeletonOverlapMargin) const
{
	return !(pt1_.x - skeletonOverlapMargin > rhs.pt2_.x || pt2_.x + skeletonOverlapMargin < rhs.pt1_.x);
}

bool SkeletonAlgorithm::LineSegment::isContained(const cv::Point& pt) const
{
	return std::abs(pt1_.y - pt.y) < 1 && pt1_.x <= pt.x && pt.x <= pt2_.x;
}

/*static*/ void SkeletonAlgorithm::countNeighbors(const cv::Mat& skeleton_bw, cv::Mat& neighborCounts)
{
	// FIXME [improve] >> Do filtering for non-zero pixels only.
	cv::Mat tmp(cv::Mat::zeros(skeleton_bw.size(), skeleton_bw.type()));
	tmp.setTo(cv::Scalar::all(1), skeleton_bw > 0);
	cv::filter2D(tmp, neighborCounts, CV_8UC1, local::NEIGHBORHOOD_KERNEL);
	neighborCounts.setTo(cv::Scalar::all(0), skeleton_bw == 0);
}

/*static*/ void SkeletonAlgorithm::computeNeighborGroups(const cv::Mat& skeleton_bw, cv::Mat& neighborGroupCounts, std::map<int, std::vector<int> >* neighborGroupsOfInternalVertices /*= NULL*/)
{
	// FIXME [improve] >> Do filtering for non-zero pixels only.

	// Isolated vertices surrounded by black pixels are removed.
	cv::Mat neighborCounts;
	countNeighbors(skeleton_bw, neighborCounts);
	std::vector<cv::Point> candidates;
	cv::findNonZero(neighborCounts, candidates);

	neighborGroupCounts = cv::Mat::zeros(skeleton_bw.size(), CV_8UC1);
	if (neighborGroupsOfInternalVertices)
	{
		std::vector<int> neighborGroupIds(local::NUM_NEIGHBORS, -1);
		for (const auto& candid : candidates)
		{
			// Vertices surrounded by white pixels are removed.
			const size_t numNeighborGroups = getNeighborGroups(skeleton_bw, candid, neighborGroupIds);  // One-based index.
			neighborGroupCounts.at<unsigned char>(candid.y, candid.x) = (unsigned char)numNeighborGroups;
			if (numNeighborGroups > 2)  // Internal vertex.
				neighborGroupsOfInternalVertices->insert(std::make_pair(candid.y * skeleton_bw.cols + candid.x, neighborGroupIds));
		}
	}
	else
	{
		for (const auto& candid : candidates)
		{
			// Vertices surrounded by white pixels are removed.
			neighborGroupCounts.at<unsigned char>(candid.y, candid.x) = (unsigned char)countNeighborGroups(skeleton_bw, candid);  // One-based index.
		}
	}
}

/*static*/ void SkeletonAlgorithm::findVerticesByNeighborGroup(const cv::Mat& neighborGroupCounts, std::list<SkeletonAlgorithm::Vertex>& vertices, int startId /*= 0*/)
{
	std::vector<cv::Point> leafEndVertices, internalVertices;
	cv::findNonZero(1 == neighborGroupCounts, leafEndVertices);  // Root and leaf vertices.
	cv::findNonZero(neighborGroupCounts > 2, internalVertices);  // Internal vertices.

	for (const auto& pt : leafEndVertices)
		vertices.push_back(SkeletonAlgorithm::Vertex(pt, SkeletonAlgorithm::LEAF_END, startId++));
	for (const auto& pt : internalVertices)
		vertices.push_back(SkeletonAlgorithm::Vertex(pt, SkeletonAlgorithm::INTERNAL, startId++));
}

/*static*/ void SkeletonAlgorithm::findLeafEndVertices(const cv::Mat& skeleton_bw, std::list<cv::Point>& leafEndVertices)
{
	// DbC [precondition] {required} >> The thickness of skeletons in an input image skeleton_bw is 1.

	// FIXME [improve] >> Do filtering for non-zero pixels only.
	cv::Mat result;
	cv::filter2D(skeleton_bw, result, CV_16S, local::NEIGHBORHOOD_KERNEL);
	result.setTo(cv::Scalar::all(0), skeleton_bw == 0);

	// FIXME [correct] >> A pixel with more than two non-zero neighbors might be a leaf vertex.
	result.setTo(cv::Scalar::all(0), result != 255);  // Single point.

	cv::Mat result_uchar;
	result.convertTo(result_uchar, CV_8UC1);

	std::vector<cv::Point> pixels;
	cv::findNonZero(result_uchar, pixels);

	leafEndVertices.assign(pixels.begin(), pixels.end());
}

/*static*/ size_t SkeletonAlgorithm::countNeighborGroups(const cv::Mat& skeleton_bw, const cv::Point& center)
	// 0 <= neighborId < NUM_NEIGHBORS.
	// If neighborGroupIds[neighborId] == 0, no neighbor (black pixel).
	// If neighborGroupIds[neighborId] > 0, neighbor group ID (white pixel). One-based index.
	// If neighborGroupIds[neighborId] < 0, no pixel (boundary).
{
	// DbC [precondition] {required} >> Assume 8 neighbors.
	//	If we use arbitrary neighbors, we should consider connected components.

	unsigned char prevVal, firstVal, lastVal;
	bool hasPrevPt = false, hasFirstVal = false, hasLastVal = false;
	{
		const cv::Point pt(center + local::NEIGHBOR_COORDINATES[local::NUM_NEIGHBORS - 1]);
		if (pt.x >= 0 && pt.x < skeleton_bw.cols && pt.y >= 0 && pt.y < skeleton_bw.rows)
		{
			lastVal = skeleton_bw.at<unsigned char>(pt.y, pt.x);
			hasLastVal = true;

			prevVal = lastVal;
			hasPrevPt = true;
		}
	}

	size_t numNeighborGroups = 0;
	for (size_t neighborId = 0; neighborId < local::NUM_NEIGHBORS; ++neighborId)
	{
		const cv::Point neighbor(center + local::NEIGHBOR_COORDINATES[neighborId]);
		if (neighbor.x < 0 || neighbor.x >= skeleton_bw.cols || neighbor.y < 0 || neighbor.y >= skeleton_bw.rows)
		{
			hasPrevPt = false;
			continue;
		}

		const unsigned char currVal = skeleton_bw.at<unsigned char>(neighbor.y, neighbor.x);
		if (0 == neighborId)
		{
			firstVal = currVal;
			hasFirstVal = true;
		}

		if (currVal)
		{
			if (0 == neighborId) ++numNeighborGroups;  // New neighbor group.
			else
			{
				if (hasPrevPt)
				{
					if (currVal != prevVal) ++numNeighborGroups;  // New neighbor group.
				}
				else ++numNeighborGroups;  // New neighbor group.
			}
		}

		prevVal = currVal;
		hasPrevPt = true;
	}

	if (hasFirstVal && hasLastVal && firstVal == lastVal)
		--numNeighborGroups;

	return numNeighborGroups;
}

/*static*/ size_t SkeletonAlgorithm::getNeighborGroups(const cv::Mat& skeleton_bw, const cv::Point& center, std::vector<int>& neighborGroupIds)
	// 0 <= neighborId < NUM_NEIGHBORS.
	// If neighborGroupIds[neighborId] == 0, no neighbor (black pixel).
	// If neighborGroupIds[neighborId] > 0, neighbor group ID (white pixel). One-based index.
	// If neighborGroupIds[neighborId] < 0, no pixel (boundary).
{
	// DbC [precondition] {required} >> Assume 8 neighbors.
	//	If we use arbitrary neighbors, we should consider connected components.

	unsigned char prevVal, firstVal, lastVal;
	bool hasPrevPt = false;
	{
		const cv::Point pt(center + local::NEIGHBOR_COORDINATES[local::NUM_NEIGHBORS - 1]);
		if (pt.x >= 0 && pt.x < skeleton_bw.cols && pt.y >= 0 && pt.y < skeleton_bw.rows)
		{
			lastVal = skeleton_bw.at<unsigned char>(pt.y, pt.x);
			prevVal = lastVal;
			hasPrevPt = true;
		}
	}

	size_t numNeighborGroups = 0;
	for (size_t neighborId = 0; neighborId < local::NUM_NEIGHBORS; ++neighborId)
	{
		const cv::Point neighbor(center + local::NEIGHBOR_COORDINATES[neighborId]);
		if (neighbor.x < 0 || neighbor.x >= skeleton_bw.cols || neighbor.y < 0 || neighbor.y >= skeleton_bw.rows)
		{
			neighborGroupIds[neighborId] = -1;  // Out of border (no pixel).
			hasPrevPt = false;
			continue;
		}

		const unsigned char currVal = skeleton_bw.at<unsigned char>(neighbor.y, neighbor.x);
		if (0 == neighborId) firstVal = currVal;

		if (currVal)
		{
			if (0 == neighborId) ++numNeighborGroups;  // New neighbor group.
			else
			{
				if (hasPrevPt)
				{
					if (currVal != prevVal) ++numNeighborGroups;  // New neighbor group.
				}
				else ++numNeighborGroups;  // New neighbor group.
			}
			neighborGroupIds[neighborId] = numNeighborGroups;
		}
		else
			neighborGroupIds[neighborId] = 0;  // No neighbor.

		prevVal = currVal;
		hasPrevPt = true;
	}

	//assert(local::NUM_NEIGHBORS == neighborGroupIds.size());

	if (neighborGroupIds[0] > 0 && neighborGroupIds[local::NUM_NEIGHBORS - 1] > 0)
	{
		if (firstVal == lastVal)
		{
			for (std::vector<int>::iterator it = neighborGroupIds.begin(); it != neighborGroupIds.end(); ++it)
				if (*it == numNeighborGroups) *it = 1;  // Change into the first neighbor group.
			--numNeighborGroups;
		}
	}

	return numNeighborGroups;
}

// REF [function] >> getNeighborGroups().
/*static*/ bool SkeletonAlgorithm::isInTheSameNeighborGroup(const std::vector<int>& neighborGroupIds, const size_t neighborId1, const size_t neighborId2)
{
	return neighborGroupIds[neighborId1] > 0 && neighborGroupIds[neighborId1] == neighborGroupIds[neighborId2];
}

// REF [function] >> getNeighborGroups().
/*static*/ bool SkeletonAlgorithm::isOnBorder(const std::vector<int>& neighborGroupIds)
{
	for (std::vector<int>::const_iterator cit = neighborGroupIds.begin(); cit != neighborGroupIds.end(); ++cit)
		if (*cit < 0) return true;
	return false;
}

// REF [function] >> getNeighborGroups().
/*static*/ bool SkeletonAlgorithm::isSurroundedBySingleNeighborGroup(const std::vector<int>& neighborGroupIds)
{
	int id = -1;
	for (std::vector<int>::const_iterator cit = neighborGroupIds.begin(); cit != neighborGroupIds.end(); ++cit)
		if (0 == *cit) return false;
	//else if (*cit > 0)
	//{
	//	if (-1 == id) id = *cit;
	//	else if (id != *cit) return false;
	//}
	return true;
	}

// REF [function] >> getNeighborGroups().
/*static*/ void SkeletonAlgorithm::getNeareastNeighborIndicesInEachNeighborGroup(const std::vector<int>& neighborGroupIds, std::vector<int>& nearestIndicesInEachNeighborGroup)
{
	// DbC [precondition] {required} >> Assume 8 neighbors.
	//	If we use arbitrary neighbors, we should consider connected components.

	nearestIndicesInEachNeighborGroup.clear();

	const int numNeighborGroups = *std::max_element(neighborGroupIds.begin(), neighborGroupIds.end());
	if (numNeighborGroups <= 0) return;  // No neighbor.

	nearestIndicesInEachNeighborGroup.resize(numNeighborGroups, -1);

	int startIdx = 0;
	if (neighborGroupIds[0] > 0)
	{
		for (int i = (int)local::NUM_NEIGHBORS - 1; i >= 0; --i)
			if (neighborGroupIds[i] <= 0)
			{
				startIdx = (i + 1) % local::NUM_NEIGHBORS;
				break;
			}
	}

	bool isInNeighborGroup = true, isEvenMinIndexFound = false;
	for (int i = 0; i < (int)local::NUM_NEIGHBORS; ++i)
	{
		const int idx = (startIdx + i) % local::NUM_NEIGHBORS;

		if (neighborGroupIds[idx] <= 0)
		{
			isInNeighborGroup = false;
			isEvenMinIndexFound = false;
		}
		else
		{
			// Even indices mean min values. It is likely to use L1 norm.
			if (!isInNeighborGroup)
			{
				isInNeighborGroup = true;
				nearestIndicesInEachNeighborGroup[neighborGroupIds[idx] - 1] = idx;  // One-based index.

				if (0 == idx % 2) isEvenMinIndexFound = true;
			}
			else if (0 == idx % 2)
			{
				if (isEvenMinIndexFound) assert(false);
				else
				{
					nearestIndicesInEachNeighborGroup[neighborGroupIds[idx] - 1] = idx;  // One-based index.
					isEvenMinIndexFound = true;
				}
			}
			else
			{
				if (!isEvenMinIndexFound)
					nearestIndicesInEachNeighborGroup[neighborGroupIds[idx] - 1] = idx;  // One-based index.
			}
		}
	}
}

/*static*/ void SkeletonAlgorithm::getNeareastNeighborsInEachNeighborGroup(const cv::Mat& bw, const cv::Point& center, std::list<cv::Point>& nearestNeighborsInEachNeighborGroup)
{
	// NOTICE [info] >> Two slow implementation.

	const std::list<cv::Point>& neighbors = local::getNeighborsUsingNeighborCoords(bw, center);
	if (neighbors.empty()) return;
	else if (1 == neighbors.size())
	{
		nearestNeighborsInEachNeighborGroup.push_back(neighbors.front());
		return;
	}
	else
	{
		std::vector<int> neighborGroupIds(local::NUM_NEIGHBORS, 0);
		for (const auto& neigh : neighbors)
			neighborGroupIds[local::getNeighborId(neigh, center)] = 1;
		bool isInNeighborGroup = false;
		int neighborGroupId = 1;
		for (size_t i = 0; i < local::NUM_NEIGHBORS; ++i)
		{
			if (neighborGroupIds[i] > 0)
			{
				if (!isInNeighborGroup) isInNeighborGroup = true;
				neighborGroupIds[i] = neighborGroupId;  // One-based index.
			}
			else if (isInNeighborGroup)
			{
				++neighborGroupId;
				isInNeighborGroup = false;
			}
		}

		if (neighborGroupIds[0] > 0 & neighborGroupIds[local::NUM_NEIGHBORS - 1] > 0)
			for (int i = (int)local::NUM_NEIGHBORS - 1; i >= 0; --i)
				if (neighborGroupIds[i] > 0)
				{
					neighborGroupIds[i] = 1;  // One-based index.
					break;
				}

		std::vector<int> nearestIndicesInEachNeighborGroup;
		getNeareastNeighborIndicesInEachNeighborGroup(neighborGroupIds, nearestIndicesInEachNeighborGroup);
		for (const auto& idx : nearestIndicesInEachNeighborGroup)
			nearestNeighborsInEachNeighborGroup.push_back(center + local::NEIGHBOR_COORDINATES[idx]);
	}
}

/*static*/ void SkeletonAlgorithm::getNeareastNeighborVerticesInEachNeighborGroup(const int numNeighborGroups, const std::vector<int>& neighborGroupIds, const std::list<cv::Point>& neighborsAsVertex, const cv::Point& curr, std::list<cv::Point>& neareastNeigborVertices)
{
	std::vector<std::list<cv::Point> > neighborGroups(numNeighborGroups);
	for (std::list<cv::Point>::const_iterator cit = neighborsAsVertex.begin(); cit != neighborsAsVertex.end(); ++cit)
	{
		const int& neighborGroupId = neighborGroupIds[local::getNeighborId(*cit, curr)];  // One-based index.
		if (neighborGroupId > 0)
			neighborGroups[neighborGroupId - 1].push_back(*cit);
		else assert(false);
	}

	for (std::vector<std::list<cv::Point> >::const_iterator citNeighborGroup = neighborGroups.begin(); citNeighborGroup != neighborGroups.end(); ++citNeighborGroup)
	{
		if (citNeighborGroup->empty()) continue;
		else if (1 == citNeighborGroup->size()) neareastNeigborVertices.push_back(citNeighborGroup->front());
		else  // When multiple vertices exist in the same neighbor groups.
		{
			std::list<cv::Point>::const_iterator cit = citNeighborGroup->begin();
			// Choose the nearest vertex.
			int minDist = std::abs(cit->x - curr.x) + std::abs(cit->y - curr.y);  // L1 norm.
			std::list<cv::Point>::const_iterator citMin = cit;
			for (++cit; cit != citNeighborGroup->end(); ++cit)
			{
				const int dist = std::abs(cit->x - curr.x) + std::abs(cit->y - curr.y);  // L1 norm.
				if (dist < minDist)
				{
					minDist = dist;
					citMin = cit;
				}
			}

			//if (neareastNeigborVertices.end() == std::find(neareastNeigborVertices.begin(), neareastNeigborVertices.end(), *citMin))
			neareastNeigborVertices.push_back(*citMin);
		}
	}
}

/*static*/ void SkeletonAlgorithm::getNeareastNeighborVerticesInEachNeighborGroup(const int numNeighborGroups, const std::vector<int>& neighborGroupIds, const std::list<SkeletonAlgorithm::Vertex>& neighborsAsVertex, const cv::Point& curr, std::list<SkeletonAlgorithm::Vertex>& neareastNeigborVertices)
{
	std::vector<std::list<SkeletonAlgorithm::Vertex> > neighborGroups(numNeighborGroups);
	for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = neighborsAsVertex.begin(); cit != neighborsAsVertex.end(); ++cit)
	{
		const int& neighborGroupId = neighborGroupIds[local::getNeighborId(cit->pt, curr)];  // One-based index.
		if (neighborGroupId > 0)
			neighborGroups[neighborGroupId - 1].push_back(*cit);
		else assert(false);
	}

	for (std::vector<std::list<SkeletonAlgorithm::Vertex> >::const_iterator citNeighborGroup = neighborGroups.begin(); citNeighborGroup != neighborGroups.end(); ++citNeighborGroup)
	{
		if (citNeighborGroup->empty()) continue;
		else if (1 == citNeighborGroup->size()) neareastNeigborVertices.push_back(citNeighborGroup->front());
		else  // When multiple vertices exist in the same neighbor groups.
		{
			std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = citNeighborGroup->begin();
			// Choose the nearest vertex.
			int minDist = std::abs(cit->pt.x - curr.x) + std::abs(cit->pt.y - curr.y);  // L1 norm.
			std::list<SkeletonAlgorithm::Vertex>::const_iterator citMin = cit;
			for (++cit; cit != citNeighborGroup->end(); ++cit)
			{
				const int dist = std::abs(cit->pt.x - curr.x) + std::abs(cit->pt.y - curr.y);  // L1 norm.
				if (dist < minDist)
				{
					minDist = dist;
					citMin = cit;
				}
			}

			//if (neareastNeigborVertices.end() == std::find_if(neareastNeigborVertices.begin(), neareastNeigborVertices.end(), local::PrVertexComparator(*citMin)))
			neareastNeigborVertices.push_back(*citMin);
				}
			}
	}

/*static*/ bool SkeletonAlgorithm::isVertex(const cv::Mat& skeleton_bw, const cv::Point& currPt, SkeletonAlgorithm::VertexType* vertexType /*= NULL*/)
	// If a point has 3 neighbor groups or more, it is a vertex.
{
#if 0
	// NOTICE [info] >> Too naive implementation.

	// Get the number of neighbor groups.
	unsigned char prevVal;
	bool hasPrevPt = false;
	{
		const cv::Point pt(currPt.x + local::NEIGHBOR_COORDINATES[local::NUM_NEIGHBORS - 1].x, currPt.y + local::NEIGHBOR_COORDINATES[local::NUM_NEIGHBORS - 1].y);
		if (pt.x >= 0 && pt.x < skeleton_bw.cols && pt.y >= 0 && pt.y < skeleton_bw.rows)
		{
			prevVal = skeleton_bw.at<unsigned char>(pt.y, pt.x);
			hasPrevPt = true;
		}
	}
	size_t numNeighborGroups = 0;
	int sumPixels = 0;
	int numValidPixels = 0;
	for (size_t i = 0; i < local::NUM_NEIGHBORS; ++i)
	{
		const cv::Point neighbor(currPt.x + local::NEIGHBOR_COORDINATES[i].x, currPt.y + local::NEIGHBOR_COORDINATES[i].y);
		if (neighbor.x < 0 || neighbor.x >= skeleton_bw.cols || neighbor.y < 0 || neighbor.y >= skeleton_bw.rows)
		{
			hasPrevPt = false;
			continue;
		}

		const unsigned char val = skeleton_bw.at<unsigned char>(neighbor.y, neighbor.x);
		sumPixels += val;
		++numValidPixels;
		if (val)
		{
			if (hasPrevPt)
			{
				if (val != prevVal) ++numNeighborGroups;
			}
			else ++numNeighborGroups;
		}

		prevVal = val;
		hasPrevPt = true;
	}

	switch (numNeighborGroups)
	{
	case 2:  // If the point is not a vertex.
		return false;
	case 0:
		if (prevVal == skeleton_bw.at<unsigned char>(currPt.y, currPt.x))
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
				if (!skeleton_bw.at<unsigned char>(currPt.y, currPt.x))
					return false;
				else  // Isolated vertex.
				{
					if (vertexType) *vertexType = SkeletonAlgorithm::ISOLATED;
					return true;
				}
				break;
			case 255:  // Surrounded by white pixels.
				if (skeleton_bw.at<unsigned char>(currPt.y, currPt.x))
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
	const unsigned char currVal = skeleton_bw.at<unsigned char>(currPt.y, currPt.x);
	if (!currVal) return false;

	// Get neighbor groups.
	std::vector<int> neighborGroupIds(local::NUM_NEIGHBORS, -1);
	const size_t numNeighborGroups = getNeighborGroups(skeleton_bw, currPt, neighborGroupIds);  // One-based index.

	switch (numNeighborGroups)
	{
	case 2:  // If the point is not a vertex.
		return false;
	case 0:  // Isolated vertex surrounded by black pixels.
		if (vertexType) *vertexType = SkeletonAlgorithm::ISOLATED;
		return true;
	case 1:
		if (isOnBorder(neighborGroupIds))  // If the current point is on the border of an image.
		{
			if (isSurroundedBySingleNeighborGroup(neighborGroupIds))  // Isolated(?) vertex surrounded by white pixels.
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

/*static*/ void SkeletonAlgorithm::constructGraphByUpwardLineScanning(const cv::Mat& skeleton_bw, const int rootVerticesRow, const int skeletonOverlapMargin, std::list<SkeletonAlgorithm::Vertex>& vertices, std::list<std::pair<const int, const int> >& edges)
{
	const int bottomRow = skeleton_bw.rows - rootVerticesRow;
	int vertexCount = 0;

	std::map<SkeletonAlgorithm::LineSegment, int> prevLineSegments;  // Pair of line segment & starting vertex ID.
	findLineSegmentsInRow(skeleton_bw, bottomRow, 0, skeleton_bw.cols - 1, prevLineSegments);
	for (std::map<SkeletonAlgorithm::LineSegment, int>::iterator itLineMap = prevLineSegments.begin(); itLineMap != prevLineSegments.end(); ++itLineMap)
	{
		itLineMap->second = vertexCount;
		vertices.push_back(SkeletonAlgorithm::Vertex(itLineMap->first.getMidPoint(), SkeletonAlgorithm::ROOT, vertexCount++));
	}

	// Find leaf vertices w/o information on relationship with other vertices (more correct).
	{
		std::list<cv::Point> leafEndVertices;
		findLeafEndVertices(skeleton_bw, leafEndVertices);
		for (std::list<cv::Point>::const_iterator itLeafEnd = leafEndVertices.begin(); itLeafEnd != leafEndVertices.end(); ++itLeafEnd)
			vertices.push_back(SkeletonAlgorithm::Vertex(*itLeafEnd, SkeletonAlgorithm::LEAF_END, vertexCount++));
	}

	//
	std::map<SkeletonAlgorithm::LineSegment, int> currLineSegments, lineSegments, overlappedLineSegments;  // Pair of line segment & starting vertex ID.
	for (int r = bottomRow - 1; r >= 0; --r)
	{
		currLineSegments.clear();
		lineSegments.clear();

		findLineSegmentsInRow(skeleton_bw, r, 0, skeleton_bw.cols - 1, lineSegments);
		if (prevLineSegments.empty() && lineSegments.empty()) continue;

		for (std::map<SkeletonAlgorithm::LineSegment, int>::iterator itPrevLineMap = prevLineSegments.begin(); itPrevLineMap != prevLineSegments.end(); ++itPrevLineMap)
		{
			overlappedLineSegments.clear();

			for (std::map<SkeletonAlgorithm::LineSegment, int>::iterator itLineMap = lineSegments.begin(); itLineMap != lineSegments.end(); ++itLineMap)
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
				for (std::map<SkeletonAlgorithm::LineSegment, int>::iterator itLineMap = overlappedLineSegments.begin(); itLineMap != overlappedLineSegments.end(); ++itLineMap)
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
			cv::cvtColor(skeleton_bw, tmp, cv::COLOR_GRAY2BGR);
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
	cv::cvtColor(skeleton_bw, rgb, cv::COLOR_GRAY2BGR);
	showVertices(rgb, "Plant Graph - Vertices by Upward Line Scanning", vertices);
#endif
}

/*static*/ void SkeletonAlgorithm::findAllVerticesByNeighborGroup(const cv::Mat& skeleton_bw, const cv::Point& start, std::set<int>& visited, int &vertexId, std::list<SkeletonAlgorithm::Vertex>& vertices)
{
	// Breadth-first search.

	std::queue<cv::Point> que;
	que.push(start);

	while (!que.empty())
	{
		const cv::Point& u = que.front();
		que.pop();

		if (visited.end() != visited.find(u.y * skeleton_bw.cols + u.x))  // If visited.
			continue;
		visited.insert(u.y * skeleton_bw.cols + u.x);  // Set visited.

														// Visit self.
		SkeletonAlgorithm::VertexType vertexType;
		if (isVertex(skeleton_bw, u, &vertexType))
			vertices.push_back(SkeletonAlgorithm::Vertex(u, vertexType, vertexId++));

		// Traverse neighbors.
		const std::list<cv::Point>& neighbors = local::getNeighborsUsingNeighborCoords(skeleton_bw, u);
		for (std::list<cv::Point>::const_iterator cit = neighbors.begin(); cit != neighbors.end(); ++cit)
			if (visited.end() == visited.find(cit->y * skeleton_bw.cols + cit->x))  // If unvisited.
				que.push(*cit);
	}
}

/*static*/ void SkeletonAlgorithm::findAdjacentVerticesByNeighborGroup(const cv::Mat& skeleton_bw, const cv::Point& curr, std::set<int>& visited, std::list<SkeletonAlgorithm::Vertex>& adjacents)
{
	// Depth-first search.

	// NOTICE [info] >> Two slow implementation.

	if (visited.end() != visited.find(curr.y * skeleton_bw.cols + curr.x))  // If visited.
		return;
	visited.insert(curr.y * skeleton_bw.cols + curr.x);  // Set visited.

															// Visit self.
															// Do nothing.

															// Traverse neighbors.
	std::list<cv::Point> nearestNeighborsInEachNeighborGroup;
	getNeareastNeighborsInEachNeighborGroup(skeleton_bw, curr, nearestNeighborsInEachNeighborGroup);
	for (const auto& neigh : nearestNeighborsInEachNeighborGroup)
		if (visited.end() == visited.find(neigh.y * skeleton_bw.cols + neigh.x))  // If unvisited.
		{
			SkeletonAlgorithm::VertexType vertexType;
			if (isVertex(skeleton_bw, neigh, &vertexType))  // If a vertex.
				adjacents.push_back(SkeletonAlgorithm::Vertex(neigh, vertexType, -1));
			else
				findAdjacentVerticesByNeighborGroup(skeleton_bw, neigh, visited, adjacents);
		}
}

/*static*/ void SkeletonAlgorithm::findAdjacentVerticesByNeighborGroup(const cv::Mat& neighborGroupCounts, const std::map<int, std::vector<int> >& neighborGroupsOfInternalVertices, const cv::Point& curr, std::set<int>& visited, std::list<cv::Point>& adjacents)
{
#if 0
	// Depth-first search.

	if (visited.end() != visited.find(curr.y * neighborGroupCounts.cols + curr.x))  // If visited.
		return;
	visited.insert(curr.y * neighborGroupCounts.cols + curr.x);  // Set visited.

																	// Visit self.
																	// Do nothing.

																	// Traverse neighbors.
	std::list<cv::Point> nearestNeighborsInEachNeighborGroup;
	getNeareastNeighborsInEachNeighborGroup(neighborGroupCounts, curr, nearestNeighborsInEachNeighborGroup);
	for (const auto& neigh : nearestNeighborsInEachNeighborGroup)
		if (visited.end() == visited.find(neigh.y * neighborGroupCounts.cols + neigh.x))  // If unvisited.
		{
			const int& numNeighborGroups = (int)neighborGroupCounts.at<unsigned char>(neigh.y, neigh.x);
			if (1 == numNeighborGroups || numNeighborGroups > 2)  // If a vertex.
				adjacents.push_back(neigh);
			else
				findAdjacentVerticesByNeighborGroup(neighborGroupCounts, neighborGroupsOfInternalVertices, neigh, visited, adjacents);
		}
#else
	// Breadth-first search.

	std::queue<cv::Point> que;
	que.push(curr);

	while (!que.empty())
	{
		const cv::Point& u = que.front();
		que.pop();

		if (visited.end() != visited.find(u.y * neighborGroupCounts.cols + u.x))  // If visited.
			continue;
		visited.insert(u.y * neighborGroupCounts.cols + u.x);  // Set visited.

																// Visit self.
																// Do nothing.

																// Traverse neighbors.
		std::list<cv::Point> nearestNeighborsInEachNeighborGroup;
		getNeareastNeighborsInEachNeighborGroup(neighborGroupCounts, u, nearestNeighborsInEachNeighborGroup);
		for (const auto& neigh : nearestNeighborsInEachNeighborGroup)
			if (visited.end() == visited.find(neigh.y * neighborGroupCounts.cols + neigh.x))  // If unvisited.
			{
				const int& numNeighborGroups = (int)neighborGroupCounts.at<unsigned char>(neigh.y, neigh.x);
				if (1 == numNeighborGroups || numNeighborGroups > 2)  // If a vertex.
					adjacents.push_back(neigh);
				else que.push(neigh);
			}
	}
#endif
}

/*static*/ bool SkeletonAlgorithm::getNeareastVertexByNeighborGroup(const cv::Mat& neighborGroupCounts, const std::map<int, std::vector<int> >& neighborGroupsOfInternalVertices, const cv::Point& curr, cv::Point& nearest)
{
	if (!(int)neighborGroupCounts.at<unsigned char>(curr.y, curr.x)) return false;

	// Breadth-first search.

	std::set<int> visited;

	std::queue<cv::Point> que;
	que.push(curr);

	while (!que.empty())
	{
		const cv::Point& u = que.front();
		que.pop();

		if (visited.end() != visited.find(u.y * neighborGroupCounts.cols + u.x))  // If visited.
			continue;
		visited.insert(u.y * neighborGroupCounts.cols + u.x);  // Set visited.

																//const std::list<cv::Point>& neighbors = local::getNeighborsUsingNeighborCoords(skeleton_bw, u);
		const std::list<cv::Point>& neighbors = local::getNeighborsUsingNeighborCoords(neighborGroupCounts, u);

		// Visit self.
		std::list<cv::Point> neighborsAsVertex;
		for (const auto& neigh : neighbors)
			if (visited.end() == visited.find(neigh.y * neighborGroupCounts.cols + neigh.x))  // If unvisited.
			{
				const int& numNeighborGroups = (int)neighborGroupCounts.at<unsigned char>(neigh.y, neigh.x);
				if (1 == numNeighborGroups || numNeighborGroups > 2)  // If a vertex.
					neighborsAsVertex.push_back(neigh);
			}

		// Traverse neighbors.
		if (neighborsAsVertex.empty())
		{
			for (const auto& neigh : neighbors)
				if (visited.end() == visited.find(neigh.y * neighborGroupCounts.cols + neigh.x))  // If unvisited.
					que.push(neigh);
		}
		else if (1 == neighborsAsVertex.size())  // Vertex found. No more traversal.
		{
			nearest = neighborsAsVertex.front();
			return true;
		}
		else  // Vertex found. No more traversal.
		{
			// Remove all vertices but the nearest one in the same neighbor group.

			// Get neighbor groups.
			const int& numNeighborGroups = (int)neighborGroupCounts.at<unsigned char>(u.y, u.x);
			std::map<int, std::vector<int> >::const_iterator citNeighborGroupIds = neighborGroupsOfInternalVertices.find(u.y * neighborGroupCounts.cols + u.x);  // One-based index.
			if (neighborGroupsOfInternalVertices.end() == citNeighborGroupIds)
				assert(false);

			std::vector<std::list<cv::Point> > neighborGroups(numNeighborGroups);
			for (std::list<cv::Point>::const_iterator cit = neighborsAsVertex.begin(); cit != neighborsAsVertex.end(); ++cit)
			{
				const int& neighborGroupId = citNeighborGroupIds->second[local::getNeighborId(*cit, u)];
				if (neighborGroupId > 0)
					neighborGroups[neighborGroupId - 1].push_back(*cit);
				else assert(false);
			}

			// TODO [improve] >> When there are multiple vertices, we should consider their distanes.
			int minDist = std::numeric_limits<int>::max();
			for (std::vector<std::list<cv::Point> >::const_iterator citNeighborGroup = neighborGroups.begin(); citNeighborGroup != neighborGroups.end(); ++citNeighborGroup)
			{
				if (citNeighborGroup->empty()) continue;
				else if (1 == citNeighborGroup->size())
				{
					const int dist = std::abs(citNeighborGroup->front().x - u.x) + std::abs(citNeighborGroup->front().y - u.y);  // L1 norm.
					if (dist < minDist)
					{
						minDist = dist;
						nearest = citNeighborGroup->front();
					}
				}
				else  // When multiple vertices exist in the same neighbor groups.
				{
					for (std::list<cv::Point>::const_iterator cit = citNeighborGroup->begin(); cit != citNeighborGroup->end(); ++cit)
					{
						const int dist = std::abs(cit->x - u.x) + std::abs(cit->y - u.y);  // L1 norm.
						if (dist < minDist)
						{
							minDist = dist;
							nearest = *cit;
						}
					}

					if (std::numeric_limits<int>::max() != minDist)
						return true;
				}
			}

			assert(false);
			return false;
		}
	}

	return false;
}

/*static*/ void SkeletonAlgorithm::findVerticesAndEdgesByNeighborGroup(const cv::Mat& skeleton_bw, const SkeletonAlgorithm::Vertex& curr, std::set<int>& visited, int& vertexId, std::list<SkeletonAlgorithm::Vertex>& vertices, std::list<std::pair<const int, const int> >& edges)
{
#if 0
	// Depth-first search.

	//if (visited.end() != visited.find(curr.y * skeleton_bw.cols + curr.x))  // If visited.
	//	return;
	//visited.insert(curr.y * skeleton_bw.cols + curr.x);  // Set visited.

	// Visit self.
	std::list<SkeletonAlgorithm::Vertex> adjacents;
	findAdjacentVerticesByNeighborGroup(skeleton_bw, curr.pt, visited, adjacents);
	for (std::list<SkeletonAlgorithm::Vertex>::iterator it = adjacents.begin(); it != adjacents.end(); ++it)
	{
		it->id = vertexId++;
		edges.push_back(std::make_pair(curr.id, it->id));
		vertices.push_back(*it);
	}

	// Traverse adjacent vertices.
	for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = adjacents.begin(); cit != adjacents.end(); ++cit)
		if (visited.end() == visited.find(cit->pt.y * skeleton_bw.cols + cit->pt.x))  // If unvisited.
			findVerticesAndEdgesByNeighborGroup(skeleton_bw, *cit, visited, vertexId, vertices, edges);
#else
	// Breadth-first search.

	std::queue<SkeletonAlgorithm::Vertex> que;
	que.push(curr);

	while (!que.empty())
	{
		const SkeletonAlgorithm::Vertex& u = que.front();
		que.pop();

		if (visited.end() != visited.find(u.pt.y * skeleton_bw.cols + u.pt.x))  // If visited.
			continue;
		//visited.insert(u.pt.y * skeleton_bw.cols + u.pt.x);  // Set visited.

		// Visit self.
		std::list<SkeletonAlgorithm::Vertex> adjacents;
		findAdjacentVerticesByNeighborGroup(skeleton_bw, u.pt, visited, adjacents);
		for (std::list<SkeletonAlgorithm::Vertex>::iterator it = adjacents.begin(); it != adjacents.end(); ++it)
		{
			it->id = vertexId++;
			edges.push_back(std::make_pair(u.id, it->id));
			vertices.push_back(*it);
		}

		// Traverse adjacent vertices.
		for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = adjacents.begin(); cit != adjacents.end(); ++cit)
			if (visited.end() == visited.find(cit->pt.y * skeleton_bw.cols + cit->pt.x))  // If unvisited.
				que.push(*cit);
	}
#endif
}

/*static*/ void SkeletonAlgorithm::findEdgesByNeighborGroup(const cv::Mat& neighborGroupCounts, const std::map<int, std::vector<int> >& neighborGroupsOfInternalVertices, const cv::Point& curr, std::set<int>& visited, const std::list<SkeletonAlgorithm::Vertex>& vertices, std::list<std::pair<const int, const int> >& edges)
{
#if 0
	// Depth-first search.

	//if (visited.end() != visited.find(curr.y * neighborGroupCounts.cols + curr.x))  // If visited.
	//	return;
	//visited.insert(curr.y * neighborGroupCounts.cols + curr.x);  // Set visited.

	// Visit self.
	std::list<cv::Point> adjacents;
	findAdjacentVerticesByNeighborGroup(neighborGroupCounts, neighborGroupsOfInternalVertices, curr.pt, visited, adjacents);
	for (std::list<SkeletonAlgorithm::Vertex>::iterator it = adjacents.begin(); it != adjacents.end(); ++it)
	{
		// FIXME [implement] >>
		edges.push_back(std::make_pair(uVtx.id, vertexMap[adj.y * neighborGroupCounts.cols + adj.x].id));
	}

	// Traverse adjacent vertices.
	for (const auto& adj : adjacents)
		if (visited.end() == visited.find(adj.y * neighborGroupCounts.cols + adj.x))  // If unvisited.
			findEdgesByNeighborGroup(neighborGroupCounts, neighborGroupsOfInternalVertices, adj, visited, vertices, edges);
#else
	// Breadth-first search.

	std::map<int, const SkeletonAlgorithm::Vertex> vertexMap;
	for (const auto& vtx : vertices)
		vertexMap.insert(std::make_pair(vtx.pt.y * neighborGroupCounts.cols + vtx.pt.x, vtx));

	std::queue<cv::Point> que;
	que.push(curr);

	while (!que.empty())
	{
		const cv::Point& u = que.front();
		que.pop();

		if (visited.end() != visited.find(u.y * neighborGroupCounts.cols + u.x))  // If visited.
			continue;
		//visited.insert(u.y * neighborGroupCounts.cols + u.x);  // Set visited.

		//const SkeletonAlgorithm::Vertex& uVtx = vertexMap[u.y * neighborGroupCounts.cols + u.x];
		std::map<int, const SkeletonAlgorithm::Vertex>::const_iterator citVtx = vertexMap.find(u.y * neighborGroupCounts.cols + u.x);
		if (vertexMap.end() == citVtx) assert(false);

		// Visit self.
		std::list<cv::Point> adjacents;
		findAdjacentVerticesByNeighborGroup(neighborGroupCounts, neighborGroupsOfInternalVertices, u, visited, adjacents);
		for (const auto& adj : adjacents)
		{
			//edges.push_back(std::make_pair(citVtx->second.id, vertexMap[adj.y * neighborGroupCounts.cols + adj.x].id));
			std::map<int, const SkeletonAlgorithm::Vertex>::const_iterator citAdj = vertexMap.find(adj.y * neighborGroupCounts.cols + adj.x);
			if (vertexMap.end() == citAdj) assert(false);
			else
				edges.push_back(std::make_pair(citVtx->second.id, citAdj->second.id));
		}

		// Traverse adjacent vertices.
		for (const auto& adj : adjacents)
			if (visited.end() == visited.find(adj.y * neighborGroupCounts.cols + adj.x))  // If unvisited.
				que.push(adj);
	}
#endif
}

/*static*/ void SkeletonAlgorithm::findEdgesByFollowingSkeleton(const std::vector<std::vector<cv::Point> >& contours, const cv::Mat& neighborGroupCounts, const std::list<SkeletonAlgorithm::Vertex>& vertices, std::list<std::pair<const int, const int> >& edges)
{
	std::map<int, const SkeletonAlgorithm::Vertex> vertexMap;
	for (const auto& vtx : vertices)
		vertexMap.insert(std::make_pair(vtx.pt.y * neighborGroupCounts.cols + vtx.pt.x, vtx));

	std::set<int> visited;
	cv::Point nextNearestVertex;
	for (const auto& contour : contours)
	{
		const SkeletonAlgorithm::Vertex *prevVertex = NULL, *firstVertex = NULL;
		std::vector<cv::Point>::const_iterator citNext;
		bool isContourSegmentVisitedBefore = false, isFirstContourSegmentVisitedBefore = false, isFirstContourSegment = true, isFirstPointAVertex = false, doesJustStart = true;
		for (std::vector<cv::Point>::const_iterator citCurr = contour.begin(); citCurr != contour.end(); ++citCurr)
		{
			citNext = citCurr + 1;

			std::map<int, const SkeletonAlgorithm::Vertex>::const_iterator citCurrVertex = vertexMap.find(citCurr->y * neighborGroupCounts.cols + citCurr->x);
			const bool isVertex = vertexMap.end() != citCurrVertex;
			const bool doesHaveNextNearestVertex = contour.end() == citNext ? false : getNextNearestVertexByFollowingSkeleton(neighborGroupCounts, *citCurr, *citNext, nextNearestVertex);
			// Look for the first vertex.
			//if (!firstVertex && !doesHaveNextNearestVertex && !isVertex) continue;

			std::map<int, const SkeletonAlgorithm::Vertex>::const_iterator citNextNearestVertex = doesHaveNextNearestVertex ? vertexMap.find(nextNearestVertex.y * neighborGroupCounts.cols + nextNearestVertex.x) : vertexMap.end();

			// Connect the first & last points to complete a contour.
			if (!firstVertex && (isVertex || doesHaveNextNearestVertex))
			{
				firstVertex = prevVertex = isVertex ? &citCurrVertex->second : &citNextNearestVertex->second;
				//continue;
			}

#if 0
			if (isVertex)
				isContourSegmentVisitedBefore = false;
#endif

			if (visited.end() == visited.find(citCurr->y * neighborGroupCounts.cols + citCurr->x))  // If unvisited.
			{
				visited.insert(citCurr->y * neighborGroupCounts.cols + citCurr->x);  // Set visited.

				if (isVertex)
				{
					if (prevVertex && prevVertex->pt != citCurrVertex->second.pt)
						edges.push_back(std::make_pair(prevVertex->id, citCurrVertex->second.id));
				}

#if 0
				if (doesHaveNextNearestVertex && prevVertex && prevVertex->pt != citNextNearestVertex->second.pt)
					if (visited.end() == visited.find(citNextNearestVertex->second.pt.y * neighborGroupCounts.cols + citNextNearestVertex->second.pt.x))  // If unvisited.
					{
						visited.insert(citNextNearestVertex->second.pt.y * neighborGroupCounts.cols + citNextNearestVertex->second.pt.x);  // Set visited.
						edges.push_back(std::make_pair(prevVertex->id, citNextNearestVertex->second.id));
					}
#else
				if (doesHaveNextNearestVertex && prevVertex && prevVertex->pt != citNextNearestVertex->second.pt && !isContourSegmentVisitedBefore)
					edges.push_back(std::make_pair(prevVertex->id, citNextNearestVertex->second.id));
#endif
			}
			else if (!isVertex)
				isContourSegmentVisitedBefore = true;

			if (doesJustStart)
			{
				if (isVertex) isFirstPointAVertex = true;
				doesJustStart = false;
			}
			if (isFirstContourSegment)
			{
				if (isContourSegmentVisitedBefore) isFirstContourSegmentVisitedBefore = true;
				isFirstContourSegment = false;
			}

			if (doesHaveNextNearestVertex)
			{
				prevVertex = &citNextNearestVertex->second;

				isContourSegmentVisitedBefore = false;
			}
			else if (isVertex) prevVertex = &citCurrVertex->second;

			// Connect the first & last points to complete a contour.
			if (contour.end() == citNext)
				if (firstVertex && !isFirstPointAVertex && !isFirstContourSegmentVisitedBefore)
					edges.push_back(std::make_pair(prevVertex->id, firstVertex->id));
		}
	}
}

/*static*/ bool SkeletonAlgorithm::getNextNearestVertexByFollowingSkeleton(const cv::Mat& neighborGroupCounts, const cv::Point& curr, const cv::Point& next, cv::Point& nextNearestVertex)
{
	// Get neighbors.
	bool neighborFlags[local::NUM_NEIGHBORS] = { false, };
	//local::getNeighborsUsingNeighborCoords(skeleton_bw, curr, neighborFlags);
	local::getNeighborsUsingNeighborCoords(neighborGroupCounts, curr, neighborFlags);

	if (!(int)neighborGroupCounts.at<unsigned char>(next.y, next.x) || !local::isNeighbor(next, curr)) return false;

	const int nextId = (int)local::getNeighborId(next, curr);
	int startIdx = -1, endIdx = -1, idx = nextId;
	for (int i = 1; i < local::NUM_NEIGHBORS; ++i)
	{
		--idx;
		if (idx < 0) idx = local::NUM_NEIGHBORS - 1;

		if (!neighborFlags[idx])
		{
			startIdx = idx + 1;
			if (startIdx >= local::NUM_NEIGHBORS) startIdx = 0;
			break;
		}
	}
	idx = nextId;
	for (int i = 1; i < local::NUM_NEIGHBORS; ++i)
	{
		++idx;
		if (idx >= local::NUM_NEIGHBORS) idx = 0;

		if (!neighborFlags[idx])
		{
			endIdx = idx - 1;
			if (endIdx < 0) endIdx = local::NUM_NEIGHBORS - 1;
			break;
		}
	}

	if (startIdx == endIdx)
	{
		nextNearestVertex = curr + local::NEIGHBOR_COORDINATES[startIdx];
		const int& numNeighborGroups = (int)neighborGroupCounts.at<unsigned char>(nextNearestVertex.y, nextNearestVertex.x);
		return (1 == numNeighborGroups || numNeighborGroups > 2);  // If a vertex.
	}

	idx = startIdx;
	bool found = false;
	for (int i = 0; i < local::NUM_NEIGHBORS; ++i)
	{
		const cv::Point& pt = curr + local::NEIGHBOR_COORDINATES[idx];
		const int& numNeighborGroups = (int)neighborGroupCounts.at<unsigned char>(pt.y, pt.x);
		if (1 == numNeighborGroups || numNeighborGroups > 2)  // If a vertex.
		{
			nextNearestVertex = pt;
			found = true;
		}

		if (endIdx == idx) return found;
		else if (0 == idx % 2 && found)  // Even index has a nearest pixel.
			return true;

		if (++idx >= local::NUM_NEIGHBORS) idx = 0;
	}

	return false;
}

/*static*/ void SkeletonAlgorithm::constructGraphByNeighborGroup(const cv::Mat& skeleton_bw, const std::list<cv::Point>& seedPoints, std::list<SkeletonAlgorithm::Vertex>& vertices, std::list<std::pair<const int, const int> >& edges)
{
#if 0
	int vertexCount = 0;

	// Set root vertices
	for (std::list<cv::Point>::const_iterator itRoot = seedPoints.begin(); itRoot != seedPoints.end(); ++itRoot)
		vertices.push_back(SkeletonAlgorithm::Vertex(*itRoot, SkeletonAlgorithm::ROOT, vertexCount++));

	// Find leaf vertices w/o information on relationship with other vertices (more correct).
	{
		std::list<cv::Point> leafEndVertices;
		findLeafEndVertices(skeleton_bw, leafEndVertices);
		for (std::list<cv::Point>::const_iterator itLeafEnd = leafEndVertices.begin(); itLeafEnd != leafEndVertices.end(); ++itLeafEnd)
			vertices.push_back(SkeletonAlgorithm::Vertex(*itLeafEnd, SkeletonAlgorithm::LEAF_END, vertexCount++));
	}

#if 1
	const std::list<SkeletonAlgorithm::Vertex> rootAndLeafVertices(vertices);
	std::set<int> visited;
	for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = rootAndLeafVertices.begin(); cit != rootAndLeafVertices.end(); ++cit)
		findAllVerticesByNeighborGroup(skeleton_bw, cit->pt, visited, vertexCount, vertices);

	// FIXME [implement] >> Find edges from vertices.

	//visited.clear();
	//std::map<int, int> vertexMap;
	//for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = vertices.begin(); cit != vertices.end(); ++cit)
	//	vertexMap.insert(std::make_pair(cit->pt.y * skeleton_bw.cols + cit->pt.x, cit->id));
	//for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = rootAndLeafVertices.begin(); cit != rootAndLeafVertices.end(); ++cit)
	//	findEdgesByTracingSkeleton(skeleton_bw, cit->pt, vertexMap, visited, edges);
#else
	const std::list<SkeletonAlgorithm::Vertex> rootAndLeafVertices(vertices);
	for (std::list<SkeletonAlgorithm::Vertex>::const_iterator cit = rootAndLeafVertices.begin(); cit != rootAndLeafVertices.end(); ++cit)
	{
		std::set<int> visited;
		findVerticesAndEdgesByNeighborGroup(skeleton_bw, *cit, visited, vertexCount, vertices, edges);

		// FIXME [correct] >>
		break;
	}
#endif
#else
	cv::Mat neighborGroupCounts;
	std::map<int, std::vector<int> > neighborGroupsOfInternalVertices;  // Retain information on neigbor groups of pixels which have more than 3 neighbor groups.
	computeNeighborGroups(skeleton_bw, neighborGroupCounts, &neighborGroupsOfInternalVertices);

	// NOTICE [info] >>
	//	If the number of neighbor groups == 1, these pixels are root or leaf vertex.
	//	If the number of neighbor groups == 2, these pixels are no vertex.
	//	If the number of neighbor groups >= 3, these pixels are internal vertex.
	//neighborGroupCounts.setTo(cv::Scalar::all(0), 2 == neighborGroupCounts);

	// Find vertices.
	findVerticesByNeighborGroup(neighborGroupCounts, vertices);

	// Set to ROOT.
	std::list<cv::Point> roots;
	for (const auto& seed : seedPoints)
	{
		cv::Point nearest;
		if (getNeareastVertexByNeighborGroup(neighborGroupCounts, neighborGroupsOfInternalVertices, seed, nearest))
		{
			if (roots.end() == std::find(roots.begin(), roots.end(), nearest))
			{
				roots.push_back(nearest);

				for (std::list<SkeletonAlgorithm::Vertex>::iterator it = vertices.begin(); it != vertices.end(); ++it)
					if (it->pt == nearest)
					{
						it->type = SkeletonAlgorithm::ROOT;
						break;
					}
			}
		}
		else assert(false);
	}

	// Find edges.
	std::set<int> visited;
	for (const auto& root : roots)
		if (visited.end() == visited.find(root.y * neighborGroupCounts.cols + root.x))  // If unvisited.
			findEdgesByNeighborGroup(neighborGroupCounts, neighborGroupsOfInternalVertices, root, visited, vertices, edges);
#endif

#if 0
	// Show vertices: for checking.
	cv::Mat rgb;
	cv::cvtColor(skeleton_bw, rgb, cv::COLOR_GRAY2BGR);
	showVertices(rgb, "Plant Graph - Vertices by Neighbor Group", vertices);
#endif
}

/*static*/ void SkeletonAlgorithm::constructGraphByFollowingSkeleton(const cv::Mat& skeleton_bw, std::list<SkeletonAlgorithm::Vertex>& vertices, std::list<std::pair<const int, const int> >& edges)
{
	cv::Mat neighborGroupCounts;
	std::map<int, std::vector<int> > neighborGroupsOfInternalVertices;  // Retain information on neigbor groups of pixels which have more than 3 neighbor groups.
	computeNeighborGroups(skeleton_bw, neighborGroupCounts, &neighborGroupsOfInternalVertices);

	// NOTICE [info] >>
	//	If the number of neighbor groups == 1, these pixels are root or leaf vertex.
	//	If the number of neighbor groups == 2, these pixels are no vertex.
	//	If the number of neighbor groups >= 3, these pixels are internal vertex.
	//neighborGroupCounts.setTo(cv::Scalar::all(0), 2 == neighborGroupCounts);

	// Find vertices.
	findVerticesByNeighborGroup(neighborGroupCounts, vertices);

	// Get contours.
	std::vector<std::vector<cv::Point> > contours;
	cv::Mat skel;
	skeleton_bw.copyTo(skel);
	cv::findContours(skel, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

	// Find edges.
	findEdgesByFollowingSkeleton(contours, neighborGroupCounts, vertices, edges);

#if 0
	// Show vertices: for checking.
	cv::Mat rgb;
	cv::cvtColor(skeleton_bw, rgb, cv::COLOR_GRAY2BGR);
	showVertices(rgb, "Plant Graph - Vertices by Following Skeleton", vertices);
#endif
}

/*static*/ bool SkeletonAlgorithm::findSkeletalPathBetweenTwoVerticesImpl(const cv::Mat& skeleton_bw, const cv::Point& curr, const cv::Point& end, std::set<int>& visited, std::list<cv::Point>& path)
{
	// Depth-first search.

	if (visited.end() != visited.find(curr.y * skeleton_bw.cols + curr.x))  // If visited.
		return true;
	visited.insert(curr.y * skeleton_bw.cols + curr.x);  // Set visited.

															// Get neighbors.
	const std::list<cv::Point>& neighbors = local::getNeighborsUsingNeighborCoords(skeleton_bw, curr);

	// Visit self.
	path.push_back(curr);
	bool hasVertex = false;
	for (std::list<cv::Point>::const_iterator cit = neighbors.begin(); cit != neighbors.end(); ++cit)
	{
		if (visited.end() == visited.find(cit->y * skeleton_bw.cols + cit->x))  // If unvisited.
		{
			if (isVertex(skeleton_bw, *cit, NULL))
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
		if (visited.end() == visited.find(cit->y * skeleton_bw.cols + cit->x))  // If unvisited.
		{
			if (findSkeletalPathBetweenTwoVerticesImpl(skeleton_bw, *cit, end, visited, path))
				return true;
			else path.pop_back();
		}

	return false;
}

/*static*/ bool SkeletonAlgorithm::findSkeletalPathBetweenTwoVertices(const cv::Mat& skeleton_bw, const cv::Point& start, const cv::Point& end, std::list<cv::Point>& path)
{
	if (!skeleton_bw.at<unsigned char>(start.y, start.x) || !skeleton_bw.at<unsigned char>(end.y, end.x))
		return false;

	std::set<int> visited;
	return findSkeletalPathBetweenTwoVerticesImpl(skeleton_bw, start, end, visited, path);
}

/*static*/ void SkeletonAlgorithm::findLineSegmentsInRow(const cv::Mat& skeleton_bw, const int row, const int colStart, const int colEnd, std::map<SkeletonAlgorithm::LineSegment, int>& lineSegments)
{
	for (int c = colStart; c <= colEnd; ++c)
	{
		if ((int)skeleton_bw.at<unsigned char>(row, c) > 0)
		{
			const int segmentStart = c;
			for (++c; c < skeleton_bw.cols; ++c)
			{
				if ((int)skeleton_bw.at<unsigned char>(row, c) == 0)
					break;
			}
			lineSegments.insert(std::make_pair(SkeletonAlgorithm::LineSegment(cv::Point(segmentStart, row), cv::Point(c - 1, row)), -1));
		}
	}
}

}  // namespace swl
