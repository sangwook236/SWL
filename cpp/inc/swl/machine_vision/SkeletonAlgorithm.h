#pragma once

#if !defined(__SWL_MACHINE_VISION__SKELETON_ALGORITHM__H_)
#define __SWL_MACHINE_VISION__SKELETON_ALGORITHM__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <set>
#include <map>
#include <list>
#include <vector>


namespace swl {

//--------------------------------------------------------------------------
// Skeleton Algorithm.

struct SWL_MACHINE_VISION_API SkeletonAlgorithm
{
public:
	enum VertexType { ROOT, LEAF_END, LEAF_CONCAVE, LEAF_CONVEX, BRANCH, JOIN, CROSS, CONTACT, INTERNAL, ISOLATED, UNDEFINED };

	struct SWL_MACHINE_VISION_API Vertex
	{
	public:
		explicit Vertex(const cv::Point& _pt, const VertexType _type, const int _id)
			: pt(_pt), type(_type), id(_id)
		{}
		Vertex(const Vertex& rhs)
			: pt(rhs.pt), type(rhs.type), id(rhs.id)
		{}

	public:
		const cv::Point pt;
		VertexType type;
		int id;
	};

	struct SWL_MACHINE_VISION_API LineSegment
	{
	public:
		explicit LineSegment(const cv::Point& pt1, const cv::Point& pt2);

	public:
		bool operator<(const LineSegment& rhs) const;

	public:
		bool isOverlapped(const LineSegment& rhs, const int skeletonOverlapMargin) const;
		bool isContained(const cv::Point& pt) const;

		cv::Point& getPt1() { return pt1_; }
		const cv::Point& getPt1() const { return pt1_; }
		cv::Point& getPt2() { return pt2_; }
		const cv::Point& getPt2() const { return pt2_; }
		cv::Point getMidPoint() const { return (pt1_ + pt2_) / 2; }

	private:
		cv::Point pt1_, pt2_;
	};

public:
	// Skeleton following/tracing approach.
	static void constructGraphByFollowingSkeleton(const cv::Mat& skeleton_bw, std::list<Vertex>& vertices, std::list<std::pair<const int, const int> >& edges);

	// Neighbor group-based approach.
	static void constructGraphByNeighborGroup(const cv::Mat& skeleton_bw, const std::list<cv::Point>& seedPoints, std::list<Vertex>& vertices, std::list<std::pair<const int, const int> >& edges);

	// Horizontal line scanning approach.
	static void constructGraphByUpwardLineScanning(const cv::Mat& skeleton_bw, const int rootVerticesRow, const int skeletonOverlapMargin, std::list<Vertex>& vertices, std::list<std::pair<const int, const int> >& edges);

	//
	static bool findSkeletalPathBetweenTwoVertices(const cv::Mat& skeleton_bw, const cv::Point& start, const cv::Point& end, std::list<cv::Point>& path);

	//
	static void computeNeighborGroups(const cv::Mat& skeleton_bw, cv::Mat& neighborGroupCounts, std::map<int, std::vector<int> >* neighborGroupsOfInternalVertices = NULL);
	static void findVerticesByNeighborGroup(const cv::Mat& neighborGroupCounts, std::list<Vertex>& vertices, int startId = 0);
	static bool getNeareastVertexByNeighborGroup(const cv::Mat& neighborGroupCounts, const std::map<int, std::vector<int> >& neighborGroupsOfInternalVertices, const cv::Point& curr, cv::Point& nearest);

	static void findLeafEndVertices(const cv::Mat& skeleton_bw, std::list<cv::Point>& leafEndVertices);

	static void countNeighbors(const cv::Mat& skeleton_bw, cv::Mat& neighborCounts);
	static bool isVertex(const cv::Mat& skeleton_bw, const cv::Point& currPt, VertexType* vertexType = NULL);

	//
	static void findLineSegmentsInRow(const cv::Mat& skeleton_bw, const int row, const int colStart, const int colEnd, std::map<LineSegment, int>& lineSegments);

private:
	// Skeleton following/tracing approach.
	static void findEdgesByFollowingSkeleton(const std::vector<std::vector<cv::Point> >& contours, const cv::Mat& neighborGroupCounts, const std::list<Vertex>& vertices, std::list<std::pair<const int, const int> >& edges);
	static bool getNextNearestVertexByFollowingSkeleton(const cv::Mat& neighborGroupCounts, const cv::Point& curr, const cv::Point& next, cv::Point& nextNearestVertex);

	// Neighbor group-based approach.
	static void findAdjacentVerticesByNeighborGroup(const cv::Mat& neighborGroupCounts, const std::map<int, std::vector<int> >& neighborGroupsOfInternalVertices, const cv::Point& curr, std::set<int>& visited, std::list<cv::Point>& adjacents);
	static void findAdjacentVerticesByNeighborGroup(const cv::Mat& skeleton_bw, const cv::Point& curr, std::set<int>& visited, std::list<Vertex>& adjacents);
	static void findAllVerticesByNeighborGroup(const cv::Mat& skeleton_bw, const cv::Point& start, std::set<int>& visited, int &vertexId, std::list<Vertex>& vertices);

	static void findEdgesByNeighborGroup(const cv::Mat& neighborGroupCounts, const std::map<int, std::vector<int> >& neighborGroupsOfInternalVertices, const cv::Point& curr, std::set<int>& visited, const std::list<Vertex>& vertices, std::list<std::pair<const int, const int> >& edges);
	static void findVerticesAndEdgesByNeighborGroup(const cv::Mat& skeleton_bw, const Vertex& curr, std::set<int>& visited, int& vertexId, std::list<Vertex>& vertices, std::list<std::pair<const int, const int> >& edges);

	static void getNeareastNeighborsInEachNeighborGroup(const cv::Mat& bw, const cv::Point& center, std::list<cv::Point>& nearestNeighborsInEachNeighborGroup);
	static void getNeareastNeighborIndicesInEachNeighborGroup(const std::vector<int>& neighborGroupIds, std::vector<int>& nearestIndicesInEachNeighborGroup);

	static size_t countNeighborGroups(const cv::Mat& skeleton_bw, const cv::Point& center);
	static size_t getNeighborGroups(const cv::Mat& skeleton_bw, const cv::Point& center, std::vector<int>& neighborGroupIds);
	static bool isInTheSameNeighborGroup(const std::vector<int>& neighborGroupIds, const size_t neighborId1, const size_t neighborId2);
	static bool isSurroundedBySingleNeighborGroup(const std::vector<int>& neighborGroupIds);
	static bool isOnBorder(const std::vector<int>& neighborGroupIds);

	static void getNeareastNeighborVerticesInEachNeighborGroup(const int numNeighborGroups, const std::vector<int>& neighborGroupIds, const std::list<cv::Point>& neighborsAsVertex, const cv::Point& curr, std::list<cv::Point>& neareastNeigborVertices);
	static void getNeareastNeighborVerticesInEachNeighborGroup(const int numNeighborGroups, const std::vector<int>& neighborGroupIds, const std::list<Vertex>& neighborsAsVertex, const cv::Point& curr, std::list<Vertex>& neareastNeigborVertices);

	//
	static bool findSkeletalPathBetweenTwoVerticesImpl(const cv::Mat& skeleton_bw, const cv::Point& curr, const cv::Point& end, std::set<int>& visited, std::list<cv::Point>& path);
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__SKELETON_ALGORITHM__H_
