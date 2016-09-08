#pragma once

#if !defined(__SWL_MACHINE_VISION__SKELETON_ALGORITHM__H_)
#define __SWL_MACHINE_VISION__SKELETON_ALGORITHM__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <set>
#include <map>
#include <list>


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

public:
	// Skeleton tracing(following) approach.
	static void findVerticesAndEdgesByTracingSkeleton(const cv::Mat& bw, const int rootVerticesLocation, std::list<Vertex>& vertices, std::list<std::pair<const int, const int> >& edges);

	// Horizontal line scanning approach.
	static void findVerticesAndEdgesByUpwardLineScanning(const cv::Mat& bw, const int rootVerticesLocation, const int skeletonOverlapMargin, std::list<Vertex>& vertices, std::list<std::pair<const int, const int> >& edges);

	static void findLeafEndVertices(const cv::Mat& bw, std::list<cv::Point>& leafEndVertices);
	static void findInternalVertices(const cv::Mat& bw, std::list<cv::Point>& internalVertices);

	static bool findSkeletalPathBetweenTwoVertices(const cv::Mat& bw, const cv::Point& start, const cv::Point& end, std::list<cv::Point>& path);

	static bool checkIfVertex(const cv::Mat& bw, const cv::Point& currPt, VertexType* vertexType);

private:
	// Skeleton tracing(following) approach.
	static void findVerticesAndEdgesByTracingSkeleton(const cv::Mat& bw, const Vertex& curr, std::set<int>& visited, int& vertexId, std::list<Vertex>& vertices, std::list<std::pair<const int, const int> >& edges);
	static void findAllVerticesByTracingSkeleton(const cv::Mat& bw, const cv::Point& startPt, std::set<int>& visited, int &vertexId, std::list<Vertex>& vertices);
	static void findAdjacentVerticesByTracingSkeleton(const cv::Mat& bw, const cv::Point& currPt, std::set<int>& visited, std::list<Vertex>& adjacents);

	static bool findSkeletalPathBetweenTwoVertices(const cv::Mat& bw, const cv::Point& curr, const cv::Point& end, std::set<int>& visited, std::list<cv::Point>& path);

	static size_t constructEdgeGroups(const cv::Mat& bw, const cv::Point& center, std::map<int, int>& edgeGroupIds);
	static bool isInTheSameEdgeGroup(const std::map<int, int>& edgeGroupIds, const cv::Point& center, const cv::Point& pt1, const cv::Point& pt2);
	static bool isOnBorder(const std::map<int, int>& edgeGroupIds);
	static bool isSurroundedBySingleEdgeGroup(const std::map<int, int>& edgeGroupIds);
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__SKELETON_ALGORITHM__H_
