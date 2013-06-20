//#include "stdafx.h"
#include "./efficient_graph_based_image_segmentation_lib/segment-graph.h"
#include "./efficient_graph_based_image_segmentation_lib/misc.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <cassert>


namespace swl {

class universe_using_map_container
{
public:
	universe_using_map_container(const std::set<int> &elements);
	~universe_using_map_container();
	int find(int x);  
	bool contain(int x) const;  
	void join(int x, int y);
	int size(int x) const;
	int num_sets() const { return num; }

private:
	std::map<int, uni_elt> elts;
	int num;
};

universe_using_map_container::universe_using_map_container(const std::set<int> &elements)
	: num(elements.size())
{
	uni_elt elt;
	for (std::set<int>::const_iterator cit = elements.begin(); cit != elements.end(); ++cit)
	{
		elt.rank = 0;
		elt.size = 1;
		elt.p = *cit;

		elts.insert(std::make_pair(*cit, elt));
	}
}

universe_using_map_container::~universe_using_map_container()
{
}

int universe_using_map_container::size(int x) const
{
#if DEBUG || _DEBUG
	std::map<int, uni_elt>::const_iterator cit = elts.find(x);
	if (elts.end() == cit)
	{
		std::cerr << "can't find an element with index, " << x << std::endl;
		return 0;
	}

	return cit->second.size;
#else
	std::map<int, uni_elt>::const_iterator cit = elts.find(x);
	return elts.end() == cit ? 0 : cit->second.size;
#endif
}

int universe_using_map_container::find(int x)
{
#if DEBUG || _DEBUG
	int y = x;
	std::map<int, uni_elt>::iterator it = elts.find(y);
	if (elts.end() == it)
	{
		std::cerr << "can't find an element with index, " << y << std::endl;
		return -1;
	}
	else
	{
		while (y != elts[y].p)
		{
			y = elts[y].p;

			std::map<int, uni_elt>::iterator it = elts.find(y);
			if (elts.end() == it)
			{
				std::cerr << "can't find an element with index, " << y << std::endl;
				return -1;
			}
		}
		elts[x].p = y;
	}
#else
	int y = x;
	while (y != elts[y].p)
		y = elts[y].p;
	elts[x].p = y;
#endif
	return y;
}

bool universe_using_map_container::contain(int x) const
{
	return elts.end() != elts.find(x);
}

void universe_using_map_container::join(int x, int y)
{
#if DEBUG || _DEBUG
	std::map<int, uni_elt>::iterator it = elts.find(x);
	if (elts.end() == it)
	{
		std::cerr << "can't find an element with index, " << x << std::endl;
		return;
	}
	it = elts.find(y);
	if (elts.end() == it)
	{
		std::cerr << "can't find an element with index, " << y << std::endl;
		return;
	}

	if (elts[x].rank > elts[y].rank)
	{
		elts[y].p = x;
		elts[x].size += elts[y].size;
	}
	else
	{
		elts[x].p = y;
		elts[y].size += elts[x].size;
		if (elts[x].rank == elts[y].rank)
			elts[y].rank++;
	}
#else
	if (elts[x].rank > elts[y].rank)
	{
		elts[y].p = x;
		elts[x].size += elts[y].size;
	}
	else
	{
		elts[x].p = y;
		elts[y].size += elts[x].size;
		if (elts[x].rank == elts[y].rank)
			elts[y].rank++;
	}
#endif
	--num;
}

universe_using_map_container * segment_graph_using_map_container(const std::set<int> &vertex_set, int num_edges, edge *edges, float k)
{ 
	// sort edges by weight
	std::sort(edges, edges + num_edges);

	// make a disjoint-set forest
	const int num_vertices = (int)vertex_set.size();
	universe_using_map_container *u = new universe_using_map_container(vertex_set);

	// init thresholds
	std::map<int, float> threshold;
	for (std::set<int>::const_iterator cit = vertex_set.begin(); cit != vertex_set.end(); ++cit)
		threshold[*cit] = THRESHOLD(1,k);

	// for each edge, in non-decreasing weight order...
	for (int i = 0; i < num_edges; ++i)
	{
		edge *pedge = &edges[i];

		// components conected by this edge
		int a = u->find(pedge->a);
		int b = u->find(pedge->b);
		if (a != b)
		{
			if ((pedge->w <= threshold[a]) && (pedge->w <= threshold[b]))
			{
				u->join(a, b);
				a = u->find(a);
				threshold[a] = pedge->w + THRESHOLD(u->size(a), k);
			}
		}
	}

	return u;
}

void convolve_even(const cv::Mat &src, cv::Mat &dst, const std::vector<float> &mask)
{
	const int width = src.cols;
	const int height = src.rows;
	const int len = mask.size();

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			float sum = mask[0] * src.at<float>(y, x);
			for (int i = 1; i < len; ++i)
			{
				sum += mask[i] * (src.at<float>(y, std::max(x-i,0)) + src.at<float>(y, std::min(x+i, width-1)));
			}
			dst.at<float>(x, y) = sum;
		}
	}
}

// normalize mask so it integrates to one
void normalize(std::vector<float> &mask)
{
	const int len = mask.size();
	float sum = 0.0f;
	for (int i = 1; i < len; ++i)
	{
		sum += fabs(mask[i]);
	}
	sum = 2*sum + fabs(mask[0]);
	for (int i = 0; i < len; ++i)
	{
		mask[i] /= sum;
	}
}

// make filters
std::vector<float> make_fgauss(float sigma, const float width)
{
	sigma = std::max(sigma, 0.01f);
	const int len = (int)ceil(sigma * width) + 1;
	std::vector<float> mask(len);
	for (int i = 0; i < len; ++i)
	{
		mask[i] = (float)std::exp(-0.5f * square(float(i) / sigma));
	}
	return mask;
}

// convolve image with gaussian filter
cv::Mat smooth(const cv::Mat &src, const float sigma, const float width)
{
	std::vector<float> mask = make_fgauss(sigma, width);
	normalize(mask);

	cv::Mat tmp(src.cols, src.rows, src.type());
	cv::Mat dst(src.rows, src.cols, src.type());
	convolve_even(src, tmp, mask);
	convolve_even(tmp, dst, mask);

	return dst;
}

// dissimilarity measure between pixels
inline float diff(const cv::Mat &red, const cv::Mat &green, const cv::Mat &blue, const int x1, const int y1, const int x2, const int y2, const float lambda)
{
#if 1
	return std::sqrt(square(red.at<float>(y1, x1) - red.at<float>(y2, x2)) +
		square(green.at<float>(y1, x1) - green.at<float>(y2, x2)) +
		square(blue.at<float>(y1, x1) - blue.at<float>(y2, x2)));
#else
	return std::sqrt(square(red.at<float>(y1, x1) - red.at<float>(y2, x2)) +
		square(green.at<float>(y1, x1) - green.at<float>(y2, x2)) +
		square(blue.at<float>(y1, x1) - blue.at<float>(y2, x2))) +
		lambda * std::sqrt(square((float)x1 - (float)x2) + square((float)y1 - (float)y2));
#endif
}

inline float diff_for_valid_depth_region(const cv::Mat &red, const cv::Mat &green, const cv::Mat &blue, const cv::Mat &depth, const int x1, const int y1, const int x2, const int y2, const float lambda, const float fx, const float fy)
{
#if 0
	return std::sqrt(square(red.at<float>(y1, x1) - red.at<float>(y2, x2)) +
		square(green.at<float>(y1, x1) - green.at<float>(y2, x2)) +
		square(blue.at<float>(y1, x1) - blue.at<float>(y2, x2)));
#else
	const unsigned short z1 = depth.at<unsigned short>(y1, x1), z2 = depth.at<unsigned short>(y2, x2);
	return std::sqrt(square(red.at<float>(y1, x1) - red.at<float>(y2, x2)) +
		square(green.at<float>(y1, x1) - green.at<float>(y2, x2)) +
		square(blue.at<float>(y1, x1) - blue.at<float>(y2, x2))) +
		lambda * std::sqrt(square(float(z1 * x1 - z2 * x2) / fx) + square(float(z1 * y1 - z2 * y2) / fy) + square((float)z1 - (float)z2));
#endif
}

inline float diff_for_depth_boundary_region(const cv::Mat &red, const cv::Mat &green, const cv::Mat &blue, const int x1, const int y1, const int x2, const int y2, const float lambda)
{
#if 1
	return std::sqrt(square(red.at<float>(y1, x1) - red.at<float>(y2, x2)) +
		square(green.at<float>(y1, x1) - green.at<float>(y2, x2)) +
		square(blue.at<float>(y1, x1) - blue.at<float>(y2, x2)));
#else
	return std::sqrt(square(red.at<float>(y1, x1) - red.at<float>(y2, x2)) +
		square(green.at<float>(y1, x1) - green.at<float>(y2, x2)) +
		square(blue.at<float>(y1, x1) - blue.at<float>(y2, x2))) +
		lambda * std::sqrt(square((float)x1 - (float)x2) + square((float)y1 - (float)y2));
#endif
}

/*
 * Segment an image
 *
 * rgb_input_image: image to segment.
 * depth_guided_mask: mask image for depth guidance.
 * sigma: to smooth the image.
 * k: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * lambda1: a weight factor for edge weight between two pixels in valid depth regions.
 * lambda2: a weight factor for edge weight between two pixels in invalid depth regions.
 * lambda3: a weight factor for edge weight between two pixels in depth boundary regions.
 * fx_rgb: the x-axis focal length of an RGB camera.
 * fy_rgb: the y-axis focal length of an RGB camera.
 * num_ccs: number of connected components in the segmentation.
 * output_image: a color image representing the segmentation.
 */
void segment_image_using_efficient_graph_based_image_segmentation_algorithm(
	const cv::Mat &rgb_input_image, const cv::Mat &depth_input_image, const cv::Mat &depth_guided_mask,
	const float sigma, const float k, const int min_size,
	const float lambda1, const float lambda2, const float lambda3, const float fx_rgb, const float fy_rgb,
	int &num_ccs, cv::Mat &output_image
)
{
	const int width = rgb_input_image.cols;
	const int height = rgb_input_image.rows;

	// smooth each color channel  
	std::vector<cv::Mat> float_ch_imgs;
	{
		std::vector<cv::Mat> byte_ch_imgs;
		cv::split(rgb_input_image, byte_ch_imgs);

		const float sigma_new = std::max(sigma, 0.01f);
		const float kwidth = 4.0f;
#if 0
		float_ch_imgs.resize(byte_ch_imgs.size());
		const int ksize = (int)std::ceil(sigma_new * kwidth) + 1;
		int i = 0;
		for (std::vector<cv::Mat>::iterator it = byte_ch_imgs.begin(); it != byte_ch_imgs.end(); ++it, ++i)
		{
			cv::Mat tmp;
			it->convertTo(tmp, CV_32FC1, 1.0, 0.0);
			cv::GaussianBlur(tmp, float_ch_imgs[i], cv::Size(ksize, ksize), sigma_new, sigma_new);
		}
#else
		float_ch_imgs.reserve(byte_ch_imgs.size());
		for (std::vector<cv::Mat>::iterator it = byte_ch_imgs.begin(); it != byte_ch_imgs.end(); ++it)
		{
			cv::Mat tmp;
			it->convertTo(tmp, CV_32FC1, 1.0, 0.0);
			float_ch_imgs.push_back(smooth(tmp, sigma_new, kwidth));
		}
#endif
	}

	// build graph
	const int total_num_edges = width * height * 4 - (width + height - 2) * 3 - 4;
	std::vector<edge> edge_list1, edge_list2, edge_list3;
	edge_list1.reserve(total_num_edges);
	edge_list2.reserve(total_num_edges);
	edge_list3.reserve(total_num_edges);
	edge ed;
	int num = 0;
#if 0
	// FIXME [correct] >> not working. run-time error.

	int num_vertices1 = 0, num_vertices2 = 0, num_vertices3 = 0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const unsigned char sa = depth_guided_mask.at<unsigned char>(y, x);
			if (255 == sa)  // for pixels in valid depth regions.
				++num_vertices1;
			else if (0 == sa)  // for pixels in invalid depth regions.
				++num_vertices2;
			else if (127 == sa)  // for pixels in depth boundary regions.
				++num_vertices3;

			if (x < width-1)
			{
				const unsigned char sb = depth_guided_mask.at<unsigned char>(y, x+1);

				ed.a = y * width + x;
				ed.b = y * width + (x+1);
				if (127 == sa || 127 == sb)  // for pixels in depth boundary regions.
				{
					if (sa != sb) ++num_vertices3;
					ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y, lambda3);
					edge_list3.push_back(ed);
				}
				else if (255 == sa && 255 == sb)  // for pixels in valid depth regions.
				{
					ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_input_image, x, y, x+1, y, lambda1, fx_rgb, fy_rgb);
					edge_list1.push_back(ed);
				}
				else if (0 == sa && 0 == sb)  // for pixels in invalid depth regions.
				{
					ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y, lambda2);
					edge_list2.push_back(ed);
				}
				else
					std::cerr << "invalid depth region state (" << (unsigned int)sa << ", " << (unsigned int)sb << ") at pixel (" << x << ", " << y << ") ..." << std::endl;
				++num;
			}

			if (y < height-1)
			{
				const unsigned char sb = depth_guided_mask.at<unsigned char>(y+1, x);

				ed.a = y * width + x;
				ed.b = (y+1) * width + x;
				if (127 == sa || 127 == sb)  // for pixels in depth boundary regions.
				{
					if (sa != sb) ++num_vertices3;
					ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x, y+1, lambda3);
					edge_list3.push_back(ed);
				}
				else if (255 == sa && 255 == sb)  // for pixels in valid depth regions.
				{
					ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_input_image, x, y, x, y+1, lambda1, fx_rgb, fy_rgb);
					edge_list1.push_back(ed);
				}
				else if (0 == sa && 0 == sb)  // for pixels in invalid depth regions.
				{
					ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x, y+1, lambda2);
					edge_list2.push_back(ed);
				}
				else
					std::cerr << "invalid depth region state (" << (unsigned int)sa << ", " << (unsigned int)sb << ") at pixel (" << x << ", " << y << ") ..." << std::endl;
				++num;
			}

			if ((x < width-1) && (y < height-1))
			{
				const unsigned char sb = depth_guided_mask.at<unsigned char>(y+1, x+1);

				ed.a = y * width + x;
				ed.b = (y+1) * width + (x+1);
				if (127 == sa || 127 == sb)  // for pixels in depth boundary regions.
				{
					if (sa != sb) ++num_vertices3;
					ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y+1, lambda3);
					edge_list3.push_back(ed);
				}
				else if (255 == sa && 255 == sb)  // for pixels in valid depth regions.
				{
					ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_input_image, x, y, x+1, y+1, lambda1, fx_rgb, fy_rgb);
					edge_list1.push_back(ed);
				}
				else if (0 == sa && 0 == sb)  // for pixels in invalid depth regions.
				{
					ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y+1, lambda2);
					edge_list2.push_back(ed);
				}
				else
					std::cerr << "invalid depth region state (" << (unsigned int)sa << ", " << (unsigned int)sb << ") at pixel (" << x << ", " << y << ") ..." << std::endl;
				++num;
			}

			if ((x < width-1) && (y > 0))
			{
				const unsigned char sb = depth_guided_mask.at<unsigned char>(y-1, x+1);

				ed.a = y * width + x;
				ed.b = (y-1) * width + (x+1);
				if (127 == sa || 127 == sb)  // for pixels in depth boundary regions.
				{
					if (sa != sb) ++num_vertices3;
					ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y-1, lambda3);
					edge_list3.push_back(ed);
				}
				else if (255 == sa && 255 == sb)  // for pixels in valid depth regions.
				{
					ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_input_image, x, y, x+1, y-1, lambda1, fx_rgb, fy_rgb);
					edge_list1.push_back(ed);
				}
				else if (0 == sa && 0 == sb)  // for pixels in invalid depth regions.
				{
					ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y-1, lambda2);
					edge_list2.push_back(ed);
				}
				else
					std::cerr << "invalid depth region state (" << (unsigned int)sa << ", " << (unsigned int)sb << ") at pixel (" << x << ", " << y << ") ..." << std::endl;
				++num;
			}
		}
	}
	float_ch_imgs.clear();

	assert(total_num_edges == num);
	assert(total_num_edges == (edge_list1.size() + edge_list2.size() + edge_list3.size()));

	// segment
	universe *u1 = segment_graph(num_vertices1, edge_list1.size(), &edge_list1[0], k);  // for pixels in valid depth regions.
	universe *u2 = segment_graph(num_vertices2, edge_list2.size(), &edge_list2[0], k);  // for pixels in invalid depth regions.
	universe *u3 = segment_graph(num_vertices3, edge_list3.size(), &edge_list3[0], k);  // for pixels in depth boundary regions.
#elif 1
	for (int state_idx = 0; state_idx < 3; ++state_idx)
	{
		const unsigned char target_state = (0 == state_idx ? 255 : (1 == state_idx ? 0 : 127));
		std::vector<edge> &edge_list = (0 == state_idx ? edge_list1 : (1 == state_idx ? edge_list2 : edge_list3));
		const float lambda = (0 == state_idx ? lambda1 : (1 == state_idx ? lambda2 : lambda3));
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				const unsigned char sa = depth_guided_mask.at<unsigned char>(y, x);

				if (x < width-1)
				{
					const unsigned char sb = depth_guided_mask.at<unsigned char>(y, x+1);

					ed.a = y * width + x;
					ed.b = y * width + (x+1);
					if (target_state == sa && target_state == sb)
					{
						if (255 == target_state)  // for pixels in valid depth regions.
							ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_guided_mask, x, y, x+1, y, lambda, fx_rgb, fy_rgb);
						else if (0 == target_state)  // for pixels in invalid depth regions.
							ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y, lambda);
						else if (127 == target_state)  // for pixels in depth boundary regions.
							ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y, lambda);
					}
					else if (target_state == sa || target_state == sb)
						ed.w = std::numeric_limits<float>::max();
					else
						ed.w = 0.0f;
					edge_list.push_back(ed);
					++num;
				}

				if (y < height-1)
				{
					const unsigned char sb = depth_guided_mask.at<unsigned char>(y+1, x);

					ed.a = y * width + x;
					ed.b = (y+1) * width + x;
					if (target_state == sa && target_state == sb)
					{
						if (255 == target_state)  // for pixels in valid depth regions.
							ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_guided_mask, x, y, x, y+1, lambda, fx_rgb, fy_rgb);
						else if (0 == target_state)  // for pixels in invalid depth regions.
							ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x, y+1, lambda);
						else if (127 == target_state)  // for pixels in depth boundary regions.
							ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x, y+1, lambda);
					}
					else if (target_state == sa || target_state == sb)
						ed.w = std::numeric_limits<float>::max();
					else
						ed.w = 0.0f;
					edge_list.push_back(ed);
					++num;
				}

				if ((x < width-1) && (y < height-1))
				{
					const unsigned char sb = depth_guided_mask.at<unsigned char>(y+1, x+1);

					ed.a = y * width + x;
					ed.b = (y+1) * width + (x+1);
					if (target_state == sa && target_state == sb)
					{
						if (255 == target_state)  // for pixels in valid depth regions.
							ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_guided_mask, x, y, x+1, y+1, lambda, fx_rgb, fy_rgb);
						else if (0 == target_state)  // for pixels in invalid depth regions.
							ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y+1, lambda);
						else if (127 == target_state)  // for pixels in depth boundary regions.
							ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y+1, lambda);
					}
					else if (target_state == sa || target_state == sb)
						ed.w = std::numeric_limits<float>::max();
					else
						ed.w = 0.0f;
					edge_list.push_back(ed);
					++num;
				}

				if ((x < width-1) && (y > 0))
				{
					const unsigned char sb = depth_guided_mask.at<unsigned char>(y-1, x+1);

					ed.a = y * width + x;
					ed.b = (y-1) * width + (x+1);
					if (target_state == sa && target_state == sb)
					{
						if (255 == target_state)  // for pixels in valid depth regions.
							ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_guided_mask, x, y, x+1, y-1, lambda, fx_rgb, fy_rgb);
						else if (0 == target_state)  // for pixels in invalid depth regions.
							ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y-1, lambda);
						else if (127 == target_state)  // for pixels in depth boundary regions.
							ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y-1, lambda);
					}
					else if (target_state == sa || target_state == sb)
						ed.w = std::numeric_limits<float>::max();
					else
						ed.w = 0.0f;
					edge_list.push_back(ed);
					++num;
				}
			}
		}
	}
	float_ch_imgs.clear();

	assert(total_num_edges == edge_list1.size());
	assert(total_num_edges == edge_list2.size());
	assert(total_num_edges == edge_list3.size());

	// segment
	universe *u1 = segment_graph(width * height, edge_list1.size(), &edge_list1[0], k);  // for pixels in valid depth regions.
	universe *u2 = segment_graph(width * height, edge_list2.size(), &edge_list2[0], k);  // for pixels in invalid depth regions.
	universe *u3 = segment_graph(width * height, edge_list3.size(), &edge_list3[0], k);  // for pixels in depth boundary regions.
#elif 0
	std::set<int> vertex_set1, vertex_set2, vertex_set3;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const unsigned char sa = depth_guided_mask.at<unsigned char>(y, x);

			if (x < width-1)
			{
				ed.a = y * width + x;
				ed.b = y * width + (x+1);
				const unsigned char sb = depth_guided_mask.at<unsigned char>(y, x+1);
				if (127 == sa || 127 == sb)  // for pixels in depth boundary regions.
				{
					vertex_set3.insert(ed.a);
					vertex_set3.insert(ed.b);
					ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y, lambda3);
					edge_list3.push_back(ed);
				}
				else if (255 == sa && 255 == sb)  // for pixels in valid depth regions.
				{
					vertex_set1.insert(ed.a);
					vertex_set1.insert(ed.b);
					ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_input_image, x, y, x+1, y, lambda1, fx_rgb, fy_rgb);
					edge_list1.push_back(ed);
				}
				else if (0 == sa && 0 == sb)  // for pixels in invalid depth regions.
				{
					vertex_set2.insert(ed.a);
					vertex_set2.insert(ed.b);
					ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y, lambda2);
					edge_list2.push_back(ed);
				}
				else
					std::cerr << "invalid depth region state (" << (unsigned int)sa << ", " << (unsigned int)sb << ") at pixel (" << x << ", " << y << ") ..." << std::endl;
				++num;
			}

			if (y < height-1)
			{
				ed.a = y * width + x;
				ed.b = (y+1) * width + x;
				const unsigned char sb = depth_guided_mask.at<unsigned char>(y+1, x);
				if (127 == sa || 127 == sb)  // for pixels in depth boundary regions.
				{
					vertex_set3.insert(ed.a);
					vertex_set3.insert(ed.b);
					ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x, y+1, lambda3);
					edge_list3.push_back(ed);
				}
				else if (255 == sa && 255 == sb)  // for pixels in valid depth regions.
				{
					vertex_set1.insert(ed.a);
					vertex_set1.insert(ed.b);
					ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_input_image, x, y, x, y+1, lambda1, fx_rgb, fy_rgb);
					edge_list1.push_back(ed);
				}
				else if (0 == sa && 0 == sb)  // for pixels in invalid depth regions.
				{
					vertex_set2.insert(ed.a);
					vertex_set2.insert(ed.b);
					ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x, y+1, lambda2);
					edge_list2.push_back(ed);
				}
				else
					std::cerr << "invalid depth region state (" << (unsigned int)sa << ", " << (unsigned int)sb << ") at pixel (" << x << ", " << y << ") ..." << std::endl;
				++num;
			}

			if ((x < width-1) && (y < height-1))
			{
				ed.a = y * width + x;
				ed.b = (y+1) * width + (x+1);
				const unsigned char sb = depth_guided_mask.at<unsigned char>(y+1, x+1);
				if (127 == sa || 127 == sb)  // for pixels in depth boundary regions.
				{
					vertex_set3.insert(ed.a);
					vertex_set3.insert(ed.b);
					ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y+1, lambda3);
					edge_list3.push_back(ed);
				}
				else if (255 == sa && 255 == sb)  // for pixels in valid depth regions.
				{
					vertex_set1.insert(ed.a);
					vertex_set1.insert(ed.b);
					ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_input_image, x, y, x+1, y+1, lambda1, fx_rgb, fy_rgb);
					edge_list1.push_back(ed);
				}
				else if (0 == sa && 0 == sb)  // for pixels in invalid depth regions.
				{
					vertex_set2.insert(ed.a);
					vertex_set2.insert(ed.b);
					ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y+1, lambda2);
					edge_list2.push_back(ed);
				}
				else
					std::cerr << "invalid depth region state (" << (unsigned int)sa << ", " << (unsigned int)sb << ") at pixel (" << x << ", " << y << ") ..." << std::endl;
				++num;
			}

			if ((x < width-1) && (y > 0))
			{
				ed.a = y * width + x;
				ed.b = (y-1) * width + (x+1);
				const unsigned char sb = depth_guided_mask.at<unsigned char>(y-1, x+1);
				if (127 == sa || 127 == sb)  // for pixels in depth boundary regions.
				{
					vertex_set3.insert(ed.a);
					vertex_set3.insert(ed.b);
					ed.w = diff_for_depth_boundary_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y-1, lambda3);
					edge_list3.push_back(ed);
				}
				else if (255 == sa && 255 == sb)  // for pixels in valid depth regions.
				{
					vertex_set1.insert(ed.a);
					vertex_set1.insert(ed.b);
					ed.w = diff_for_valid_depth_region(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], depth_input_image, x, y, x+1, y-1, lambda1, fx_rgb, fy_rgb);
					edge_list1.push_back(ed);
				}
				else if (0 == sa && 0 == sb)  // for pixels in invalid depth regions.
				{
					vertex_set2.insert(ed.a);
					vertex_set2.insert(ed.b);
					ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y-1, lambda2);
					edge_list2.push_back(ed);
				}
				else
					std::cerr << "invalid depth region state (" << (unsigned int)sa << ", " << (unsigned int)sb << ") at pixel (" << x << ", " << y << ") ..." << std::endl;
				++num;
			}
		}
	}
	float_ch_imgs.clear();

	assert(total_num_edges == num);
	assert(total_num_edges == (edge_list1.size() + edge_list2.size() + edge_list3.size()));

	// segment
	universe_using_map_container *u1 = segment_graph_using_map_container(vertex_set1, edge_list1.size(), &edge_list1[0], k);  // for pixels in valid depth regions.
	universe_using_map_container *u2 = segment_graph_using_map_container(vertex_set2, edge_list2.size(), &edge_list2[0], k);  // for pixels in invalid depth regions.
	universe_using_map_container *u3 = segment_graph_using_map_container(vertex_set3, edge_list3.size(), &edge_list3[0], k);  // for pixels in depth boundary regions.
	vertex_set1.clear();
	vertex_set2.clear();
	vertex_set3.clear();
#endif

	// post process small components
	{
		for (std::vector<edge>::const_iterator cit = edge_list1.begin(); cit != edge_list1.end(); ++cit)
		{
			const int a = u1->find(cit->a);
			const int b = u1->find(cit->b);
			if ((a != b) && ((u1->size(a) < min_size) || (u1->size(b) < min_size)))
				u1->join(a, b);
		}
		edge_list1.clear();
	}
	{
		for (std::vector<edge>::const_iterator cit = edge_list2.begin(); cit != edge_list2.end(); ++cit)
		{
			const int a = u2->find(cit->a);
			const int b = u2->find(cit->b);
			if ((a != b) && ((u2->size(a) < min_size) || (u2->size(b) < min_size)))
				u2->join(a, b);
		}
		edge_list2.clear();
	}
	{
		for (std::vector<edge>::const_iterator cit = edge_list3.begin(); cit != edge_list3.end(); ++cit)
		{
			const int a = u3->find(cit->a);
			const int b = u3->find(cit->b);
			if ((a != b) && ((u3->size(a) < min_size) || (u3->size(b) < min_size)))
				u3->join(a, b);
		}
		edge_list3.clear();
	}
	num_ccs = u1->num_sets() + u2->num_sets() + u3->num_sets();

	// pick random colors for each component
	std::vector<cv::Vec3b> colors;
	colors.reserve(width * height);
	for (int i = 0; i < width*height; ++i)
		colors.push_back(cv::Vec3b((uchar)(std::rand() / 256), (uchar)(std::rand() / 256), (uchar)(std::rand() / 256)));

	output_image = cv::Mat::zeros(rgb_input_image.size(), rgb_input_image.type());
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
#if 1  // when using (original) universe.
			// FIXME [modify] >> how to use u1, u2, & u3? or how to integrate u1, u2, & u3?
			const int comp = u1->find(y * width + x);
			//const int comp = u2->find(y * width + x);
			//const int comp = u3->find(y * width + x);
			output_image.at<cv::Vec3b>(y, x) = colors[comp];
#else  // when using universe_using_map_container.
			const int vtx = y * width + x;
			int comp = -1;
			if (u1->contain(vtx))
				comp = u1->find(vtx);
			else if (u2->contain(vtx))
				comp = u2->find(vtx);
			else if (u3->contain(vtx))
				comp = u3->find(vtx);
			else
				std::cerr << "invalid vertex index, " << vtx << " at (" << y << ", " << x << ") ..." << std::endl;
			if (-1 != comp)
				output_image.at<cv::Vec3b>(y, x) = colors[comp];
#endif
		}
	}  

	delete u1;
	delete u2;
	delete u3;
}

void segment_image_using_efficient_graph_based_image_segmentation_algorithm(const cv::Mat &rgb_input_image, const float sigma, const float k, const int min_size, int &num_ccs, cv::Mat &output_image)
{
	const int width = rgb_input_image.cols;
	const int height = rgb_input_image.rows;

	// smooth each color channel  
	std::vector<cv::Mat> float_ch_imgs;
	{
		std::vector<cv::Mat> byte_ch_imgs;
		cv::split(rgb_input_image, byte_ch_imgs);

		const float sigma_new = std::max(sigma, 0.01f);
		const float kwidth = 4.0f;
#if 0
		float_ch_imgs.resize(byte_ch_imgs.size());
		const int ksize = (int)std::ceil(sigma_new * kwidth) + 1;
		int i = 0;
		for (std::vector<cv::Mat>::iterator it = byte_ch_imgs.begin(); it != byte_ch_imgs.end(); ++it, ++i)
		{
			cv::Mat tmp;
			it->convertTo(tmp, CV_32FC1, 1.0, 0.0);
			cv::GaussianBlur(tmp, float_ch_imgs[i], cv::Size(ksize, ksize), sigma_new, sigma_new);
		}
#else
		float_ch_imgs.reserve(byte_ch_imgs.size());
		for (std::vector<cv::Mat>::iterator it = byte_ch_imgs.begin(); it != byte_ch_imgs.end(); ++it)
		{
			cv::Mat tmp;
			it->convertTo(tmp, CV_32FC1, 1.0, 0.0);
			float_ch_imgs.push_back(smooth(tmp, sigma_new, kwidth));
		}
#endif
	}

	// build graph
	const int total_num_edges = width * height * 4 - (width + height - 2) * 3 - 4;
	std::vector<edge> edge_list;
	edge_list.reserve(total_num_edges);
	edge ed;
	int num = 0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			if (x < width-1)
			{
				ed.a = y * width + x;
				ed.b = y * width + (x+1);
				ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y, 0.0);
				edge_list.push_back(ed);
				++num;
			}

			if (y < height-1)
			{
				ed.a = y * width + x;
				ed.b = (y+1) * width + x;
				ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x, y+1, 0.0);
				edge_list.push_back(ed);
				++num;
			}

			if ((x < width-1) && (y < height-1))
			{
				ed.a = y * width + x;
				ed.b = (y+1) * width + (x+1);
				ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y+1, 0.0);
				edge_list.push_back(ed);
				++num;
			}

			if ((x < width-1) && (y > 0))
			{
				ed.a = y * width + x;
				ed.b = (y-1) * width + (x+1);
				ed.w = diff(float_ch_imgs[0], float_ch_imgs[1], float_ch_imgs[2], x, y, x+1, y-1, 0.0);
				edge_list.push_back(ed);
				++num;
			}
		}
	}
	float_ch_imgs.clear();

	assert(total_num_edges == edge_list.size());

	// segment
	universe *u = segment_graph(width * height, edge_list.size(), &edge_list[0], k);

	// post process small components
	{
		for (std::vector<edge>::const_iterator cit = edge_list.begin(); cit != edge_list.end(); ++cit)
		{
			const int a = u->find(cit->a);
			const int b = u->find(cit->b);
			if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
				u->join(a, b);
		}
		edge_list.clear();
	}
	num_ccs = u->num_sets();

	// pick random colors for each component
	std::vector<cv::Vec3b> colors;
	colors.reserve(width * height);
	for (int i = 0; i < width*height; ++i)
		colors.push_back(cv::Vec3b((uchar)(std::rand() / 256), (uchar)(std::rand() / 256), (uchar)(std::rand() / 256)));

	output_image = cv::Mat::zeros(rgb_input_image.size(), rgb_input_image.type());
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const int comp = u->find(y * width + x);
			output_image.at<cv::Vec3b>(y, x) = colors[comp];
		}
	}  

	delete u;
}

}  // namespace swl
