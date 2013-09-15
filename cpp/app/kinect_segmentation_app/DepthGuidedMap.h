#if !defined(__SWL_KINECT_SEGMENTATION_APP__DEPTH_GUIDED_MAP__H_)
#define __SWL_KINECT_SEGMENTATION_APP__DEPTH_GUIDED_MAP__H_ 1


#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

const unsigned char SWL_BGD    = cv::GC_BGD;  // background
const unsigned char SWL_FGD    = cv::GC_FGD;  // foreground
const unsigned char SWL_PR_BGD = cv::GC_PR_BGD;  // most probably background
const unsigned char SWL_PR_FGD = cv::GC_PR_FGD;  // most probably foreground

void construct_depth_guided_map_using_superpixel(const cv::Mat &rgb_image, const cv::Mat &depth_image, const cv::Mat &depth_validity_mask, cv::Mat &depth_guided_map);
void construct_depth_guided_map_using_edge_detection_and_morphological_operation(const cv::Mat &depth_image, const cv::Mat &depth_validity_mask, cv::Mat &depth_guided_map);
void construct_depth_guided_map_using_depth_variation(const cv::Mat &depth_variation_mask, const cv::Mat &depth_input_image, cv::Mat &depth_guided_map, cv::Mat &filtered_depth_variation_mask);

}  // namespace swl


#endif  // __SWL_KINECT_SEGMENTATION_APP__DEPTH_GUIDED_MAP__H_
