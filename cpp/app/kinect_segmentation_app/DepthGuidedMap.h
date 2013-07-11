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
void construct_depth_guided_map_using_morphological_operation_of_depth_boundary(const cv::Mat &depth_image, const cv::Mat &depth_validity_mask, cv::Mat &depth_guided_map);
void construct_depth_guided_map_using_structure_tensor(const cv::Mat &structure_tensor_mask, cv::Mat &depth_guided_map);

}  // namespace swl


#endif  // __SWL_KINECT_SEGMENTATION_APP__DEPTH_GUIDED_MAP__H_