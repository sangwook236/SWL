//#include "stdafx.h"
#include "swl/Config.h"
//#include "swl/machine_vision/BoundaryLabeling.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <list>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

namespace {
namespace local {

void simple_boundary_labeling()
{
	std::list<std::string> img_filenames;

	for (const auto &img_filename : img_filenames)
	{
		cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
		if (img_filename.empty())
		{
			std::cerr << "File not found: " << img_filename << std::endl;
			continue;
		}
	}
}

}  // namespace local
}  // unnamed namespace

void boundary_labeling()
{
	local::simple_boundary_labeling();
}
