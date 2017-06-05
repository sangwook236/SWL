#define _CRT_SECURE_NO_WARNINGS
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <ctime>
#include <cassert>
#include <boost/filesystem.hpp>
#include <boost/timer/timer.hpp>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include "swl/machine_vision/BoundaryExtraction.h"


namespace {
namespace local {

bool create_directory(const std::string &dir_path_name)
{
	boost::filesystem::path dir_path(dir_path_name);
	return boost::filesystem::exists(dir_path) ? true : boost::filesystem::create_directory(dir_path);
}

bool copy_files(const std::string &src_dir_path_name, const std::string &dst_dir_path_name, const std::string &src_file_prefix, const std::string &src_file_suffix, const std::string &dst_file_suffix)
{
	boost::filesystem::path src_dir_path(src_dir_path_name);
	if (!boost::filesystem::exists(src_dir_path))
	{
		std::cerr << "Source directory not found: " << src_dir_path_name << std::endl;
		return false;
	}
	boost::filesystem::path dst_dir_path(dst_dir_path_name);
	if (!boost::filesystem::exists(dst_dir_path))
	{
		std::cerr << "Destination directory not found: " << dst_dir_path_name << std::endl;
		return false;
	}

	std::string::size_type pos;
	boost::filesystem::directory_iterator end_iter;
	for (boost::filesystem::directory_iterator dir_itr(src_dir_path); dir_itr != end_iter; ++dir_itr)
	{
		if (!boost::filesystem::is_regular_file(*dir_itr)) continue;

		std::string filename(dir_itr->path().filename().generic_string());
		pos = filename.find(src_file_prefix);
		if (!src_file_prefix.empty() && 0 != pos) continue;
		pos = filename.rfind(src_file_suffix);
		if (!src_file_suffix.empty() && filename.length() != pos + src_file_suffix.length()) continue;
		
		boost::filesystem::path dst_path(dst_dir_path);
		//boost::filesystem::copy(*dir_itr, dst_path.append(filename));
		boost::filesystem::copy(*dir_itr, dst_path.append(filename.replace(pos, src_file_suffix.length(), dst_file_suffix)));
	}

	return true;
}

bool rename_files(const std::string &src_dir_path_name, const std::string &dst_dir_path_name, const std::string &src_file_prefix, const std::string &src_file_suffix, const std::string &dst_file_suffix)
{
	boost::filesystem::path src_dir_path(src_dir_path_name);
	if (!boost::filesystem::exists(src_dir_path))
	{
		std::cerr << "Source directory not found: " << src_dir_path_name << std::endl;
		return false;
	}
	boost::filesystem::path dst_dir_path(dst_dir_path_name);
	if (!boost::filesystem::exists(dst_dir_path))
	{
		std::cerr << "Destination directory not found: " << dst_dir_path_name << std::endl;
		return false;
	}

	std::string::size_type pos;
	boost::filesystem::directory_iterator end_iter;
	for (boost::filesystem::directory_iterator dir_itr(src_dir_path); dir_itr != end_iter; ++dir_itr)
	{
		if (!boost::filesystem::is_regular_file(*dir_itr)) continue;

		std::string filename(dir_itr->path().filename().generic_string());
		pos = filename.find(src_file_prefix);
		if (!src_file_prefix.empty() && 0 != pos) continue;
		pos = filename.rfind(src_file_suffix);
		if (!src_file_suffix.empty() && filename.length() != pos + src_file_suffix.length()) continue;

		boost::filesystem::path dst_path(dst_dir_path);
		//boost::filesystem::rename(*dir_itr, dst_path.append(filename));
		boost::filesystem::rename(*dir_itr, dst_path.append(filename.replace(pos, src_file_suffix.length(), dst_file_suffix)));
	}

	return true;
}

bool remove_files(const std::string &dir_path_name, const std::string &file_prefix, const std::string &file_suffix)
{
	boost::filesystem::path dir_path(dir_path_name);
	if (!boost::filesystem::exists(dir_path))
	{
		std::cerr << "Directory not found: " << dir_path_name << std::endl;
		return false;
	}

	std::string::size_type pos;
	boost::filesystem::directory_iterator end_iter;
	for (boost::filesystem::directory_iterator dir_itr(dir_path); dir_itr != end_iter; ++dir_itr)
	{
		if (!boost::filesystem::is_regular_file(*dir_itr)) continue;

		std::string filename(dir_itr->path().filename().generic_string());
		pos = filename.find(file_prefix);
		if (!file_prefix.empty() && 0 != pos) continue;
		pos = filename.rfind(file_suffix);
		if (!file_suffix.empty() && filename.length() != pos + file_suffix.length()) continue;

		boost::filesystem::remove(*dir_itr);
	}

	return true;
}

bool generate_file_name_list(const std::string &dir_path_name, const std::string &file_prefix, const std::string &file_suffix, std::list<std::string> &file_name_list)
{
	boost::filesystem::path dir_path(dir_path_name);
	if (!boost::filesystem::exists(dir_path))
	{
		std::cerr << "Directory not found: " << dir_path_name << std::endl;
		return false;
	}

	std::string::size_type pos1, pos2;
	boost::filesystem::directory_iterator end_iter;
	for (boost::filesystem::directory_iterator dir_itr(dir_path); dir_itr != end_iter; ++dir_itr)
	{
		if (!boost::filesystem::is_regular_file(*dir_itr)) continue;

		const std::string filename(dir_itr->path().filename().generic_string());
		pos1 = filename.find(file_prefix);
		if (!file_prefix.empty() && 0 != pos1) continue;
		pos2 = filename.rfind(file_suffix);
		if (!file_suffix.empty() && filename.length() != pos2 + file_suffix.length()) continue;

		file_name_list.push_back(filename.substr(pos1 + file_prefix.length(), pos2 - file_prefix.length()));
	}

	return true;
}

void crop_roi_from_image(const std::string &src_dataset_dir, const std::string &dst_dataset_dir, const std::list<std::string> &filename_list, const std::list<cv::Rect> &roi_list, const std::string &src_file_suffix, const std::string &dst_file_suffix)
{
	if (filename_list.size() != roi_list.size())
	{
		std::cerr << "ERROR: size unmatched." << std::endl;
		return;
	}

	size_t idx = 0;
	std::list<cv::Rect>::const_iterator citRoi = roi_list.begin();
	for (const auto &img_filename : filename_list)
	{
		if (0 == citRoi->width || 0 == citRoi->height) continue;

		//const cv::Mat img(cv::imread(src_dataset_dir + '/' + img_filename + src_file_suffix, cv::IMREAD_GRAYSCALE));
		const cv::Mat img(cv::imread(src_dataset_dir + '/' + img_filename + src_file_suffix, cv::IMREAD_UNCHANGED));

		const cv::Mat img2(img(*citRoi));
		
#if 0
		cv::imshow("FB Image", img2);
		cv::waitKey(0);
#endif

		cv::imwrite(dst_dataset_dir + '/' + img_filename + dst_file_suffix, img2);

		++citRoi;
	}
}

void adjust_image(const std::string &src_dataset_dir, const std::string &dst_dataset_dir, const std::list<std::string> &filename_list)
{
	for (const auto &img_filename : filename_list)
	{
		const cv::Mat img(cv::imread(src_dataset_dir + '/' + img_filename + std::string(".tif"), cv::IMREAD_GRAYSCALE));

		cv::Mat img2(img);
		img2.setTo(192, img >= 192);

		cv::imshow("Image", img2);
		cv::waitKey(0);

		cv::imwrite(dst_dataset_dir + '/' + img_filename + std::string(".tif"), img2);
	}
}

void generate_label_image_from_lear_masks(const std::string &src_dataset_dir, const std::string &dst_dataset_dir, const std::list<std::string> &filename_list, const std::list<size_t> &mask_count_list)
{
	if (filename_list.size() != mask_count_list.size())
	{
		std::cerr << "ERROR: size unmatched." << std::endl;
		return;
	}

	std::list<size_t>::const_iterator citMaskCount = mask_count_list.begin();
	for (const auto &img_filename : filename_list)
	{
		// Load image masks.
		std::vector<cv::Mat> masks;
		masks.reserve(*citMaskCount);
		for (size_t maskId = 0; maskId < *citMaskCount; ++maskId)
		{
			const std::string img_filepath(src_dataset_dir + '/' + img_filename + std::string(".mask.") + std::to_string(maskId) + std::string(".tif"));
			cv::Mat img(cv::imread(img_filepath, cv::IMREAD_GRAYSCALE));
			if (img.empty())
			{
				std::cerr << "File not found: " << img_filepath << std::endl;
				continue;
			}
			masks.push_back(img);
		}
		if (masks.empty() || masks.size() != *citMaskCount)
		{
			std::cerr << "Failed to load all images: " << img_filename << std::endl;
			continue;
		}

		// Accumulate all the masks.
		cv::Mat accumulated(cv::Mat::zeros(masks.front().size(), CV_16UC1));
		for (const auto &mask : masks)
		{
			cv::Mat tmp(cv::Mat::zeros(mask.size(), CV_16UC1));
			tmp.setTo(cv::Scalar::all(1), 0 == mask);
			accumulated += tmp;
		}
		
#if 1
		const std::string accumul_filepath(dst_dataset_dir + '/' + img_filename + std::string("_accumulated.tif"));
		cv::imwrite(accumul_filepath, accumulated);
#endif

		// Label the masks.
		unsigned short lbl = 1;
		cv::Mat label_img(cv::Mat::zeros(masks.front().size(), CV_16UC1));
		for (const auto &mask : masks)
		{
			label_img.setTo(cv::Scalar::all(lbl), 0 == mask);
			++lbl;
		}

		// Output the result.
		const std::string out_filepath(dst_dataset_dir + '/' + img_filename + std::string("_label.tif"));
		cv::imwrite(out_filepath, label_img);

		++citMaskCount;
	}
}

// REF [function] >> compute_boudary_weight() in ${SWL_CPP_HOME}/test/machine_vision_test/BoundaryExtractionTest.cpp
void compute_boundary_weight(const cv::Mat &boundary, cv::Mat &boundaryWeight)
{
	// Distance transform.
	cv::Mat dist;
	cv::Mat binary(cv::Mat::ones(boundary.size(), CV_8UC1));
	binary.setTo(cv::Scalar::all(0), boundary > 0);
	//cv::distanceTransform(binary, dist, cv::DIST_L2, cv::DIST_MASK_3);
	cv::distanceTransform(binary, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);

#if 0
	// Show the result.
	{
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(dist, NULL, &maxVal);
		cv::Mat dist_uchar;
		dist.convertTo(dist_uchar, CV_8UC1, 255.0 / maxVal, 0.0);
		cv::imshow("Boundary weight - Distance transform", dist_uchar);
	}
#endif

#if 1
	// Gaussian weighting.
	const double sigma2 = 20.0;  // sigma^2.
	cv::multiply(dist, dist, dist);
	cv::exp(-dist / (2.0 * sigma2), boundaryWeight);
#else
	// Linear weighting.
	double minVal = 0.0, maxVal = 0.0;
	cv::minMaxLoc(dist, NULL, &maxVal);
	boundaryWeight = maxVal - dist;
#endif

#if 0
	// Show the result.
	{
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(boundaryWeight, NULL, &maxVal);
		cv::Mat boundaryWeight_uchar;
		boundaryWeight.convertTo(boundaryWeight_uchar, CV_8UC1, 255.0 / maxVal, 0.0);
		cv::imshow("Boundary weight", boundaryWeight_uchar);
		//cv::imshow("Boundary weight", boundaryWeight);
	}
#endif
}

void create_foreground_mask_for_unet(const cv::Mat &label, const cv::Mat &border, cv::Mat &mask)
// mask = label - border.
{
	mask = cv::Mat::zeros(label.size(), CV_8UC1);
	mask.setTo(cv::Scalar::all(255), label > 0);
	mask.setTo(cv::Scalar::all(0), border > 0);
}

// REF [function] >> boundary_extraction() in ${SWL_CPP_HOME}/test/machine_vision_test/BoundaryExtractionTest.cpp
void extract_boundary_from_label_image(const std::string &src_dataset_dir, const std::string &dst_dataset_dir, const std::list<std::string> &filename_list, const std::string &src_file_suffix, const std::string &dst_file_suffix)
{
	// Create a boundary extractor.
	//swl::IBoundaryExtraction &extractor = swl::NaiveBoundaryExtraction(true);
	swl::IBoundaryExtraction &extractor = swl::ContourBoundaryExtraction();  // Slower.

	for (const auto &filename : filename_list)
	{
		// Load a label image.
		const std::string label_filepath(src_dataset_dir + '/' + filename + src_file_suffix);
		cv::Mat label(cv::imread(label_filepath, cv::IMREAD_UNCHANGED));
		if (label.empty())
		{
			std::cerr << "File not found: " << label_filepath << std::endl;
			continue;
		}

		if (1 != label.channels())
		{
			std::clog << "WARNING: Invalid channel of the label image: " << label.channels() << std::endl;

			cv::Mat label2(label);
			cv::cvtColor(label2, label, cv::COLOR_BGR2GRAY);  // TODO [enhance] >>
		}
		if (CV_16UC1 != label.type())
		{
			std::clog << "WARNING: Invalid type of the label image: " << label.type() << std::endl;

			cv::Mat label2(label);
			label = cv::Mat::zeros(label2.size(), CV_16UC1);
			label2.convertTo(label, CV_16UC1);
		}

		// Extract boundaries.
		cv::Mat boundary(cv::Mat::zeros(label.size(), label.type()));
		{
			boost::timer::auto_cpu_timer timer;
			extractor.extractBoundary(label, boundary);
		}

#if 0
		// For foreign body detection.
		cv::dilate(boundary, boundary, cv::Mat(), cv::Point(-1, -1), 3, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
		boundary.setTo(cv::Scalar::all(0), label > 1);  // Exclude insides of objects.
#elif 0
		cv::dilate(boundary, boundary, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
#endif

		// Output the result.
#if 0
		// N labels.
		const std::string out_filepath(dst_dataset_dir + '/' + filename + dst_file_suffix);
		cv::imwrite(out_filepath, boundary);  // Label boundary image.
#elif 0
		// For foreign body detection.
		// 3 labels: background & two types of boundaries.
		cv::Mat boundary_ushort(cv::Mat::zeros(boundary.size(), CV_16UC1));
		boundary_ushort.setTo(cv::Scalar::all(1), 1 == boundary);  // The boundaries of (object - background).
		boundary_ushort.setTo(cv::Scalar::all(2), boundary > 1);  // The boundaries of (object - foreign-body) & (foreign body - foreign-body).
		const std::string out_filepath(dst_dataset_dir + '/' + filename + dst_file_suffix);
		cv::imwrite(out_filepath, boundary_ushort);  // Simplified label boundary image.
#elif 0
		// 2 labels: foreground(boundaries) & background.
		cv::Mat boundary_uchar(cv::Mat::zeros(boundary.size(), CV_8UC1));
		boundary_uchar.setTo(cv::Scalar::all(255), boundary > 0);  // All boundaries.
		const std::string out_filepath(dst_dataset_dir + '/' + filename + dst_file_suffix);
		cv::imwrite(out_filepath, boundary_uchar);  // Boundary mask.
#elif 1
		// A boundary mask for U-Net.
		cv::Mat unet_bdry(boundary.size(), CV_8UC1, cv::Scalar::all(255));
		unet_bdry.setTo(cv::Scalar::all(0), boundary > 0);

		// Output the result.
		const std::string out_filepath(dst_dataset_dir + '/' + filename + dst_file_suffix);
		cv::imwrite(out_filepath, unet_bdry);
#endif

#if 0
		// Compute boundary weight.
		{
			cv::Mat boundaryWeight_float;
			compute_boundary_weight(boundary, boundaryWeight_float);

			// NOTICE [info] >> Cannot save images of 32-bit (signed/unsigned) integer or float.

			cv::Mat boundaryWeight_filtered(cv::Mat::zeros(boundary.size(), CV_16UC1));
			//double minVal = 0.0, maxVal = 0.0;
			//cv::minMaxLoc(boundaryWeight_float, &minVal, &maxVal);
			cv::Mat boundaryWeight_ushort;
			boundaryWeight_float.convertTo(boundaryWeight_ushort, boundaryWeight_filtered.type(), std::numeric_limits<unsigned short>::max(), 0.0);

			//boundaryWeight_filtered = boundaryWeight_ushort;  // Do not filter out.
			boundaryWeight_ushort.copyTo(boundaryWeight_filtered, boundary > 0 | 0 == label);  // On boundaries or outside of objects.
			//boundaryWeight_ushort.copyTo(boundaryWeight_filtered, boundary > 0 | 0 != label);  // On boundaries or inside of objects.

			//cv::imshow("Boundary extraction - Boundary weight", boundaryWeight_ushort);
			//cv::waitKey(0);
		}
#endif
	}
}

void extract_occlusion_border_from_label_image(const std::string &src_dataset_dir, const std::string &dst_dataset_dir, const std::list<std::string> &filename_list, const std::string &src_file_suffix, const std::string &dst_file_suffix)
{
	// Create a boundary extractor.
	swl::IBoundaryExtraction &extractor = swl::NaiveOcclusionBorderExtraction(true);

	for (const auto &filename : filename_list)
	{
		// Load a label image.
		const std::string label_filepath(src_dataset_dir + '/' + filename + src_file_suffix);
		cv::Mat label(cv::imread(label_filepath, cv::IMREAD_UNCHANGED));
		if (label.empty())
		{
			std::cerr << "File not found: " << label_filepath << std::endl;
			continue;
		}

		if (1 != label.channels())
		{
			std::clog << "WARNING: Invalid channel of the label image: " << label.channels() << std::endl;

			cv::Mat label2(label);
			cv::cvtColor(label2, label, cv::COLOR_BGR2GRAY);  // TODO [enhance] >>
		}
		if (label.type() != CV_16UC1)
		{
			std::cout << "WARNING: Invalid type of the label image: " << label.type() << std::endl;

			cv::Mat label2(label);
			label = cv::Mat::zeros(label2.size(), CV_16UC1);
			label2.convertTo(label, CV_16UC1);
		}

		// Extract occlusion borders.
		cv::Mat occlusionBorder(cv::Mat::zeros(label.size(), label.type()));
		{
			boost::timer::auto_cpu_timer timer;
			extractor.extractBoundary(label, occlusionBorder);
		}

#if 0
		// For foreign body detection.
		cv::dilate(occlusionBorder, occlusionBorder, cv::Mat(), cv::Point(-1, -1), 3, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
		occlusionBorder.setTo(cv::Scalar::all(0), label > 1);  // Exclude insides of objects.
#elif 0
		cv::dilate(occlusionBorder, occlusionBorder, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
#endif

		// Output the result.
#if 0
		// N labels.
		const std::string out_filepath(dst_dataset_dir + '/' + filename + dst_file_suffix);
		cv::imwrite(out_filepath, occlusionBorder);  // Label occlusion border image.
#elif 0
		// 2 labels: foreground(borders) & background.
		cv::Mat border_uchar(cv::Mat::zeros(occlusionBorder.size(), CV_8UC1));
		border_uchar.setTo(cv::Scalar::all(255), occlusionBorder > 0);  // All borders.
		const std::string out_filepath(dst_dataset_dir + '/' + filename + dst_file_suffix);
		cv::imwrite(out_filepath, border_uchar);  // Occlusion border mask.
#elif 1
		// A foreground mask for U-Net.
		cv::Mat unet_fg;
		create_foreground_mask_for_unet(label, occlusionBorder, unet_fg);

		// Output the result.
		const std::string out_filepath(dst_dataset_dir + '/' + filename + dst_file_suffix);
		cv::imwrite(out_filepath, unet_fg);
#endif
	}
}

void cvppp_2017(const std::string &dataset_home_dir)
{
#if 1
	// Create boundary images/masks.
	{
		const std::string src_dataset_dir(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1");
		const std::string src_file_prefix("");
		const std::string src_file_suffix("_label.png");
#if 0
		const std::string dst_dataset_dir(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1/label_bdry");
		//const std::string dst_dataset_dir(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1/boundary");
		const std::string dst_file_suffix("_bdry.png");
#elif 1
		const std::string dst_dataset_dir(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1/unet_bdry");
		const std::string dst_file_suffix("_unet_bdry.png");
#endif

		if (!local::create_directory(dst_dataset_dir))
		{
			std::cerr << "Failed to create a directory: " << dst_dataset_dir << std::endl;
			return;
		}
		std::list<std::string> file_name_list;
		if (!generate_file_name_list(src_dataset_dir, src_file_prefix, src_file_suffix, file_name_list) || file_name_list.empty())
		{
			std::cerr << "Failed to generate a file list." << std::endl;
			return;
		}

		extract_boundary_from_label_image(src_dataset_dir, dst_dataset_dir, file_name_list, src_file_suffix, dst_file_suffix);
	}
#endif
#if 1
	// Create occlusion border images/masks.
	{
		const std::string src_dataset_dir(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1");
		const std::string src_file_prefix("");
		const std::string src_file_suffix("_label.png");
#if 0
		const std::string dst_dataset_dir(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1/label_brdr");
		//const std::string dst_dataset_dir(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1/border");
		const std::string dst_file_suffix("_brdr.png");
#elif 1
		const std::string dst_dataset_dir(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1/unet_fg");
		const std::string dst_file_suffix("_unet_fg.png");
#endif

		if (!local::create_directory(dst_dataset_dir))
		{
			std::cerr << "Failed to create a directory: " << dst_dataset_dir << std::endl;
			return;
		}
		std::list<std::string> file_name_list;
		if (!generate_file_name_list(src_dataset_dir, src_file_prefix, src_file_suffix, file_name_list) || file_name_list.empty())
		{
			std::cerr << "Failed to generate a file list." << std::endl;
			return;
		}

		extract_occlusion_border_from_label_image(src_dataset_dir, dst_dataset_dir, file_name_list, src_file_suffix, dst_file_suffix);
	}
#endif
}

void hamburger_patty(const std::string &dataset_home_dir)
{
	const int IMG_WIDTH = 400, IMG_HEIGHT = 400;
	const std::list<size_t> mask_count_list({ 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3 });
	std::list<std::string> filename_list;
	std::list<cv::Rect> fb_roi_list;
	{
		filename_list.push_back(std::string("Bad_hamburger_161940_536171"));
		filename_list.push_back(std::string("Bad_hamburger_162100_616218"));
		filename_list.push_back(std::string("Bad_hamburger_162248_723984"));
		filename_list.push_back(std::string("Bad_hamburger_162943_1139484"));
		filename_list.push_back(std::string("Bad_hamburger_163951_1747437"));
		filename_list.push_back(std::string("Bad_hamburger_164148_1864171"));
		filename_list.push_back(std::string("Bad_hamburger_164202_1877953"));
		filename_list.push_back(std::string("Bad_hamburger_164233_1908890"));
		filename_list.push_back(std::string("Bad_hamburger_164452_2048218"));
		filename_list.push_back(std::string("Bad_hamburger_164501_2056812"));
		filename_list.push_back(std::string("Bad_hamburger_164516_2072171"));
		filename_list.push_back(std::string("Bad_hamburger_164528_2084562"));
		filename_list.push_back(std::string("Bad_hamburger_164544_2100062"));
		filename_list.push_back(std::string("Bad_hamburger_164558_2114046"));
		filename_list.push_back(std::string("Bad_hamburger_164620_2136000"));
		filename_list.push_back(std::string("Bad_hamburger_164739_2214828"));

		fb_roi_list.push_back(cv::Rect());
		fb_roi_list.push_back(cv::Rect());
		fb_roi_list.push_back(cv::Rect());
		fb_roi_list.push_back(cv::Rect(279, 276, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect(290, 376, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect(233, 258, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect());
		fb_roi_list.push_back(cv::Rect());
		fb_roi_list.push_back(cv::Rect(169, 298, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect(185, 305, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect(297, 262, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect(240, 297, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect(217, 300, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect(193, 270, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect(185, 294, IMG_WIDTH, IMG_HEIGHT));
		fb_roi_list.push_back(cv::Rect(198, 316, IMG_WIDTH, IMG_HEIGHT));
	}

#if 0
	// Adjust images.
	{
		const std::string src_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/image");
		const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/image_adjusted");

		adjust_image(src_dataset_dir, dst_dataset_dir, filename_list);
	}
#endif

#if 0
	// Generate label images from LEAR masks.
	{
		const std::string src_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/lear/processed2");
		const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/label");

		generate_label_image_from_lear_masks(src_dataset_dir, dst_dataset_dir, filename_list, mask_count_list);
	}
#endif

#if 0
	// Create boundary images/masks.
	{
		const std::string src_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/label");
		const std::string src_file_suffix("_label.tif");
#if 0
		const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/label_bdry");
		//const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/boundary");
		const std::string dst_file_suffix("_bdry.tif");
#elif 1
		const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/unet_bdry");
		const std::string dst_file_suffix("_unet_bdry.tif");
#endif

		extract_boundary_from_label_image(src_dataset_dir, dst_dataset_dir, filename_list, src_file_suffix, dst_file_suffix));
	}
#endif
#if 0
	// Create occlusion border images/masks.
	{
		const std::string src_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/label");
		const std::string src_file_suffix("_label.tif");
#if 0
		const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/label_brdr");
		//const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/border");
		const std::string dst_file_suffix("_brdr.tif");
#elif 1
		const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/unet_fg");
		const std::string dst_file_suffix("_unet_fg.tif");
#endif

		extract_occlusion_border_from_label_image(src_dataset_dir, dst_dataset_dir, filename_list, src_file_suffix, dst_file_suffix);
	}
#endif

#if 0
	// Crop patty images to extract FB regions.
	{
#if 0
		const std::string src_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/image");
		const std::string src_file_suffix(".tif");
		const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/FB_trimmed/image");
		const std::string dst_file_suffix(".tif");
#elif 1
		const std::string src_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/unet_fg");
		const std::string src_file_suffix("_unet_fg.tif");
		const std::string dst_dataset_dir(dataset_home_dir + "/failure_analysis/defect/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/FB_trimmed/unet_fg");
		const std::string dst_file_suffix("_unet_fg.tif");
#endif

		crop_roi_from_image(src_dataset_dir, dst_dataset_dir, filename_list, fb_roi_list, src_file_suffix, dst_file_suffix);
	}
#endif
}

}  // namespace local
}  // unnamed namespace

int main()
{
	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));
		cv::theRNG();

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) ||defined(WIN32)
		const std::string dataset_home_dir("D:/dataset");
#elif defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)
		const std::string dataset_home_dir("/home/sangwook/my_dataset");
		//const std::string dataset_home_dir("/home/HDD1/sangwook/my_dataset");
#else
		const std::string dataset_home_dir("/dataset");
#endif

#if 0
		// Basic utility.
		{
			const std::string src_dir_path_name(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1");
			const std::string src_file_prefix("");
			const std::string src_file_suffix("_rgb.png");
			const std::string dst_dir_path_name(dataset_home_dir + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1/unet_fg");
			const std::string dst_file_suffix("_rgb_unet_fg.png");

			if (!local::create_directory(dst_dir_path_name))
			{
				std::cerr << "Failed to create a directory: " << dst_dir_path_name << std::endl;
				return EXIT_FAILURE;
			}

#if 0
			if (!local::copy_files(src_dir_path_name, dst_dir_path_name, src_file_prefix, src_file_suffix, dst_file_suffix))
			{
				std::cerr << "Failed to copy files from " << src_dir_path_name << " to " << dst_dir_path_name << std::endl;
				return EXIT_FAILURE;
			}
#endif
#if 0
			if (!local::rename_files(src_dir_path_name, dst_dir_path_name, src_file_prefix, src_file_suffix, dst_file_suffix))
			{
				std::cerr << "Failed to rename files from " << src_dir_path_name << " to " << dst_dir_path_name << std::endl;
				return EXIT_FAILURE;
			}
#endif
#if 0
			if (!local::remove_files(src_dir_path_name, src_file_prefix, src_file_suffix))
			{
				std::cerr << "Failed to remove files from " << src_dir_path_name << std::endl;
				return EXIT_FAILURE;
			}
#endif
		}
#endif

		// Phenotyping.
		//local::cvppp_2017(dataset_home_dir);

		// Defect analysis.
		//local::hamburger_patty(dataset_home_dir);
	}
	catch (const cv::Exception &ex)
	{
		//std::cout << "OpenCV exception caught: " << ex.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(ex.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tDescription: " << ex.err << std::endl
			<< "\tLine:        " << ex.line << std::endl
			<< "\tFunction:    " << ex.func << std::endl
			<< "\tFile:        " << ex.file << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::bad_alloc &ex)
	{
		std::cout << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cout << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
