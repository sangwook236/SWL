//#include "stdafx.h"
#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/space/grid_space.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#include <opengm/inference/auxiliary/minstcutboost.hxx>
#include <opengm/functions/function_registration.hxx>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/timer/timer.hpp>


//#define __USE_GRID_SPACE 1
#define __USE_8_NEIGHBORHOOD_SYSTEM 1

namespace {
namespace local {
	
double compute_constrast_parameter(const cv::Mat &rgb_img)
{
	const std::size_t Nx = rgb_img.cols;  // width of the grid
	const std::size_t Ny = rgb_img.rows;  // height of the grid

	double sum = 0.0;
	std::size_t count = 0;
	// An 4-neighborhood or 8-neighborhood system in 2D (4-connectivity or 8-connectivity).
	for (std::size_t y = 0; y < Ny; ++y)
	{
		for (std::size_t x = 0; x < Nx; ++x)
		{
			const cv::Vec3b &pix1 = rgb_img.at<cv::Vec3b>(y, x);
			if (x + 1 < Nx)  // (x, y) -- (x + 1, y)
			{
				const cv::Vec3b &pix2 = rgb_img.at<cv::Vec3b>(y, x + 1);
				const double norm = cv::norm(pix1 - pix2);
				sum += norm * norm;
				++count;
			}
			if (y + 1 < Ny)  // (x, y) -- (x, y + 1)
			{
				const cv::Vec3b &pix2 = rgb_img.at<cv::Vec3b>(y + 1, x);
				const double norm = cv::norm(pix1 - pix2);
				sum += norm * norm;
				++count;
			}
#if defined(__USE_8_NEIGHBORHOOD_SYSTEM)
			if (x + 1 < Nx && y + 1 < Ny)  // (x, y) -- (x + 1, y + 1)
			{
				const cv::Vec3b &pix2 = rgb_img.at<cv::Vec3b>(y + 1, x + 1);
				const double norm = cv::norm(pix1 - pix2);
				sum += norm * norm;
				++count;
			}
			if (x + 1 < Nx && int(y - 1) >= 0)  // (x, y) -- (x + 1, y - 1)
			{
				const cv::Vec3b &pix2 = rgb_img.at<cv::Vec3b>(y - 1, x + 1);
				const double norm = cv::norm(pix1 - pix2);
				sum += norm * norm;
				++count;
			}
#endif  // __USE_8_NEIGHBORHOOD_SYSTEM
		}
	}

	return 0.5 * count / sum;
}

#if defined(__USE_GRID_SPACE)
typedef opengm::GridSpace<std::size_t, std::size_t> Space;
#else
typedef opengm::SimpleDiscreteSpace<std::size_t, std::size_t> Space;
#endif
typedef opengm::ExplicitFunction<double> ExplicitFunction;

// construct a graphical model with
// - addition as the operation (template parameter Adder)
typedef opengm::GraphicalModel<double, opengm::Adder, ExplicitFunction, Space> GraphicalModel;

// this function maps a node (x, y) in the grid to a unique variable index
inline std::size_t getVariableIndex(const std::size_t Nx, const std::size_t x, const std::size_t y)
{
	return x + Nx * y;
}

// [ref]
//	createGraphicalModelForPottsModel() in ${CPP_RND_HOME}/test/probabilistic_graphical_model/opengm/opengm_inference_algorithms.cpp
//	${OPENGM_HOME}/src/examples/image-processing-examples/grid_potts.cxx
//	"Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images", Yuri Y. Boykov and Marie-Pierre Jolly, ICCV, 2001.
bool create_single_layered_graphical_model(const cv::Mat &rgb_img, const cv::Mat &depth_img, const std::size_t numOfLabels, const double lambda, const double lambda_rgb, const double lambda_depth, const cv::Mat &histForeground_rgb, const cv::Mat &histBackground_rgb, const cv::Mat &histForeground_depth, const cv::Mat &histBackground_depth, GraphicalModel &gm)
{
	// model parameters (global variables are used only in example code)
	const std::size_t Nx = rgb_img.cols;  // width of the grid
	const std::size_t Ny = rgb_img.rows;  // height of the grid

	// construct a label space with
	//	- Nx * Ny variables
	//	- each having numOfLabels many labels
#if defined(__USE_GRID_SPACE)
	Space space(Nx, Ny, numOfLabels);
#else
	Space space(Nx * Ny, numOfLabels);
#endif

	gm = GraphicalModel(space);

	// constrast term.
	const double inv_beta = compute_constrast_parameter(rgb_img);

	const double tol = 1.0e-50;
	//const double minVal = std::numeric_limits<double>::min();
	const double minVal = tol;
	const std::size_t shape1[] = { numOfLabels };
	const std::size_t shape2[] = { numOfLabels, numOfLabels };
	for (std::size_t y = 0; y < Ny; ++y)
	{
		for (std::size_t x = 0; x < Nx; ++x)
		{
			// Add 1st order functions and factors.
			// For each node (x, y) in the grid, i.e. for each variable variableIndex(Nx, x, y) of the model,
			// add one 1st order functions and one 1st order factor
			{
				const cv::Vec3b &rgb_pix = rgb_img.at<cv::Vec3b>(y, x);
				const double probForeground_rgb = (double)histForeground_rgb.at<float>(rgb_pix[0], rgb_pix[1], rgb_pix[2]);
				assert(0.0 <= probForeground_rgb && probForeground_rgb <= 1.0);
				const double probBackground_rgb = (double)histBackground_rgb.at<float>(rgb_pix[0], rgb_pix[1], rgb_pix[2]);
				assert(0.0 <= probBackground_rgb && probBackground_rgb <= 1.0);

				const unsigned short &depth_pix = depth_img.at<unsigned short>(y, x);
				const double probForeground_depth = (double)histForeground_depth.at<float>(depth_pix);
				assert(0.0 <= probForeground_depth && probForeground_depth <= 1.0);
				const double probBackground_depth = (double)histBackground_depth.at<float>(depth_pix);
				assert(0.0 <= probBackground_depth && probBackground_depth <= 1.0);

				// function
				ExplicitFunction func1(shape1, shape1 + 1);
				const double probBackground = lambda_rgb * probBackground_rgb + lambda_depth * probBackground_depth;
				const double probForeground = lambda_rgb * probForeground_rgb + lambda_depth * probForeground_depth;
				func1(0) = -std::log(std::fabs(probBackground) > tol ? probBackground : minVal);  // background (state = 1)
				func1(1) = -std::log(std::fabs(probForeground) > tol ? probForeground : minVal);  // foreground (state = 0)

				const GraphicalModel::FunctionIdentifier fid1 = gm.addFunction(func1);

				// factor
				const std::size_t variableIndices[] = { getVariableIndex(Nx, x, y) };
				gm.addFactor(fid1, variableIndices, variableIndices + 1);
			}

			// Add 2nd order functions and factors.
			// For each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
			// add one factor that connects the corresponding variable indices.
			// An 4-neighborhood or 8-neighborhood system in 2D (4-connectivity or 8-connectivity).
			{
				// factor
				const cv::Vec3b &pix1 = rgb_img.at<cv::Vec3b>(y, x);
				if (x + 1 < Nx)  // (x, y) -- (x + 1, y)
				{
					const cv::Vec3b &pix2 = rgb_img.at<cv::Vec3b>(y, x + 1);
					const double norm = cv::norm(pix2 - pix1);
					const double B = lambda * std::exp(-norm * norm * inv_beta);

					// function
					ExplicitFunction func2(shape2, shape2 + 2);
					func2(0, 0) = 0.0;
					func2(0, 1) = B;
					func2(1, 0) = B;
					func2(1, 1) = 0.0;
					const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x + 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (y + 1 < Ny)  // (x, y) -- (x, y + 1)
				{
					const cv::Vec3b &pix2 = rgb_img.at<cv::Vec3b>(y + 1, x);
					const double norm = cv::norm(pix2 - pix1);
					const double B = lambda * std::exp(-norm * norm * inv_beta);

					// function
					ExplicitFunction func2(shape2, shape2 + 2);
					func2(0, 0) = 0.0;
					func2(0, 1) = B;
					func2(1, 0) = B;
					func2(1, 1) = 0.0;
					const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
#if defined(__USE_8_NEIGHBORHOOD_SYSTEM)
				if (x + 1 < Nx && y + 1 < Ny)  // (x, y) -- (x + 1, y + 1)
				{
					const cv::Vec3b &pix2 = rgb_img.at<cv::Vec3b>(y + 1, x + 1);
					const double norm = cv::norm(pix2 - pix1);
					const double B = lambda * std::exp(-norm * norm * inv_beta);

					// function
					ExplicitFunction func2(shape2, shape2 + 2);
					func2(0, 0) = 0.0;
					func2(0, 1) = B;
					func2(1, 0) = B;
					func2(1, 1) = 0.0;
					const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x + 1, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (x + 1 < Nx && int(y - 1) >= 0)  // (x, y) -- (x + 1, y - 1)
				{
					const cv::Vec3b &pix2 = rgb_img.at<cv::Vec3b>(y - 1, x + 1);
					const double norm = cv::norm(pix2 - pix1);
					const double B = lambda * std::exp(-norm * norm * inv_beta);

					// function
					ExplicitFunction func2(shape2, shape2 + 2);
					func2(0, 0) = 0.0;
					func2(0, 1) = B;
					func2(1, 0) = B;
					func2(1, 1) = 0.0;
					const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x + 1, y - 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
#endif  // __USE_8_NEIGHBORHOOD_SYSTEM
			}
		}
	}

	return true;
}

// [ref] run_inference_algorithm() in ${CPP_RND_HOME}/test/probabilistic_graphical_model/opengm/opengm_inference_algorithms.cpp
template<typename GraphicalModel, typename InferenceAlgorithm>
void run_inference_algorithm(InferenceAlgorithm &algorithm, std::vector<typename GraphicalModel::LabelType> &labelings)
{
	// optimize (approximately)
	typename InferenceAlgorithm::VerboseVisitorType visitor;
	//typename InferenceAlgorithm::TimingVisitorType visitor;
	//typename InferenceAlgorithm::EmptyVisitorType visitor;
	std::cout << "start inferring ..." << std::endl;
	{
		boost::timer::auto_cpu_timer timer;
		algorithm.infer(visitor);
	}
	std::cout << "end inferring ..." << std::endl;
	std::cout << "value: " << algorithm.value() << ", bound: " << algorithm.bound() << std::endl;

	// obtain the (approximate) argmax.
	algorithm.arg(labelings);
}

}  // namespace local
}  // unnamed namespace

namespace swl {

// [ref] Util.cpp
void normalize_histogram(cv::MatND &hist, const double factor);

// [ref] EfficientGraphBasedImageSegmentation.cpp
void segment_image_using_efficient_graph_based_image_segmentation_algorithm(
	const cv::Mat &rgb_input_image, const cv::Mat &depth_input_image, const cv::Mat &depth_guided_mask,
	const float sigma, const float k, const int min_size,
	const float lambda1, const float lambda2, const float lambda3, const float fx_rgb, const float fy_rgb,
	int &num_ccs, cv::Mat &output_image
);
void segment_image_using_efficient_graph_based_image_segmentation_algorithm(const cv::Mat &rgb_input_image, const float sigma, const float k, const int min_size, int &num_ccs, cv::Mat &output_image);

void run_efficient_graph_based_image_segmentation(const cv::Mat &rgb_input_image, const cv::Mat &depth_input_image, const cv::Mat &depth_guided_mask, const double fx_rgb, const double fy_rgb)
{
	const float sigma = 0.5f;
	const float k = 500.0f;
	const int min_size = 50;

	int num_ccs = 0;
	cv::Mat output_image;
#if 0
	//const float lambda1 = 1.0f, lambda2 = 1.0f, lambda3 = 1.0f;
	//const float lambda1 = 0.1f, lambda2 = 0.1f, lambda3 = 0.1f;
	//const float lambda1 = 0.5f, lambda2 = 0.5f, lambda3 = 0.5f;
	const float lambda1 = 0.01f, lambda2 = 0.0f, lambda3 = 0.0f;
	//const float lambda1 = 0.0f, lambda2 = 0.0f, lambda3 = 0.0f;

	segment_image_using_efficient_graph_based_image_segmentation_algorithm(rgb_input_image, depth_input_image, depth_guided_mask, sigma, k, min_size, lambda1, lambda2, lambda3, (float)fx_rgb, (float)fy_rgb, num_ccs, output_image);

#if 1
	std::cout << "got " << num_ccs << " components" << std::endl;
	cv::imshow("result of depth-guided efficient graph based image segmentation algorithm", output_image);
#endif
#else
	segment_image_using_efficient_graph_based_image_segmentation_algorithm(rgb_input_image, sigma, k, min_size, num_ccs, output_image);

#if 1
	std::cout << "got " << num_ccs << " components" << std::endl;
	cv::imshow("result of the orignal efficient graph based image segmentation algorithm", output_image);

	//cv::Mat tmp_image(output_image.size(), output_image.type(), cv::Scalar::all(0));
	//output_image.copyTo(tmp_image, SWL_PR_FGD == depth_guided_mask);
	//cv::imshow("boundry region of the result of the original efficient graph based image segmentation algorithm", tmp_image);
#endif
#endif
}

void run_binary_segmentation_using_min_cut(const cv::Mat &rgb_input_image, const cv::Mat &depth_input_image, const cv::Mat &foreground_mask, const cv::Mat &background_mask, const cv::Mat &foreground_info_mask, const cv::Mat &background_info_mask)
{
	// foreground & background probability distributions
	cv::MatND histForeground_rgb, histBackground_rgb;  // CV_32FC1, 3-dim (rows = bins1, cols = bins2, 3-dim = bins3)
	{
		const int dims = 3;
		const int bins1 = 256, bins2 = 256, bins3 = 256;
		const int histSize[] = { bins1, bins2, bins3 };
		const float range1[] = { 0, 256 };
		const float range2[] = { 0, 256 };
		const float range3[] = { 0, 256 };
		const float *ranges[] = { range1, range2, range3 };
		const int channels[] = { 0, 1, 2 };

		// calculate histograms.
		cv::calcHist(
			&rgb_input_image, 1, channels, foreground_info_mask,
			histForeground_rgb, dims, histSize, ranges,
			true, // the histogram is uniform
			false
		);
		cv::calcHist(
			&rgb_input_image, 1, channels, background_info_mask,
			histBackground_rgb, dims, histSize, ranges,
			true, // the histogram is uniform
			false
		);

		// normalize histograms.
		normalize_histogram(histForeground_rgb, 1.0);
		normalize_histogram(histBackground_rgb, 1.0);
	}

	// foreground & background probability distributions
	cv::MatND histForeground_depth, histBackground_depth;  // CV_32FC1, 1-dim (rows = bins, cols = 1)
	{
		double minVal, maxVal;
		cv::minMaxLoc(depth_input_image, &minVal, &maxVal);

		const int dims = 1;
		const int bins = 256;
		const int histSize[] = { bins };
		const float range[] = { (int)std::floor(minVal), (int)std::ceil(maxVal) + 1 };
		const float *ranges[] = { range };
		const int channels[] = { 0 };

		// calculate histograms.
		cv::calcHist(
			&depth_input_image, 1, channels, foreground_info_mask,
			histForeground_depth, dims, histSize, ranges,
			true, // the histogram is uniform
			false
		);
		cv::calcHist(
			&depth_input_image, 1, channels, background_info_mask,
			histBackground_depth, dims, histSize, ranges,
			true, // the histogram is uniform
			false
		);

		// normalize histograms.
		normalize_histogram(histForeground_depth, 1.0);
		normalize_histogram(histBackground_depth, 1.0);
	}

	// create graphical model.
	local::GraphicalModel gm;
	const std::size_t numOfLabels = 2;
	const double lambda = 0.2;
	const double lambda_rgb = 1.0;  // [0, 1]
	const double lambda_depth = 1.0 - lambda_rgb;  // [0, 1]
	if (local::create_single_layered_graphical_model(rgb_input_image, depth_input_image, numOfLabels, lambda, lambda_rgb, lambda_depth, histForeground_rgb, histBackground_rgb, histForeground_depth, histBackground_depth, gm))
		std::cout << "A single-layered graphical model for binary segmentation is created." << std::endl;
	else
	{
		std::cout << "A single-layered graphical model for binary segmentation fails to be created." << std::endl;
		return;
	}

	// run inference algorithm.
	std::vector<local::GraphicalModel::LabelType> labelings(gm.numberOfVariables());
	{
#if 1
		typedef opengm::external::MinSTCutKolmogorov<std::size_t, double> MinStCut;
		typedef opengm::GraphCut<local::GraphicalModel, opengm::Minimizer, MinStCut> MinGraphCut;

		MinGraphCut mincut(gm);
#else
		typedef opengm::MinSTCutBoost<std::size_t, long, opengm::PUSH_RELABEL> MinStCut;
		typedef opengm::GraphCut<GraphicalModel, opengm::Minimizer, MinStCut> MinGraphCut;

		const MinGraphCut::ValueType scale = 1000000;
		const MinGraphCut::Parameter parameter(scale);
		MinGraphCut mincut(gm, parameter);
#endif

		local::run_inference_algorithm<local::GraphicalModel>(mincut, labelings);
	}

	// output results.
	{
#if 1
		cv::Mat label_img(rgb_input_image.size(), CV_8UC1, cv::Scalar::all(0));
		for (local::GraphicalModel::IndexType row = 0; row < (std::size_t)label_img.rows; ++row)
			for (local::GraphicalModel::IndexType col = 0; col < (std::size_t)label_img.cols; ++col)
				label_img.at<unsigned char>(row, col) = (unsigned char)(255 * labelings[local::getVariableIndex(label_img.cols, col, row)] / (numOfLabels - 1));

		cv::imshow("interactive graph cuts - labeling", label_img);
#elif 0
		cv::Mat label_img(rgb_input_image.size(), CV_16UC1, cv::Scalar::all(0));
		for (local::GraphicalModel::IndexType row = 0; row < label_img.rows; ++row)
			for (local::GraphicalModel::IndexType col = 0; col < label_img.cols; ++col)
				label_img.at<unsigned short>(row, col) = (unsigned short)labelings[local::getVariableIndex(label_img.cols, col, row)];

		cv::imshow("interactive graph cuts - labeling", label_img);
#else
		std::cout << algorithm.name() << " has found the labeling ";
		for (typename local::GraphicalModel::LabelType i = 0; i < labeling.size(); ++i)
			std::cout << labeling[i] << ' ';
		std::cout << std::endl;
#endif
	}
}

}  // namespace swl
