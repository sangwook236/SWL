//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>


int main(int argc, char *argv[])
{
	void tree_traversal();
	void graph_traversal();
	void graph_algorithm();

	void levenshtein_distance();
	void dynamic_time_warping();

	void hough_transform();
	void line2d_estimation_using_ransac();
	void circle2d_estimation_using_ransac();
	void plane3d_estimation_using_ransac();
	void quadratic2d_estimation_using_ransac();

	void rejection_sampling();
	void sampling_importance_resampling();
	void metropolis_hastings_algorithm();

	void kalman_filter();
	void extended_kalman_filter();
	void unscented_kalman_filter();
	void unscented_kalman_filter_with_additive_noise();

	void univariate_normal_mixture_model();
	void multivariate_normal_mixture_model();
	void von_mises_mixture_model();

	void hmm_with_multinomial_observation_densities();
	void hmm_with_univariate_normal_observation_densities();
	void hmm_with_univariate_normal_mixture_observation_densities();
	void hmm_with_von_mises_observation_densities();
	void hmm_with_von_mises_mixture_observation_densities();

	void ar_hmm_with_univariate_normal_observation_densities();
	void ar_hmm_with_univariate_normal_mixture_observation_densities();

	void hmm_segmentation();
	void crf_segmentation();

	int retval = EXIT_SUCCESS;
	try
	{
		// Tree traversal -------------------------------------------
		//tree_traversal();

		// Graph traversal ------------------------------------------
		//graph_traversal();

		// Graph algorithm ------------------------------------------
		graph_algorithm();

		// Distance measure -----------------------------------------
		//levenshtein_distance();  // Levenshtein / edit distance.
		//dynamic_time_warping();  // Dynamic time warping (DTW).

		// Estimation -----------------------------------------------
		//hough_transform();

		// Robust estimation ----------------------------------------
		//line2d_estimation_using_ransac();
		//circle2d_estimation_using_ransac();
		//plane3d_estimation_using_ransac();

		//	- Use more samples(10) than the minimal size(3) required to estimate model parameters.
		//		Apply least squares method to estimated model parameters.
		//		REF [function] >> Quadratic2RansacEstimator::estimateModel().
		//	- Verify an estimated model based on anchor points.
		//		Can make use of anchor points to estimate a model itself.
		//		REF [function] >> Quadratic2RansacEstimator::verifyModel().
		//	- Refine an estimated model using inliers.
		//		REF [function] >> Quadratic2RansacEstimator::estimateModelFromInliers().
		//quadratic2d_estimation_using_ransac();

		// Sampling / Resampling ------------------------------------
		//rejection_sampling();
		//sampling_importance_resampling();  // Sequential importance sampling (SIS), sampling importance resampling (SIR), particle filter, bootstrap filter.
		//metropolis_hastings_algorithm();  // Markov chain Monte Carlo (MCMC).

		// Bayesian filtering ---------------------------------------
		//kalman_filter();
		//extended_kalman_filter();
		//unscented_kalman_filter();
		//unscented_kalman_filter_with_additive_noise();

		// Mixture model (MM) ---------------------------------------
		//univariate_normal_mixture_model();
		//multivariate_normal_mixture_model();
		//von_mises_mixture_model();

		// Hidden Markov model (HMM) --------------------------------
		//hmm_with_multinomial_observation_densities();
		//hmm_with_univariate_normal_observation_densities();
		//hmm_with_univariate_normal_mixture_observation_densities();
		//hmm_with_von_mises_observation_densities();
		//hmm_with_von_mises_mixture_observation_densities();

		// Autoregressive hidden Markov model (AR HMM) --------------
		//ar_hmm_with_univariate_normal_observation_densities();
		//ar_hmm_with_univariate_normal_mixture_observation_densities();

		//-----------------------------------------------------------
		// Application.

		// HMM segmentation -----------------------------------------
		//hmm_segmentation();  // Not yet implemented.
		//crf_segmentation();  // Not yet implemented.
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	//std::cout << "Press any key to exit ..." << std::endl;
	//std::cin.get();

	return retval;
}

void output_data_to_file(std::ostream &stream, const std::string &variable_name, const std::vector<double> &data)
{
	stream << variable_name.c_str() << " = [" << std::endl;
	for (std::vector<double>::const_iterator cit = data.begin(); cit != data.end(); ++cit)
		stream << *cit << std::endl;
	stream << "];" << std::endl;
}
