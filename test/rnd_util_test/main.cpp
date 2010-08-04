#include "stdafx.h"
#include <string>
#include <vector>
#include <iostream>


int main()
{
	void hough_transform();
	void estimate_3d_plane_using_ransac();
	void sampling_importance_resampling();
	void metropolis_hastings_algorithm();
	void kalman_filter();
	void extended_kalman_filter();
	void unscented_kalman_filter();
	void unscented_kalman_filter_with_additive_noise();
	void imu_kalman_filter();
	void imu_calibration();
	void imu_filter_with_calibration();

	//hough_transform();
	//estimate_3d_plane_using_ransac();

	//sampling_importance_resampling();  // sequential importance sampling (SIS), sampling importance resampling (SIR), particle filter, bootstrap filter
	//metropolis_hastings_algorithm();  // Markov chain Monte Carlo (MCMC)

	//kalman_filter();
	//extended_kalman_filter();
	//unscented_kalman_filter();
	//unscented_kalman_filter_with_additive_noise();

	//imu_kalman_filter();
	//imu_calibration();
	imu_filter_with_calibration();

	std::wcout << L"press any key to exit !!!" << std::endl;
	std::wcout.flush();
	std::wcin.get();

    return 0;
}

void output_data_to_file(std::ostream &stream, const std::string &variable_name, const std::vector<double> &data)
{
	stream << variable_name.c_str() << " = [" << std::endl;
	for (std::vector<double>::const_iterator cit = data.begin(); cit != data.end(); ++cit)
		stream << *cit << std::endl;
	stream << "];" << std::endl;
}
