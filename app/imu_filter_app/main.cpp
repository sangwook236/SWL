#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <string>
#include <vector>
#include <iostream>


int main(int argc, char *argv[])
{
	void imu_kalman_filter();
	void imu_calibration();
	void imu_extended_Kalman_filter_with_calibration();
	void imu_unscented_Kalman_filter_with_calibration();

	try
	{
		//imu_kalman_filter();
		//imu_calibration();
		//imu_extended_Kalman_filter_with_calibration();
		//imu_unscented_Kalman_filter_with_calibration();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		std::cin.get();
		return -1;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		std::cin.get();
		return -1;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}

void output_data_to_file(std::ostream &stream, const std::string &variable_name, const std::vector<double> &data)
{
	stream << variable_name.c_str() << " = [" << std::endl;
	for (std::vector<double>::const_iterator cit = data.begin(); cit != data.end(); ++cit)
		stream << *cit << std::endl;
	stream << "];" << std::endl;
}
