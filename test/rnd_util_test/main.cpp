#include "stdafx.h"
#include <iostream>

int main()
{
	void estimate_3d_plane_using_ransac();
	void kalman_filter();
	void extended_kalman_filter();
	void imu_kalman_filter();

	//estimate_3d_plane_using_ransac();

	//kalman_filter();
	//extended_kalman_filter();

	imu_kalman_filter();

	std::wcout << L"press any key to exit !!!" << std::endl;
	std::wcout.flush();
	std::wcin.get();

    return 0;
}
