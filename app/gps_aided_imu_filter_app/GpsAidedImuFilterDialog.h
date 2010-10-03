#pragma once

#include "Adis16350Interface.h"
#include "GpsInterface.h"
#include "GpsAidedImuFilterRunner.h"
#include <boost/smart_ptr.hpp>
#include <string>


// CGpsAidedImuFilterDialog dialog

class CGpsAidedImuFilterDialog : public CDialog
{
	DECLARE_DYNAMIC(CGpsAidedImuFilterDialog)

public:
	CGpsAidedImuFilterDialog(CWnd* pParent = NULL);   // standard constructor
	virtual ~CGpsAidedImuFilterDialog();

// Dialog Data
	enum { IDD = IDD_GPS_AIDED_IMU_FILTER_DIALOG };

private:
	bool initializeSensors();
	void saveRawData();

	void checkImu();
	void checkGps();
	void runFilter();

private:
	static const UINT IMU_TIMER_ID = 1;
	static const UINT GPS_TIMER_ID = 2;
	static const UINT FILTER_TIMER_ID = 3;
	static const UINT SAVER_TIMER_ID = 4;

	static const size_t SAVER_SAMPLING_INTERVAL = 50;
	static const size_t IMU_SAMPLING_INTERVAL = 50;
	static const size_t GPS_SAMPLING_INTERVAL = 100;
	static const size_t FILTER_SAMPLING_INTERVAL = 50;

#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring gpsPortName_;
#else
	const std::string gpsPortName_;
#endif
	const unsigned int gpsBaudRate_;

	boost::scoped_ptr<swl::Adis16350Interface> imu_;
	boost::scoped_ptr<swl::GpsInterface> gps_;
	boost::scoped_ptr<swl::GpsAidedImuFilterRunner> runner_;

	swl::ImuData::Accel initialGravity_;
	swl::ImuData::Gyro initialAngularVel_;
	swl::EarthData::ECEF initialGpsECEF_;
	swl::EarthData::Speed initialGpsSpeed_;

	LARGE_INTEGER freq_;
	LARGE_INTEGER prevPerformanceCount_;
	swl::EarthData::Time prevGpsUtc_;
	swl::EarthData::ECEF prevGpsECEF_;

	bool isSensorsInitialized_;
	size_t step_;

	std::list<__int64> imuTimeStamps_;
	std::list<swl::ImuData::Accel> imuAccels_;
	std::list<swl::ImuData::Gyro> imuGyros_;
	std::list<long> gpsTimeStamps_;
	std::list<swl::EarthData::Geodetic> gpsGeodetics_;
	std::list<swl::EarthData::Speed> gpsSpeeds_;

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
public:
	virtual BOOL OnInitDialog();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnBnClickedButtonInitializeImuAndGps();
	afx_msg void OnBnClickedButtonSaveRawData();
	afx_msg void OnBnClickedButtonCheckImu();
	afx_msg void OnBnClickedButtonCheckGps();
	afx_msg void OnBnClickedButtonRunFilter();
};
