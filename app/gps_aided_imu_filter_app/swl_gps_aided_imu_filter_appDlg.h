
// swl_gps_aided_imu_filter_appDlg.h : header file
//

#pragma once

#include "Adis16350Interface.h"
#include "GpsInterface.h"
#include "GpsAidedImuFilterRunner.h"
#include <boost/smart_ptr.hpp>
#include <string>


// Cswl_gps_aided_imu_filter_appDlg dialog
class Cswl_gps_aided_imu_filter_appDlg : public CDialog
{
// Construction
public:
	Cswl_gps_aided_imu_filter_appDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_SWL_GPS_AIDED_IMU_FILTER_APP_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support

private:
	void checkImu();
	void checkGps();
	void runFilter();
	void saveRawData();

private:
	static const UINT IMU_TIMER_ID = 1;
	static const UINT GPS_TIMER_ID = 2;
	static const UINT FILTER_TIMER_ID = 3;

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


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButtonCheckImu();
	afx_msg void OnBnClickedButtonCheckGps();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnBnClickedButtonRunFilter();
};
