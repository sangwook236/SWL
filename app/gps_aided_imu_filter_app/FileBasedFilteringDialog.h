#pragma once

#include "Adis16350Interface.h"
#include "GpsInterface.h"
#include "GpsAidedImuFilterRunner.h"
#include <boost/smart_ptr.hpp>
#include <string>
#include <list>


// CFileBasedFilteringDialog dialog

class CFileBasedFilteringDialog : public CDialog
{
	DECLARE_DYNAMIC(CFileBasedFilteringDialog)

public:
	CFileBasedFilteringDialog(CWnd* pParent = NULL);   // standard constructor
	virtual ~CFileBasedFilteringDialog();

// Dialog Data
	enum { IDD = IDD_FILE_BASED_FILTERING_DIALOG };

private:
	bool loadData(const CString &filename);
	void runFilter();

private:
	static const UINT FILTER_TIMER_ID = 1;
	static const size_t FILTER_LOOPING_INTERVAL = 1;

	boost::scoped_ptr<swl::GpsAidedImuFilterRunner> runner_;

	swl::ImuData::Accel initialGravity_;
	swl::ImuData::Gyro initialAngularVel_;
	swl::EarthData::ECEF initialGpsECEF_;
	swl::EarthData::Speed initialGpsSpeed_;

	LARGE_INTEGER freq_;
	LARGE_INTEGER prevPerformanceCount_;
	swl::EarthData::Time prevGpsUtc_;
	swl::EarthData::ECEF prevGpsECEF_;

	size_t Ndata_;
	size_t step_;

	std::list<swl::EarthData::ECEF> poses_, vels_, accels_;

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
public:
	virtual BOOL OnInitDialog();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnBnClickedButtonOpenFile();
	afx_msg void OnBnClickedButtonStartFiltering();
};
