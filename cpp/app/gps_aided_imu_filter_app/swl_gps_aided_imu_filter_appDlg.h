
// swl_gps_aided_imu_filter_appDlg.h : header file
//

#pragma once


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
	afx_msg void OnBnClickedButtonUseSensorInput();
	afx_msg void OnBnClickedButtonUseFileInput();
};
