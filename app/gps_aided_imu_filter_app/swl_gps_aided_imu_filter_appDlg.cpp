
// swl_gps_aided_imu_filter_appDlg.cpp : implementation file
//

#include "stdafx.h"
#include "swl_gps_aided_imu_filter_app.h"
#include "swl_gps_aided_imu_filter_appDlg.h"
#include "Adis16350Interface.h"
#include "GpsInterface.h"
#include <cstdlib>
#include <ctime>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// Cswl_gps_aided_imu_filter_appDlg dialog




Cswl_gps_aided_imu_filter_appDlg::Cswl_gps_aided_imu_filter_appDlg(CWnd* pParent /*=NULL*/)
	: CDialog(Cswl_gps_aided_imu_filter_appDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void Cswl_gps_aided_imu_filter_appDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(Cswl_gps_aided_imu_filter_appDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BUTTON_CHECK_IMU, &Cswl_gps_aided_imu_filter_appDlg::OnBnClickedButtonCheckImu)
	ON_BN_CLICKED(IDC_BUTTON_CHECK_GPS, &Cswl_gps_aided_imu_filter_appDlg::OnBnClickedButtonCheckGps)
END_MESSAGE_MAP()


// Cswl_gps_aided_imu_filter_appDlg message handlers

BOOL Cswl_gps_aided_imu_filter_appDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	std::srand((unsigned int)time(NULL));

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void Cswl_gps_aided_imu_filter_appDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void Cswl_gps_aided_imu_filter_appDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR Cswl_gps_aided_imu_filter_appDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void Cswl_gps_aided_imu_filter_appDlg::OnBnClickedButtonCheckImu()
{
	const size_t Ninitial = 1000;

	// initialize ADIS16350
	swl::Adis16350Interface imu;

	// load calibration parameters
	{
		const std::string calibration_param_filename("..\\data\\adis16350_data_20100801\\imu_calibration_result.txt");
		if (!imu.loadCalibrationParam(calibration_param_filename)) {
			AfxMessageBox(_T("fail to load a IMU's calibration parameters"), MB_ICONERROR | MB_OK);
			return;
		}
	}

	// set the initial local gravity & the initial Earth's angular velocity
	swl::ImuData::Accel initialGravity(0.0, 0.0, 0.0);
	swl::ImuData::Gyro initialAngularVel(0.0, 0.0, 0.0);
	if (!imu.setInitialAttitude(Ninitial, initialGravity, initialAngularVel))
	{
		AfxMessageBox(_T("fail to set the initial local gravity & the initial Earth's angular velocity"), MB_ICONERROR | MB_OK);
		return;
	}

	//
	GetDlgItem(IDC_BUTTON_CHECK_IMU)->EnableWindow(FALSE);

	swl::ImuData::Accel measuredAccel(0.0, 0.0, 0.0), calibratedAccel(0.0, 0.0, 0.0);
	swl::ImuData::Gyro measuredAngularVel(0.0, 0.0, 0.0), calibratedAngularVel(0.0, 0.0, 0.0);

	const size_t Nstep = 10000;
	CString msg;
	for (size_t i = 0; i < Nstep; ++i)
	{
		// get measurements of IMU
		imu.readData(measuredAccel, measuredAngularVel);

		imu.calculateCalibratedAcceleration(measuredAccel, calibratedAccel);
		imu.calculateCalibratedAngularRate(measuredAngularVel, calibratedAngularVel);

		msg.Format(_T("%f"), measuredAccel.x);
		GetDlgItem(IDC_EDIT_IMU_ACCEL_X)->SetWindowText(msg);
		msg.Format(_T("%f"), measuredAccel.y);
		GetDlgItem(IDC_EDIT_IMU_ACCEL_Y)->SetWindowText(msg);
		msg.Format(_T("%f"), measuredAccel.z);
		GetDlgItem(IDC_EDIT_IMU_ACCEL_Z)->SetWindowText(msg);
		msg.Format(_T("%f"), measuredAngularVel.x);
		GetDlgItem(IDC_EDIT_IMU_GYRO_X)->SetWindowText(msg);
		msg.Format(_T("%f"), measuredAngularVel.y);
		GetDlgItem(IDC_EDIT_IMU_GYRO_Y)->SetWindowText(msg);
		msg.Format(_T("%f"), measuredAngularVel.z);
		GetDlgItem(IDC_EDIT_IMU_GYRO_Z)->SetWindowText(msg);
	}

	GetDlgItem(IDC_BUTTON_CHECK_IMU)->EnableWindow(TRUE);
}

void Cswl_gps_aided_imu_filter_appDlg::OnBnClickedButtonCheckGps()
{
	const size_t Ninitial = 100;

	// initialize GPS
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring gpsPortName = L"COM4";
#else
	const std::string gpsPortName = "COM4";
#endif
	const unsigned int gpsBaudRate = 9600;

	swl::GpsInterface gps(gpsPortName, gpsBaudRate);
	if (!gps.isConnected())
	{
		AfxMessageBox(_T("fail to connect a GPS"), MB_ICONERROR | MB_OK);
		return;
	}

	// set the initial position and spped of GPS
	swl::EarthData::ECEF initialGpsECEF(0.0, 0.0, 0.0);
	swl::EarthData::Speed initialGpsSpeed(0.0);
	if (!gps.setInitialState(Ninitial, initialGpsECEF, initialGpsSpeed))
	{
		AfxMessageBox(_T("fail to set the initial position & speed of the GPS"), MB_ICONERROR | MB_OK);
		return;
	}

	//
	GetDlgItem(IDC_BUTTON_CHECK_GPS)->EnableWindow(FALSE);

	swl::EarthData::Geodetic measuredGpsGeodetic(0.0, 0.0, 0.0);
	swl::EarthData::ECEF measuredGpsECEF(0.0, 0.0, 0.0);
	swl::EarthData::Speed measuredGpsSpeed(0.0);

	const size_t Nstep = 1000;
	for (size_t i = 0; i < Nstep; ++i)
	{
		// get measurements of GPS
		gps.readData(measuredGpsGeodetic, measuredGpsSpeed);

		swl::EarthData::geodetic_to_ecef(measuredGpsGeodetic, measuredGpsECEF);
	}

	GetDlgItem(IDC_BUTTON_CHECK_IMU)->EnableWindow(TRUE);
}
