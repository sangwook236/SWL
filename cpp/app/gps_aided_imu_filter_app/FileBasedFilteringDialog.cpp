// FileBasedFilteringDialog.cpp : implementation file
//

#include "stdafx.h"
#include "swl_gps_aided_imu_filter_app.h"
#include "FileBasedFilteringDialog.h"
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cassert>

#if defined(max)
#undef max
#endif


// CFileBasedFilteringDialog dialog

IMPLEMENT_DYNAMIC(CFileBasedFilteringDialog, CDialog)

CFileBasedFilteringDialog::CFileBasedFilteringDialog(CWnd* pParent /*=NULL*/)
	: CDialog(CFileBasedFilteringDialog::IDD, pParent),
	  initialGravity_(0.0, 0.0, 0.0), initialAngularVel_(0.0, 0.0, 0.0), initialGpsECEF_(0.0, 0.0, 0.0), initialGpsSpeed_(0.0),
	  prevGpsUtc_(0, 0, 0, 0), prevGpsECEF_(0.0, 0.0, 0.0),
	  Ndata_(0),
	  Q_(NULL), R_(NULL)
{

}

CFileBasedFilteringDialog::~CFileBasedFilteringDialog()
{
}

void CFileBasedFilteringDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CFileBasedFilteringDialog, CDialog)
	ON_WM_TIMER()
	ON_BN_CLICKED(IDC_BUTTON_OPEN_FILE, &CFileBasedFilteringDialog::OnBnClickedButtonOpenFile)
	ON_BN_CLICKED(IDC_BUTTON_START_FILTERING, &CFileBasedFilteringDialog::OnBnClickedButtonStartFiltering)
END_MESSAGE_MAP()


// CFileBasedFilteringDialog message handlers

BOOL CFileBasedFilteringDialog::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  Add extra initialization here
	std::srand((unsigned int)time(NULL));

	prevPerformanceCount_.LowPart = 0;
	prevPerformanceCount_.HighPart = 0;
	prevGpsUtc_.hour = prevGpsUtc_.min = prevGpsUtc_.sec = prevGpsUtc_.msec = 0;

	Ndata_ = 0;

	GetDlgItem(IDC_BUTTON_START_FILTERING)->EnableWindow(FALSE);

	return TRUE;  // return TRUE unless you set the focus to a control
	// EXCEPTION: OCX Property Pages should return FALSE
}

void CFileBasedFilteringDialog::OnTimer(UINT_PTR nIDEvent)
{
	switch (nIDEvent)
	{
	case FILTER_TIMER_ID:  // GPS-aided IMU filter
		runFilter();
		break;
	}

	CDialog::OnTimer(nIDEvent);
}

void CFileBasedFilteringDialog::OnBnClickedButtonOpenFile()
{
	const CString filter(_T("All Files(*.*) |*.*||"));
	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY, filter);

	if (IDOK == dlg.DoModal())
	{
		poses_.clear();
		vels_.clear();
		accels_.clear();

		if (dlg.GetPathName().IsEmpty() || !loadData(dlg.GetPathName()))
		{
			AfxMessageBox(_T("fail to load data"), MB_ICONERROR | MB_OK);
			return;
		}

		Ndata_ = poses_.size();
		assert(vels_.size() == Ndata_);
		assert(accels_.size() == Ndata_);

		GetDlgItem(IDC_EDIT_FILENAME)->SetWindowTextW(dlg.GetFileName());
		GetDlgItem(IDC_BUTTON_START_FILTERING)->EnableWindow(TRUE);
	}
}

void CFileBasedFilteringDialog::OnBnClickedButtonStartFiltering()
{
	//
	static bool toggle = true;

	if (toggle)
	{
#if 0
		// initialize ADIS16350
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("initialize ADIS16350"));
		try
		{
			imu_.reset(new swl::Adis16350Interface());
			if (!imu_)
			{
				AfxMessageBox(_T("fail to create an IMU"), MB_ICONERROR | MB_OK);
				return;
			}
		}
		catch (const std::runtime_error &e)
		{
			AfxMessageBox(CString(_T("fail to create an IMU: ")) + CString(e.what()), MB_ICONERROR | MB_OK);
			return;
		}
		// initialize GPS
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("initialize GPS"));
		try
		{
			gps_.reset(new swl::GpsInterface(gpsPortName_, gpsBaudRate_));
			if (!gps_)
			{
				AfxMessageBox(_T("fail to create a GPS"), MB_ICONERROR | MB_OK);
				return;
			}
		}
		catch (const std::runtime_error &e)
		{
			AfxMessageBox(CString(_T("fail to create an GPS: ")) + CString(e.what()), MB_ICONERROR | MB_OK);
			return;
		}

		// connect GPS
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("connect GPS"));
		if (!gps_->isConnected())
		{
			AfxMessageBox(_T("fail to connect a GPS"), MB_ICONERROR | MB_OK);
			return;
		}

		// load calibration parameters
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("load calibration parameters of ADIS16350"));
		{
			const std::string calibration_param_filename("./data/adis16350_data_20100801/imu_calibration_result.txt");
			if (!imu_->loadCalibrationParam(calibration_param_filename))
			{
				AfxMessageBox(_T("fail to load a IMU's calibration parameters"), MB_ICONERROR | MB_OK);
				return;
			}
		}
#endif

#if 0
		// set the initial local gravity & the initial Earth's angular velocity
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("set the initial local gravity & the initial Earth's angular velocity"));
		initialGravity_.x = initialGravity_.y = initialGravity_.z = 0.0;
		initialAngularVel_.x = initialAngularVel_.y = initialAngularVel_.z = 0.0;
		if (!imu_->setInitialAttitude(Ndata, initialGravity_, initialAngularVel_))
		{
			AfxMessageBox(_T("fail to set the initial local gravity & the initial Earth's angular velocity"), MB_ICONERROR | MB_OK);
			return;
		}

		// FIXME [modify] >>
		initialAngularVel_.x = initialAngularVel_.y = initialAngularVel_.z = 0.0;

		// set the initial position and speed of GPS
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("set the initial position and speed of GPS"));
		if (!gps_->setInitialState(Ndata, initialGpsECEF_, initialGpsSpeed_))
		{
			AfxMessageBox(_T("fail to set the initial position & speed of the GPS"), MB_ICONERROR | MB_OK);
			return;
		}

		// FIXME [modify] >>
		initialGpsSpeed_.val = 0.0;
#elif 0
		// FIXME [modify] >>
		if (!isSensorsInitialized_)
		{
			AfxMessageBox(_T("fail to set the initial parameters of the IMU & the GPS"), MB_ICONERROR | MB_OK);
			return;
		}
#else
		// TODO [check] >>
		initialGravity_.x = initialGravity_.y = initialGravity_.z = 0.0;
		initialAngularVel_.x = initialAngularVel_.y = initialAngularVel_.z = 0.0;
		initialGpsECEF_.x = initialGpsECEF_.y = initialGpsECEF_.z = 0.0;
		initialGpsSpeed_.val = 0.0;
#endif

		// FIXME [check] >>
		freq_.LowPart = 0;
		freq_.HighPart = 0;
		QueryPerformanceFrequency(&freq_);
		prevPerformanceCount_.LowPart = 0;
		prevPerformanceCount_.HighPart = 0;
		QueryPerformanceCounter(&prevPerformanceCount_);
		prevGpsUtc_.hour = prevGpsUtc_.min = prevGpsUtc_.sec = prevGpsUtc_.msec = 0;
		prevGpsECEF_.x = prevGpsECEF_.y = prevGpsECEF_.z = 0.0;

		step_ = 0;

		//
		const double Ts = 0.1;  // [sec]
		const size_t stateDim = 16;
		const size_t inputDim = 3;
		const size_t outputDim = 6;
		const size_t processNoiseDim = stateDim;
		const size_t observationNoiseDim = outputDim;

		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("initialize a runner of GPS-aided IMU filter"));
		runner_.reset(new swl::GpsAidedImuFilterRunner(Ts, initialGravity_, initialAngularVel_));

		// set x0 & P0
		gsl_vector *x0 = gsl_vector_alloc(stateDim);
		gsl_vector_set_zero(x0);
		gsl_vector_set(x0, 6, 1.0);  // e0 = 1.0
		gsl_matrix *P0 = gsl_matrix_alloc(stateDim, stateDim);
		gsl_matrix_set_identity(P0);
		gsl_matrix_scale(P0, 1.0e-10);  // the initial estimate is completely unknown

		runner_->initialize(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim, x0, P0);

		gsl_vector_free(x0);  x0 = NULL;
		gsl_matrix_free(P0);  P0 = NULL;

		// set Q & R
		Q_ = gsl_matrix_alloc(processNoiseDim, processNoiseDim);
		gsl_matrix_set_identity(Q_);
		R_ = gsl_matrix_alloc(observationNoiseDim, observationNoiseDim);
		gsl_matrix_set_identity(R_);

		{
			// FIXME [modify] >>
			const double Qc = 0.01;
			gsl_matrix_set(Q_, 0, 0, Qc);
			gsl_matrix_set(Q_, 1, 1, Qc);
			gsl_matrix_set(Q_, 2, 2, Qc);
			gsl_matrix_set(Q_, 3, 3, Qc);
			gsl_matrix_set(Q_, 4, 4, Qc);
			gsl_matrix_set(Q_, 5, 5, Qc);
			gsl_matrix_set(Q_, 6, 6, Qc);
			gsl_matrix_set(Q_, 7, 7, Qc);
			gsl_matrix_set(Q_, 8, 8, Qc);
			gsl_matrix_set(Q_, 9, 9, Qc);
			gsl_matrix_set(Q_, 10, 10, Qc);
			gsl_matrix_set(Q_, 11, 11, Qc);
			gsl_matrix_set(Q_, 12, 12, Qc);
			gsl_matrix_set(Q_, 13, 13, Qc);
			gsl_matrix_set(Q_, 14, 14, Qc);
			gsl_matrix_set(Q_, 15, 15, Qc);

			// FIXME [modify] >>
			const double Rc = 0.01;
			gsl_matrix_set(R_, 0, 0, Rc);
			gsl_matrix_set(R_, 1, 1, Rc);
			gsl_matrix_set(R_, 2, 2, Rc);
			gsl_matrix_set(R_, 3, 3, Rc);
			gsl_matrix_set(R_, 4, 4, Rc);
			gsl_matrix_set(R_, 5, 5, Rc);
		}

		//
		GetDlgItem(IDC_BUTTON_START_FILTERING)->SetWindowText(_T("Stop GPS-aided IMU Filter"));
		SetTimer(FILTER_TIMER_ID, FILTER_LOOPING_INTERVAL, NULL);

		toggle = false;
	}
	else
	{
		GetDlgItem(IDC_BUTTON_START_FILTERING)->SetWindowText(_T("Start GPS-aided IMU Filter"));
		KillTimer(FILTER_TIMER_ID);

		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("terminate a runner of GPS-aided IMU filter"));
		runner_->finalize();
		runner_.reset();

		gsl_matrix_free(Q_);  Q_ = NULL;
		gsl_matrix_free(R_);  R_ = NULL;

		toggle = true;
	}
}

bool CFileBasedFilteringDialog::loadData(const CString &filename)
{
#if defined(UNIODE) || defined(_UNICODE)
	std::wifstream stream((wchar_t *)(LPCTSTR)filename);
#else
	std::ifstream stream((char *)(LPCTSTR)filename);
#endif

	if (!stream) return false;

	double pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, accel_x, accel_y, accel_z;
	while (stream && !stream.eof())
	{
		stream >> pos_x >> pos_y >> pos_z >> vel_x >> vel_y >> vel_z >> accel_x >> accel_y >> accel_z;

		poses_.push_back(swl::EarthData::ECEF(pos_x, pos_y, pos_z));
		vels_.push_back(swl::EarthData::ECEF(vel_x, vel_y, vel_z));
		accels_.push_back(swl::EarthData::ECEF(accel_x, accel_y, accel_z));
	}

	stream.close();

	return true;
}

void CFileBasedFilteringDialog::runFilter()
{
	swl::ImuData::Accel measuredAccel(0.0, 0.0, 0.0), calibratedAccel(0.0, 0.0, 0.0);
	swl::ImuData::Gyro measuredAngularVel(0.0, 0.0, 0.0), calibratedAngularVel(0.0, 0.0, 0.0);
	LARGE_INTEGER performanceCount;
	swl::EarthData::Geodetic measuredGpsGeodetic(0.0, 0.0, 0.0);
	swl::EarthData::ECEF measuredGpsECEF(0.0, 0.0, 0.0);
	swl::EarthData::ECEF measuredGpsVel(0.0, 0.0, 0.0);
	swl::EarthData::Speed measuredGpsSpeed(0.0);
	swl::EarthData::Time gpsUtc(0, 0, 0, 0);

#if 0
	// get measurements of IMU & GPS
	if (!imu_->readData(measuredAccel, measuredAngularVel, performanceCount) ||
		!gps_->readData(measuredGpsGeodetic, measuredGpsSpeed, gpsUtc))
		return;

	//
	imu_->calculateCalibratedAcceleration(measuredAccel, calibratedAccel);
	imu_->calculateCalibratedAngularRate(measuredAngularVel, calibratedAngularVel);

	swl::EarthData::geodetic_to_ecef(measuredGpsGeodetic, measuredGpsECEF);

	measuredGpsECEF.x -= initialGpsECEF_.x;
	measuredGpsECEF.y -= initialGpsECEF_.y;
	measuredGpsECEF.z -= initialGpsECEF_.z;

	//
	const __int64 imuElapsedTime = (0 == performanceCount.HighPart && 0 == performanceCount.LowPart) ? 0 : ((performanceCount.QuadPart - prevPerformanceCount_.QuadPart) * 1000 / freq_.QuadPart);
	const __int64 gpsElapsedTime = ((gpsUtc.min - prevGpsUtc_.min) * 60 + (gpsUtc.sec - prevGpsUtc_.sec)) * 1000 + (gpsUtc.msec - prevGpsUtc_.msec);

	measuredGpsVel.x = (measuredGpsECEF.x - prevGpsECEF_.x) / gpsElapsedTime * 1000;
	measuredGpsVel.y = (measuredGpsECEF.y - prevGpsECEF_.y) / gpsElapsedTime * 1000;
	measuredGpsVel.z = (measuredGpsECEF.z - prevGpsECEF_.z) / gpsElapsedTime * 1000;
#else
	performanceCount.LowPart = 0;
	performanceCount.HighPart = 0;
	QueryPerformanceCounter(&performanceCount);

	measuredAccel.x = accels_[step_].x;  measuredAccel.y = accels_[step_].y;  measuredAccel.z = accels_[step_].z;
	// FIXME [modify] >> now assume no rotational motion
	measuredAngularVel.x = measuredAngularVel.y = measuredAngularVel.z = 0.0;
	measuredGpsECEF = poses_[step_];
	measuredGpsVel = vels_[step_];
	measuredGpsSpeed.val = 0.0;

	calibratedAccel = measuredAccel;
	calibratedAngularVel = measuredAngularVel;

	const __int64 imuElapsedTime = (0 == performanceCount.HighPart && 0 == performanceCount.LowPart) ? 0 : ((performanceCount.QuadPart - prevPerformanceCount_.QuadPart) * 1000 / freq_.QuadPart);
	const __int64 gpsElapsedTime = imuElapsedTime;
#endif

	//
	if (!runner_->runStep(Q_, R_, calibratedAccel, calibratedAngularVel, measuredGpsECEF, measuredGpsVel, measuredGpsSpeed))
		throw std::runtime_error("GPS-aided IMU filter error !!!");

	//
	prevPerformanceCount_ = performanceCount;
	prevGpsUtc_ = gpsUtc;
	prevGpsECEF_ = measuredGpsECEF;

	//
	const gsl_vector *pos = runner_->getFilteredPos();
	const gsl_vector *vel = runner_->getFilteredVel();
	//const gsl_vector *accel = runner_->getFilteredAccel();
	const gsl_vector *quat = runner_->getFilteredQuaternion();
	//const gsl_vector *angVel = runner_->getFilteredAngularVel();

	const double dist = std::sqrt(gsl_vector_get(pos, 0)*gsl_vector_get(pos, 0) + gsl_vector_get(pos, 1)*gsl_vector_get(pos, 1) + gsl_vector_get(pos, 2)*gsl_vector_get(pos, 2));

	++step_;

	CString msg;
	msg.Format(_T("%f"), gsl_vector_get(pos, 0));
	GetDlgItem(IDC_EDIT_STATE_POS_X)->SetWindowText(msg);
	msg.Format(_T("%f"), gsl_vector_get(pos, 1));
	GetDlgItem(IDC_EDIT_STATE_POS_Y)->SetWindowText(msg);
	msg.Format(_T("%f"), gsl_vector_get(pos, 2));
	GetDlgItem(IDC_EDIT_STATE_POS_Z)->SetWindowText(msg);
	msg.Format(_T("%f"), gsl_vector_get(vel, 0));
	GetDlgItem(IDC_EDIT_STATE_VEL_X)->SetWindowText(msg);
	msg.Format(_T("%f"), gsl_vector_get(vel, 1));
	GetDlgItem(IDC_EDIT_STATE_VEL_Y)->SetWindowText(msg);
	msg.Format(_T("%f"), gsl_vector_get(vel, 2));
	GetDlgItem(IDC_EDIT_STATE_VEL_Z)->SetWindowText(msg);
	msg.Format(_T("%f"), gsl_vector_get(quat, 0));
	GetDlgItem(IDC_EDIT_STATE_E0)->SetWindowText(msg);
	msg.Format(_T("%f"), gsl_vector_get(quat, 1));
	GetDlgItem(IDC_EDIT_STATE_E1)->SetWindowText(msg);
	msg.Format(_T("%f"), gsl_vector_get(quat, 2));
	GetDlgItem(IDC_EDIT_STATE_E2)->SetWindowText(msg);
	msg.Format(_T("%f"), gsl_vector_get(quat, 3));
	GetDlgItem(IDC_EDIT_STATE_E3)->SetWindowText(msg);
	msg.Format(_T("%f"), dist);
	GetDlgItem(IDC_EDIT_FILTER_DISTANCE)->SetWindowText(msg);
	msg.Format(_T("%d"), std::max(imuElapsedTime, gpsElapsedTime));
	GetDlgItem(IDC_EDIT_FILTER_ELAPSED_TIME)->SetWindowText(msg);

	msg.Format(_T("%d"), step_);
	GetDlgItem(IDC_EDIT_STEP)->SetWindowText(msg);

	if (step_ >= Ndata_)
		OnBnClickedButtonStartFiltering();
}
