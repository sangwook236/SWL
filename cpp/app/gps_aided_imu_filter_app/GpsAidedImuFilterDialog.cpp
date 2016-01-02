// GpsAidedImuFilterDialog.cpp : implementation file
//

#include "stdafx.h"
#include "swl_gps_aided_imu_filter_app.h"
#include "GpsAidedImuFilterDialog.h"
#include <fstream>
#include <cstdlib>
#include <ctime>

#if defined(max)
#undef max
#endif


// CGpsAidedImuFilterDialog dialog

IMPLEMENT_DYNAMIC(CGpsAidedImuFilterDialog, CDialog)

CGpsAidedImuFilterDialog::CGpsAidedImuFilterDialog(CWnd* pParent /*=NULL*/)
	: CDialog(CGpsAidedImuFilterDialog::IDD, pParent),
#if defined(_UNICODE) || defined(UNICODE)
	  gpsPortName_(L"COM8"),
#else
	  gpsPortName_("COM8"),
#endif
	  gpsBaudRate_(9600),
	  initialGravity_(0.0, 0.0, 0.0), initialAngularVel_(0.0, 0.0, 0.0), initialGpsECEF_(0.0, 0.0, 0.0), initialGpsSpeed_(0.0),
	  prevGpsUtc_(0, 0, 0, 0), prevGpsECEF_(0.0, 0.0, 0.0),
	  Q_(NULL), R_(NULL)
{

}

CGpsAidedImuFilterDialog::~CGpsAidedImuFilterDialog()
{
}

void CGpsAidedImuFilterDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CGpsAidedImuFilterDialog, CDialog)
	ON_WM_TIMER()
	ON_BN_CLICKED(IDC_BUTTON_INITIALIZE_IMU_AND_GPS, &CGpsAidedImuFilterDialog::OnBnClickedButtonInitializeImuAndGps)
	ON_BN_CLICKED(IDC_BUTTON_SAVE_RAW_DATA, &CGpsAidedImuFilterDialog::OnBnClickedButtonSaveRawData)
	ON_BN_CLICKED(IDC_BUTTON_CHECK_IMU, &CGpsAidedImuFilterDialog::OnBnClickedButtonCheckImu)
	ON_BN_CLICKED(IDC_BUTTON_CHECK_GPS, &CGpsAidedImuFilterDialog::OnBnClickedButtonCheckGps)
	ON_BN_CLICKED(IDC_BUTTON_RUN_FILTER, &CGpsAidedImuFilterDialog::OnBnClickedButtonRunFilter)
END_MESSAGE_MAP()


// CGpsAidedImuFilterDialog message handlers

BOOL CGpsAidedImuFilterDialog::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  Add extra initialization here
	std::srand((unsigned int)time(NULL));

	isSensorsInitialized_ = false;

	prevPerformanceCount_.LowPart = 0;
	prevPerformanceCount_.HighPart = 0;
	prevGpsUtc_.hour = prevGpsUtc_.min = prevGpsUtc_.sec = prevGpsUtc_.msec = 0;

	GetDlgItem(IDC_BUTTON_CHECK_IMU)->SetWindowText(_T("Start IMU"));
	GetDlgItem(IDC_BUTTON_CHECK_GPS)->SetWindowText(_T("Start GPS"));
	GetDlgItem(IDC_BUTTON_RUN_FILTER)->SetWindowText(_T("Start GPS-aided IMU Filter"));
	GetDlgItem(IDC_BUTTON_SAVE_RAW_DATA)->SetWindowText(_T("Start Saving Raw Data"));
	GetDlgItem(IDC_BUTTON_CHECK_IMU)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_CHECK_GPS)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_RUN_FILTER)->EnableWindow(FALSE);

	return TRUE;  // return TRUE unless you set the focus to a control
	// EXCEPTION: OCX Property Pages should return FALSE
}

void CGpsAidedImuFilterDialog::OnTimer(UINT_PTR nIDEvent)
{
	switch (nIDEvent)
	{
	case IMU_TIMER_ID:  // IMU
		checkImu();
		break;
	case GPS_TIMER_ID:  // GPS
		checkGps();
		break;
	case FILTER_TIMER_ID:  // GPS-aided IMU filter
		runFilter();
		break;
	case SAVER_TIMER_ID:
		saveRawData();
		break;
	}

	CDialog::OnTimer(nIDEvent);
}

void CGpsAidedImuFilterDialog::OnBnClickedButtonInitializeImuAndGps()
{
	isSensorsInitialized_ = false;
	GetDlgItem(IDC_BUTTON_CHECK_IMU)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_CHECK_GPS)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_RUN_FILTER)->EnableWindow(FALSE);

	CString msg;

	if (initializeSensors())
	{
		msg.Format(_T("%f"), initialGravity_.x);
		GetDlgItem(IDC_EDIT_IMU_ACCEL_X)->SetWindowText(msg);
		msg.Format(_T("%f"), initialGravity_.y);
		GetDlgItem(IDC_EDIT_IMU_ACCEL_Y)->SetWindowText(msg);
		msg.Format(_T("%f"), initialGravity_.z);
		GetDlgItem(IDC_EDIT_IMU_ACCEL_Z)->SetWindowText(msg);
		msg.Format(_T("%f"), initialAngularVel_.x);
		GetDlgItem(IDC_EDIT_IMU_GYRO_X)->SetWindowText(msg);
		msg.Format(_T("%f"), initialAngularVel_.y);
		GetDlgItem(IDC_EDIT_IMU_GYRO_Y)->SetWindowText(msg);
		msg.Format(_T("%f"), initialAngularVel_.z);
		GetDlgItem(IDC_EDIT_IMU_GYRO_Z)->SetWindowText(msg);
		GetDlgItem(IDC_EDIT_IMU_ELAPSED_TIME)->SetWindowText(_T(""));

		GetDlgItem(IDC_EDIT_GPS_LAT)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_LON)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_ALT)->SetWindowText(_T(""));
		msg.Format(_T("%f"), initialGpsECEF_.x);
		GetDlgItem(IDC_EDIT_GPS_X)->SetWindowText(msg);
		msg.Format(_T("%f"), initialGpsECEF_.y);
		GetDlgItem(IDC_EDIT_GPS_Y)->SetWindowText(msg);
		msg.Format(_T("%f"), initialGpsECEF_.z);
		GetDlgItem(IDC_EDIT_GPS_Z)->SetWindowText(msg);
		GetDlgItem(IDC_EDIT_GPS_VEL_X)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_VEL_Y)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_VEL_Z)->SetWindowText(_T(""));
		msg.Format(_T("%f"), initialGpsSpeed_.val);
		GetDlgItem(IDC_EDIT_GPS_SPEED)->SetWindowText(msg);
		GetDlgItem(IDC_EDIT_GPS_ELAPSED_TIME)->SetWindowText(_T(""));

		//
		isSensorsInitialized_ = true;

		GetDlgItem(IDC_BUTTON_CHECK_IMU)->EnableWindow(TRUE);
		GetDlgItem(IDC_BUTTON_CHECK_GPS)->EnableWindow(TRUE);
		GetDlgItem(IDC_BUTTON_RUN_FILTER)->EnableWindow(TRUE);
	}
	else
	{
		GetDlgItem(IDC_EDIT_IMU_ACCEL_X)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_IMU_ACCEL_Y)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_IMU_ACCEL_Z)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_IMU_GYRO_X)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_IMU_GYRO_Y)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_IMU_GYRO_Z)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_IMU_ELAPSED_TIME)->SetWindowText(_T(""));

		GetDlgItem(IDC_EDIT_GPS_LAT)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_LON)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_ALT)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_X)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_Y)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_Z)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_VEL_X)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_VEL_Y)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_VEL_Z)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_SPEED)->SetWindowText(_T(""));
		GetDlgItem(IDC_EDIT_GPS_ELAPSED_TIME)->SetWindowText(_T(""));

		AfxMessageBox(_T("fail to initialize an IMU & a GPS"), MB_ICONERROR | MB_OK);
	}
}

void CGpsAidedImuFilterDialog::OnBnClickedButtonSaveRawData()
{
	static bool toggle = true;

	if (toggle)
	{
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

		freq_.LowPart = 0;
		freq_.HighPart = 0;
		QueryPerformanceFrequency(&freq_);
		prevPerformanceCount_.LowPart = 0;
		prevPerformanceCount_.HighPart = 0;
		QueryPerformanceCounter(&prevPerformanceCount_);
		prevGpsUtc_.hour = prevGpsUtc_.min = prevGpsUtc_.sec = prevGpsUtc_.msec = 0;
		prevGpsECEF_.x = prevGpsECEF_.y = prevGpsECEF_.z = 0.0;

		imuTimeStamps_.clear();
		imuAccels_.clear();
		imuGyros_.clear();
		gpsTimeStamps_.clear();
		gpsGeodetics_.clear();
		gpsSpeeds_.clear();

		step_ = 0;

		//
		GetDlgItem(IDC_BUTTON_SAVE_RAW_DATA)->SetWindowText(_T("Stop Saving Raw Data"));
		SetTimer(SAVER_TIMER_ID, SAVER_SAMPLING_INTERVAL, NULL);

		toggle = false;
	}
	else
	{
		GetDlgItem(IDC_BUTTON_SAVE_RAW_DATA)->SetWindowText(_T("Start Saving Raw Data"));
		KillTimer(SAVER_TIMER_ID);

		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("terminate ADIS16350 & GPS"));
		imu_.reset();
		gps_.reset();

		//
		const std::string raw_data_filename("./data/adis16350_data_20100930/mesaured_raw_data.txt");
		std::ofstream stream(raw_data_filename.c_str());
		if (stream)
		{
			if (imuTimeStamps_.size() != step_ || imuAccels_.size() != step_ || imuGyros_.size() != step_ ||
				gpsTimeStamps_.size() != step_ || gpsGeodetics_.size() != step_ || gpsSpeeds_.size() != step_)
			{
				AfxMessageBox(_T("the sizes of measured raw datasets are not matched"), MB_ICONERROR | MB_OK);
				return;
			}

			std::list<__int64>::iterator itImuTimeStamp = imuTimeStamps_.begin();
			std::list<swl::ImuData::Accel>::iterator itImuAccel = imuAccels_.begin();
			std::list<swl::ImuData::Gyro>::iterator itImuGyro = imuGyros_.begin();
			std::list<long>::iterator itGpsTimeStamp = gpsTimeStamps_.begin();
			std::list<swl::EarthData::Geodetic>::iterator itGpsGeodetic = gpsGeodetics_.begin();
			std::list<swl::EarthData::Speed>::iterator itGpsSpeed = gpsSpeeds_.begin();

			for (size_t i = 0; i < step_; ++i)
			{
				stream << *itImuTimeStamp << '\t' << itImuAccel->x << '\t' << itImuAccel->y << '\t' << itImuAccel->z << '\t' << itImuGyro->x << '\t' << itImuGyro->y << '\t' << itImuGyro->z << '\t'
					<< *itGpsTimeStamp << '\t' << itGpsGeodetic->lat << '\t' << itGpsGeodetic->lon << '\t' << itGpsGeodetic->alt << '\t' << itGpsSpeed->val << std::endl;

				++itImuTimeStamp;
				++itImuAccel;
				++itImuGyro;
				++itGpsTimeStamp;
				++itGpsGeodetic;
				++itGpsSpeed;
			}

			stream.flush();
			stream.close();

			imuTimeStamps_.clear();
			imuAccels_.clear();
			imuGyros_.clear();
			gpsTimeStamps_.clear();
			gpsGeodetics_.clear();
			gpsSpeeds_.clear();
		}

		toggle = true;
	}
}

void CGpsAidedImuFilterDialog::OnBnClickedButtonCheckImu()
{
	static bool toggle = true;

	if (toggle)
	{
		const size_t Ninitial = 1000;

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

#if 0
		// set the initial local gravity & the initial Earth's angular velocity
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("set the initial local gravity & the initial Earth's angular velocity"));
		initialGravity_.x = initialGravity_.y = initialGravity_.z = 0.0;
		initialAngularVel_.x = initialAngularVel_.y = initialAngularVel_.z = 0.0;
		if (!imu_->setInitialAttitude(Ninitial, initialGravity_, initialAngularVel_))
		{
			AfxMessageBox(_T("fail to set the initial local gravity & the initial Earth's angular velocity"), MB_ICONERROR | MB_OK);
			return;
		}
#else
		if (!isSensorsInitialized_)
		{
			AfxMessageBox(_T("fail to set the initial local gravity & the initial Earth's angular velocity"), MB_ICONERROR | MB_OK);
			return;
		}
#endif

		freq_.LowPart = 0;
		freq_.HighPart = 0;
		QueryPerformanceFrequency(&freq_);
		prevPerformanceCount_.LowPart = 0;
		prevPerformanceCount_.HighPart = 0;
		QueryPerformanceCounter(&prevPerformanceCount_);

		step_ = 0;

		//
		GetDlgItem(IDC_BUTTON_CHECK_IMU)->SetWindowText(_T("Stop IMU"));
		SetTimer(IMU_TIMER_ID, IMU_SAMPLING_INTERVAL, NULL);

		toggle = false;
	}
	else
	{
		GetDlgItem(IDC_BUTTON_CHECK_IMU)->SetWindowText(_T("Start IMU"));
		KillTimer(IMU_TIMER_ID);

		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("terminate ADIS16350"));
		imu_.reset();

		toggle = true;
	}
}

void CGpsAidedImuFilterDialog::OnBnClickedButtonCheckGps()
{
	static bool toggle = true;

	if (toggle)
	{
		const size_t Ninitial = 100;

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

#if 0
		// set the initial position and speed of GPS
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("set the initial position and speed of GPS"));
		if (!gps_->setInitialState(Ninitial, initialGpsECEF_, initialGpsSpeed_))
		{
			AfxMessageBox(_T("fail to set the initial position & speed of the GPS"), MB_ICONERROR | MB_OK);
			return;
		}
#else
		if (!isSensorsInitialized_)
		{
			AfxMessageBox(_T("fail to set the initial position & speed of the GPS"), MB_ICONERROR | MB_OK);
			return;
		}
#endif

		// FIXME [check] >>
		prevGpsUtc_.hour = prevGpsUtc_.min = prevGpsUtc_.sec = prevGpsUtc_.msec = 0;
		prevGpsECEF_.x = prevGpsECEF_.y = prevGpsECEF_.z = 0.0;

		step_ = 0;

		//
		GetDlgItem(IDC_BUTTON_CHECK_GPS)->SetWindowText(_T("Stop GPS"));
		SetTimer(GPS_TIMER_ID, GPS_SAMPLING_INTERVAL, NULL);

		toggle = false;
	}
	else
	{
		GetDlgItem(IDC_BUTTON_CHECK_GPS)->SetWindowText(_T("Start GPS"));
		KillTimer(GPS_TIMER_ID);

		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("terminate GPS"));
		gps_.reset();

		toggle = true;
	}
}

void CGpsAidedImuFilterDialog::OnBnClickedButtonRunFilter()
{
	static bool toggle = true;

	if (toggle)
	{
		const size_t Nimu = 10000;
		const size_t Ngps = 100;

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

#if 0
		// set the initial local gravity & the initial Earth's angular velocity
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("set the initial local gravity & the initial Earth's angular velocity"));
		initialGravity_.x = initialGravity_.y = initialGravity_.z = 0.0;
		initialAngularVel_.x = initialAngularVel_.y = initialAngularVel_.z = 0.0;
		if (!imu_->setInitialAttitude(Nimu, initialGravity_, initialAngularVel_))
		{
			AfxMessageBox(_T("fail to set the initial local gravity & the initial Earth's angular velocity"), MB_ICONERROR | MB_OK);
			return;
		}

		// FIXME [modify] >>
		initialAngularVel_.x = initialAngularVel_.y = initialAngularVel_.z = 0.0;

		// set the initial position and speed of GPS
		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("set the initial position and speed of GPS"));
		if (!gps_->setInitialState(Ngps, initialGpsECEF_, initialGpsSpeed_))
		{
			AfxMessageBox(_T("fail to set the initial position & speed of the GPS"), MB_ICONERROR | MB_OK);
			return;
		}

		// FIXME [modify] >>
		initialGpsSpeed_.val = 0.0;
#else
		if (!isSensorsInitialized_)
		{
			AfxMessageBox(_T("fail to set the initial parameters of the IMU & the GPS"), MB_ICONERROR | MB_OK);
			return;
		}
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
		const double Ts = 0.01;  // [sec]
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
		gsl_matrix_scale(P0, 1.0e-8);  // the initial estimate is completely unknown

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
			const double Qc = 1.0e-10;
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
			const double Rc = 1.0e-10;
			gsl_matrix_set(R_, 0, 0, Rc);
			gsl_matrix_set(R_, 1, 1, Rc);
			gsl_matrix_set(R_, 2, 2, Rc);
			gsl_matrix_set(R_, 3, 3, Rc);
			gsl_matrix_set(R_, 4, 4, Rc);
			gsl_matrix_set(R_, 5, 5, Rc);
		}

		//
		GetDlgItem(IDC_BUTTON_RUN_FILTER)->SetWindowText(_T("Stop GPS-aided IMU Filter"));
		SetTimer(FILTER_TIMER_ID, FILTER_SAMPLING_INTERVAL, NULL);

		toggle = false;
	}
	else
	{
		GetDlgItem(IDC_BUTTON_RUN_FILTER)->SetWindowText(_T("Start GPS-aided IMU Filter"));
		KillTimer(FILTER_TIMER_ID);

		GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("terminate a runner of GPS-aided IMU filter & ADIS16350, GPS"));
		runner_->finalize();
		runner_.reset();
		imu_.reset();
		gps_.reset();

		gsl_matrix_free(Q_);  Q_ = NULL;
		gsl_matrix_free(R_);  R_ = NULL;

		toggle = true;
	}
}

bool CGpsAidedImuFilterDialog::initializeSensors()
{
	const size_t Nimu = 10000;
	const size_t Ngps = 100;

	// initialize ADIS16350
	GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("initialize ADIS16350"));
	try
	{
		imu_.reset(new swl::Adis16350Interface());
		if (!imu_)
		{
			AfxMessageBox(_T("fail to create an IMU"), MB_ICONERROR | MB_OK);
			return false;
		}
	}
	catch (const std::runtime_error &e)
	{
		AfxMessageBox(CString(_T("fail to create an IMU: ")) + CString(e.what()), MB_ICONERROR | MB_OK);
		return false;
	}
	// initialize GPS
	GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("initialize GPS"));
	try
	{
		gps_.reset(new swl::GpsInterface(gpsPortName_, gpsBaudRate_));
		if (!gps_)
		{
			AfxMessageBox(_T("fail to create a GPS"), MB_ICONERROR | MB_OK);
			return false;
		}
	}
	catch (const std::runtime_error &e)
	{
		AfxMessageBox(CString(_T("fail to create an GPS: ")) + CString(e.what()), MB_ICONERROR | MB_OK);
		return false;
	}

	// connect GPS
	GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("connect GPS"));
	if (!gps_->isConnected())
	{
		AfxMessageBox(_T("fail to connect a GPS"), MB_ICONERROR | MB_OK);
		return false;
	}

	// load calibration parameters
	GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("load calibration parameters of ADIS16350"));
	{
		const std::string calibration_param_filename("./data/adis16350_data_20100801/imu_calibration_result.txt");
		if (!imu_->loadCalibrationParam(calibration_param_filename))
		{
			AfxMessageBox(_T("fail to load a IMU's calibration parameters"), MB_ICONERROR | MB_OK);
			return false;
		}
	}

	// set the initial local gravity & the initial Earth's angular velocity
	GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("set the initial local gravity & the initial Earth's angular velocity"));
	initialGravity_.x = initialGravity_.y = initialGravity_.z = 0.0;
	initialAngularVel_.x = initialAngularVel_.y = initialAngularVel_.z = 0.0;
	if (!imu_->setInitialAttitude(Nimu, initialGravity_, initialAngularVel_))
	{
		AfxMessageBox(_T("fail to set the initial local gravity & the initial Earth's angular velocity"), MB_ICONERROR | MB_OK);
		return false;
	}

	// FIXME [modify] >>
	initialAngularVel_.x = initialAngularVel_.y = initialAngularVel_.z = 0.0;

	// set the initial position and speed of GPS
	GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("set the initial position and speed of GPS"));
	if (!gps_->setInitialState(Ngps, initialGpsECEF_, initialGpsSpeed_))
	{
		AfxMessageBox(_T("fail to set the initial position & speed of the GPS"), MB_ICONERROR | MB_OK);
		return false;
	}

	// FIXME [modify] >>
	initialGpsSpeed_.val = 0.0;

	//
	GetDlgItem(IDC_EDIT_MESSAGE)->SetWindowText(_T("terminate ADIS16350 & GPS"));
	imu_.reset();
	gps_.reset();

	return true;
}

void CGpsAidedImuFilterDialog::saveRawData()
{
	swl::ImuData::Accel measuredAccel(0.0, 0.0, 0.0);
	swl::ImuData::Gyro measuredAngularVel(0.0, 0.0, 0.0);
	LARGE_INTEGER performanceCount;
	swl::EarthData::Geodetic measuredGpsGeodetic(0.0, 0.0, 0.0);
	swl::EarthData::Speed measuredGpsSpeed(0.0);
	swl::EarthData::Time gpsUtc(0, 0, 0, 0);

	// get measurements of IMU & GPS
	if (!imu_->readData(measuredAccel, measuredAngularVel, performanceCount) ||
		!gps_->readData(measuredGpsGeodetic, measuredGpsSpeed, gpsUtc))
		return;

	const __int64 imuTimeStamp = (0 == performanceCount.HighPart && 0 == performanceCount.LowPart) ? 0 : ((performanceCount.QuadPart) * 1000 / freq_.QuadPart);
	const long gpsTimeStamp = (gpsUtc.min * 60 + gpsUtc.sec) * 1000 + gpsUtc.msec;

	imuTimeStamps_.push_back(imuTimeStamp);
	imuAccels_.push_back(measuredAccel);
	imuGyros_.push_back(measuredAngularVel);
	gpsTimeStamps_.push_back(gpsTimeStamp);
	gpsGeodetics_.push_back(measuredGpsGeodetic);
	gpsSpeeds_.push_back(measuredGpsSpeed);

	//
	const __int64 imuElapsedTime = (0 == performanceCount.HighPart && 0 == performanceCount.LowPart) ? 0 : ((performanceCount.QuadPart - prevPerformanceCount_.QuadPart) * 1000 / freq_.QuadPart);
	const __int64 gpsElapsedTime = ((gpsUtc.min - prevGpsUtc_.min) * 60 + (gpsUtc.sec - prevGpsUtc_.sec)) * 1000 + (gpsUtc.msec - prevGpsUtc_.msec);

	//
	prevPerformanceCount_ = performanceCount;
	prevGpsUtc_ = gpsUtc;

	++step_;

	//
	CString msg;
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
	msg.Format(_T("%d"), imuElapsedTime);
	GetDlgItem(IDC_EDIT_IMU_ELAPSED_TIME)->SetWindowText(msg);

	msg.Format(_T("%f"), measuredGpsGeodetic.lat);
	GetDlgItem(IDC_EDIT_GPS_LAT)->SetWindowText(msg);
	msg.Format(_T("%f"), measuredGpsGeodetic.lon);
	GetDlgItem(IDC_EDIT_GPS_LON)->SetWindowText(msg);
	msg.Format(_T("%f"), measuredGpsGeodetic.alt);
	GetDlgItem(IDC_EDIT_GPS_ALT)->SetWindowText(msg);
	GetDlgItem(IDC_EDIT_GPS_X)->SetWindowText(_T(""));
	GetDlgItem(IDC_EDIT_GPS_Y)->SetWindowText(_T(""));
	GetDlgItem(IDC_EDIT_GPS_Z)->SetWindowText(_T(""));
	GetDlgItem(IDC_EDIT_GPS_VEL_X)->SetWindowText(_T(""));
	GetDlgItem(IDC_EDIT_GPS_VEL_Y)->SetWindowText(_T(""));
	GetDlgItem(IDC_EDIT_GPS_VEL_Z)->SetWindowText(_T(""));
	msg.Format(_T("%f"), measuredGpsSpeed.val);
	GetDlgItem(IDC_EDIT_GPS_SPEED)->SetWindowText(msg);
	msg.Format(_T("%d"), gpsElapsedTime);
	GetDlgItem(IDC_EDIT_GPS_ELAPSED_TIME)->SetWindowText(msg);

	msg.Format(_T("%d"), step_);
	GetDlgItem(IDC_EDIT_STEP)->SetWindowText(msg);
}

void CGpsAidedImuFilterDialog::checkImu()
{
	swl::ImuData::Accel measuredAccel(0.0, 0.0, 0.0), calibratedAccel(0.0, 0.0, 0.0);
	swl::ImuData::Gyro measuredAngularVel(0.0, 0.0, 0.0), calibratedAngularVel(0.0, 0.0, 0.0);
	LARGE_INTEGER performanceCount;

	// get measurements of IMU
	if (!imu_->readData(measuredAccel, measuredAngularVel, performanceCount)) return;

	// elpased time [msec]
	const __int64 elapsedTime = (0 == performanceCount.HighPart && 0 == performanceCount.LowPart) ? 0 : ((performanceCount.QuadPart - prevPerformanceCount_.QuadPart) * 1000 / freq_.QuadPart);
	prevPerformanceCount_ = performanceCount;

	//
	imu_->calculateCalibratedAcceleration(measuredAccel, calibratedAccel);
	imu_->calculateCalibratedAngularRate(measuredAngularVel, calibratedAngularVel);

	//
	++step_;

	//
	CString msg;
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
	msg.Format(_T("%d"), elapsedTime);
	GetDlgItem(IDC_EDIT_IMU_ELAPSED_TIME)->SetWindowText(msg);

	msg.Format(_T("%d"), step_);
	GetDlgItem(IDC_EDIT_STEP)->SetWindowText(msg);
}

void CGpsAidedImuFilterDialog::checkGps()
{
	swl::EarthData::Geodetic measuredGpsGeodetic(0.0, 0.0, 0.0);
	swl::EarthData::ECEF measuredGpsECEF(0.0, 0.0, 0.0);
	swl::EarthData::Speed measuredGpsSpeed(0.0);
	swl::EarthData::Time gpsUtc(0, 0, 0, 0);

	// get measurements of GPS
	if (!gps_->readData(measuredGpsGeodetic, measuredGpsSpeed, gpsUtc)) return;
	swl::EarthData::geodetic_to_ecef(measuredGpsGeodetic, measuredGpsECEF);

	measuredGpsECEF.x -= initialGpsECEF_.x;
	measuredGpsECEF.y -= initialGpsECEF_.y;
	measuredGpsECEF.z -= initialGpsECEF_.z;

	// elpased time [msec]
	// TODO [check] >>
	const long elapsedTime = ((gpsUtc.min - prevGpsUtc_.min) * 60 + (gpsUtc.sec - prevGpsUtc_.sec)) * 1000 + (gpsUtc.msec - prevGpsUtc_.msec);

	prevGpsUtc_ = gpsUtc;
	prevGpsECEF_ = measuredGpsECEF;

	//
	++step_;

	CString msg;
	msg.Format(_T("%f"), measuredGpsGeodetic.lat);
	GetDlgItem(IDC_EDIT_GPS_LAT)->SetWindowText(msg);
	msg.Format(_T("%f"), measuredGpsGeodetic.lon);
	GetDlgItem(IDC_EDIT_GPS_LON)->SetWindowText(msg);
	msg.Format(_T("%f"), measuredGpsGeodetic.alt);
	GetDlgItem(IDC_EDIT_GPS_ALT)->SetWindowText(msg);
	msg.Format(_T("%f"), measuredGpsECEF.x);
	GetDlgItem(IDC_EDIT_GPS_X)->SetWindowText(msg);
	msg.Format(_T("%f"), measuredGpsECEF.y);
	GetDlgItem(IDC_EDIT_GPS_Y)->SetWindowText(msg);
	msg.Format(_T("%f"), measuredGpsECEF.z);
	GetDlgItem(IDC_EDIT_GPS_Z)->SetWindowText(msg);
	msg.Format(_T("%f"), 0 == elapsedTime ? 0 : (measuredGpsECEF.x - prevGpsECEF_.x) / elapsedTime * 1000);
	GetDlgItem(IDC_EDIT_GPS_VEL_X)->SetWindowText(msg);
	msg.Format(_T("%f"), 0 == elapsedTime ? 0 : (measuredGpsECEF.y - prevGpsECEF_.y) / elapsedTime * 1000);
	GetDlgItem(IDC_EDIT_GPS_VEL_Y)->SetWindowText(msg);
	msg.Format(_T("%f"), 0 == elapsedTime ? 0 : (measuredGpsECEF.z - prevGpsECEF_.z) / elapsedTime * 1000);
	GetDlgItem(IDC_EDIT_GPS_VEL_Z)->SetWindowText(msg);
	msg.Format(_T("%f"), measuredGpsSpeed.val);
	GetDlgItem(IDC_EDIT_GPS_SPEED)->SetWindowText(msg);
	msg.Format(_T("%d"), elapsedTime);
	GetDlgItem(IDC_EDIT_GPS_ELAPSED_TIME)->SetWindowText(msg);

	msg.Format(_T("%d"), step_);
	GetDlgItem(IDC_EDIT_STEP)->SetWindowText(msg);
}

void CGpsAidedImuFilterDialog::runFilter()
{
	swl::ImuData::Accel measuredAccel(0.0, 0.0, 0.0), calibratedAccel(0.0, 0.0, 0.0);
	swl::ImuData::Gyro measuredAngularVel(0.0, 0.0, 0.0), calibratedAngularVel(0.0, 0.0, 0.0);
	LARGE_INTEGER performanceCount;
	swl::EarthData::Geodetic measuredGpsGeodetic(0.0, 0.0, 0.0);
	swl::EarthData::ECEF measuredGpsECEF(0.0, 0.0, 0.0);
	swl::EarthData::ECEF measuredGpsVel(0.0, 0.0, 0.0);
	swl::EarthData::Speed measuredGpsSpeed(0.0);
	swl::EarthData::Time gpsUtc(0, 0, 0, 0);

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
}
