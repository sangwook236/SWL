//#include "stdafx.h"
#include "swl/Config.h"
#include "ImuSystem.h"
#include "swl/rnd_util/KalmanFilter.h"
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <cmath>
#include <ctime>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

struct Acceleration
{
	Acceleration(const double &_x, const double &_y, const double &_z)
	: x(_x), y(_y), z(_z)
	{}
	Acceleration(const Acceleration &rhs)
	: x(rhs.x), y(rhs.y), z(rhs.z)
	{}

	double x, y, z;
};

struct Gyro
{
	Gyro(const double &_x, const double &_y, const double &_z)
	: x(_x), y(_y), z(_z)
	{}
	Gyro(const Gyro &rhs)
	: x(rhs.x), y(rhs.y), z(rhs.z)
	{}

	double x, y, z;
};

void read_adis16350(std::list<Acceleration> &accels, std::list<Gyro> &gyros)
{
#if 0
	std::ifstream stream("./data/adis16350_data_20100706/adis16350_raw_data_20100701.txt");

	if (!stream.is_open())
	{
		std::cout << "file open error !!!" << std::endl;
		return;
	}

	double xAccel, yAccel, zAccel, xGyro, yGyro, zGyro;

	// eliminate the 1st line
	std::string str;
	std::getline(stream, str);

	while (!stream.eof())
	{
		stream >> xAccel >> yAccel >> zAccel >> xGyro >> yGyro >> zGyro;
		accels.push_back(Acceleration(xAccel, yAccel, zAccel));
		gyros.push_back(Gyro(xGyro, yGyro, zGyro));
	}

	stream.close();
#elif 1
	std::ifstream stream("./data/adis16350_data_20100706/adis16350_raw_data_no_motion_20100706.txt");
	//std::ifstream stream("./data/adis16350_data_20100706/adis16350_raw_data_x_20100706.txt");
	//std::ifstream stream("./data/adis16350_data_20100706/adis16350_raw_data_y_20100706.txt");
	//std::ifstream stream("./data/adis16350_data_20100706/adis16350_raw_data_xy45_20100706.txt");
	//std::ifstream stream("./data/adis16350_data_20100706/adis16350_raw_data_rotation_small_circle_20100706.txt");
	//std::ifstream stream("./data/adis16350_data_20100706/adis16350_raw_data_rotation_large_circle_20100706.txt");

	// data format:
	//	gpsPos_.latitude / gpsPos_.longitude / compass_.heading_ / adjustOrientation /
	//	imu_.adis_.AccX / imu_.adis_.AccY / imu_.adis_.AccZ / imu_.adis_.WdX / imu_.adis_.WdY / imu_.adis_.WdZ /
	//	imu_.estimatedXAccel / imu_.estimatedYAccel / imu_.estimatedZAccel / imu_.estimatedXAngularVel / imu_.estimatedYAngularVel / imu_.estimatedZAngularVel /
	//	dst_pos.x / ref_pos.x / dst_scale / varDisturbance

	if (!stream.is_open())
	{
		std::cout << "file open error !!!" << std::endl;
		return;
	}

	double xAccel, yAccel, zAccel, xGyro, yGyro, zGyro;

	double dummy;
	char slash;
	while (!stream.eof())
	{
		stream >> dummy >> slash >> dummy >> slash >> dummy >> slash >> dummy >> slash >>
			xAccel >> slash >> yAccel >> slash >> zAccel >> slash >> xGyro >> slash >> yGyro >> slash >> zGyro >> slash >>
			dummy >> slash >> dummy >> slash >> dummy >> slash >> dummy >> slash >> dummy >> slash >> dummy >> slash >> 
			dummy >> slash >> dummy >> slash >> dummy >> slash >> dummy;
		accels.push_back(Acceleration(xAccel, yAccel, zAccel));
		gyros.push_back(Gyro(xGyro, yGyro, zGyro));
	}

	stream.close();
#endif
}

}  // namespace local
}  // unnamed namespace

void imu_kalman_filter()
{
	gsl_vector *Xa0 = gsl_vector_alloc(swl::AccelSystem::stateDim);
	gsl_matrix *Pa0 = gsl_matrix_alloc(swl::AccelSystem::stateDim, swl::AccelSystem::stateDim);
	gsl_vector_set_zero(Xa0);
	gsl_matrix_set_zero(Pa0);

	// 2010/07/01 & 2010/07/06
	const double Ts = 0.25;  // sampling time

#if 0
	const double beta_a_x = 0.0, beta_a_y = 0.0, beta_a_z = 0.0;  // correlation time (time constant)
	const double Qv_x = std::sqrt(0.003), Qv_y = std::sqrt(0.003), Qv_z = std::sqrt(0.003);  // variance of a process noise related to a velocity state
	const double Qa_x = 0.003, Qa_y = 0.003, Qa_z = 0.003;  // variance of a process noise related to an acceleration state
	const double Qab_x = 0.0, Qab_y = 0.0, Qab_z = 0.0;  // variance of a process noise related to an acceleration's bias state  ==>  random constant model
	const double Ra_x = 0.01, Ra_y = 0.01, Ra_z = 0.01;  // variance of a measurement noise
#elif 0
	const double beta_a_x = 0.0, beta_a_y = 0.0, beta_a_z = 0.0;  // correlation time (time constant)
	const double Qv_x = std::sqrt(0.003), Qv_y = std::sqrt(0.003), Qv_z = std::sqrt(0.003);  // variance of a process noise related to a velocity state
	const double Qa_x = 0.003, Qa_y = 0.003, Qa_z = 0.003;  // variance of a process noise related to an acceleration state
	const double Qab_x = 0.0001 * 0.0001, Qab_y = 0.0001 * 0.0001, Qab_z = 0.0001 * 0.0001;  // variance of a process noise related to an acceleration's bias state  ==>  random walk model
	const double Ra_x = 0.01, Ra_y = 0.01, Ra_z = 0.01;  // variance of a measurement noise
#elif 0
	const double beta_a_x = 0.0, beta_a_y = 0.0, beta_a_z = 0.0;  // correlation time (time constant)
	const double Qv_x = std::sqrt(0.0003), Qv_y = std::sqrt(0.0003), Qv_z = std::sqrt(0.0003);  // variance of a process noise related to a velocity state
	const double Qa_x = 0.0003, Qa_y = 0.0003, Qa_z = 0.0003;  // variance of a process noise related to an acceleration state
	const double Qab_x = 0.001 * 0.001, Qab_y = 0.001 * 0.001, Qab_z = 0.001 * 0.001;  // variance of a process noise related to an acceleration's bias state  ==>  random walk model
	const double Ra_x = 0.001, Ra_y = 0.001, Ra_z = 0.001;  // variance of a measurement noise
#elif 0
	const double beta_a_x = 0.0, beta_a_y = 0.0, beta_a_z = 0.0;  // correlation time (time constant)
	const double Qv_x = std::sqrt(0.0003), Qv_y = std::sqrt(0.0003), Qv_z = std::sqrt(0.0003);  // variance of a process noise related to a velocity state
	const double Qa_x = 0.0003, Qa_y = 0.0003, Qa_z = 0.0003;  // variance of a process noise related to an acceleration state
	const double Qab_x = 0.001 * 0.001, Qab_y = 0.001 * 0.001, Qab_z = 0.001 * 0.001;  // variance of a process noise related to an acceleration's bias state  ==>  random walk model
	const double Ra_x = 0.001, Ra_y = 0.001, Ra_z = 0.001;  // variance of a measurement noise
#elif 1
	const double beta_a_x = 0.0, beta_a_y = 0.0, beta_a_z = 0.0;  // correlation time (time constant)
	const double Qv_x = std::sqrt(0.03), Qv_y = std::sqrt(0.03), Qv_z = std::sqrt(0.03);  // variance of a process noise related to a velocity state
	const double Qa_x = 0.03, Qa_y = 0.03, Qa_z = 0.03;  // variance of a process noise related to an acceleration state
	const double Qab_x = 1.0, Qab_y = 0.5, Qab_z = 0.8;  // variance of a process noise related to an acceleration's bias state  ==>  random walk model
	const double Ra_x = 1.5, Ra_y = 0.7, Ra_z = 1.0;  // variance of a measurement noise
#elif 0
	const double beta_a_x = 1.0 / 100.0, beta_a_y = 1.0 / 100.0, beta_a_z = 1.0 / 100.0;  // correlation time (time constant)
	const double Qv_x = std::sqrt(0.0003), Qv_y = std::sqrt(0.0003), Qv_z = std::sqrt(0.0003);  // variance of a process noise related to a velocity state
	const double Qa_x = 0.0003, Qa_y = 0.0003, Qa_z = 0.0003;  // variance of a process noise related to an acceleration state
	const double Qab_x = 0.00015, Qab_y = 0.00015, Qab_z = 0.00015;  // variance of a process noise related to an acceleration's bias state  ==>  colored noise model
	const double Ra_x = 0.001, Ra_y = 0.001, Ra_z = 0.001;  // variance of a measurement noise
#endif
	const swl::AccelSystem xAccelSystem(Ts, beta_a_x, Qv_x, Qa_x, Qab_x, Ra_x);
	swl::DiscreteKalmanFilter xAccelFilter(xAccelSystem, Xa0, Pa0);
	const swl::AccelSystem yAccelSystem(Ts, beta_a_y, Qv_y, Qa_y, Qab_y, Ra_y);
	swl::DiscreteKalmanFilter yAccelFilter(yAccelSystem, Xa0, Pa0);
	const swl::AccelSystem zAccelSystem(Ts, beta_a_z, Qv_z, Qa_z, Qab_z, Ra_z);
	swl::DiscreteKalmanFilter zAccelFilter(zAccelSystem, Xa0, Pa0);

	gsl_vector_free(Xa0);  Xa0 = NULL;
	gsl_matrix_free(Pa0);  Pa0 = NULL;

	gsl_vector *accelU = gsl_vector_alloc(swl::AccelSystem::inputDim);
	gsl_vector_set_zero(accelU);
	gsl_vector *accelBu = gsl_vector_alloc(swl::AccelSystem::stateDim);
	gsl_vector_set_zero(accelBu);
	gsl_vector *accelDu = gsl_vector_alloc(swl::AccelSystem::outputDim);
	gsl_vector_set_zero(accelDu);
	gsl_vector *accelMeasurement = gsl_vector_alloc(swl::AccelSystem::outputDim);
	gsl_vector_set_zero(accelMeasurement);

	//
	gsl_vector *Xg0 = gsl_vector_alloc(swl::GyroSystem::stateDim);
	gsl_matrix *Pg0 = gsl_matrix_alloc(swl::GyroSystem::stateDim, swl::GyroSystem::stateDim);
	gsl_vector_set_zero(Xg0);
	gsl_matrix_set_zero(Pg0);

#if 0
	const double beta_g_x = 0.0, beta_g_y = 0.0, beta_g_z = 0.0;  // correlation time (time constant)
	const double Qw_x = 0.2, Qw_y = 0.2, Qw_z = 0.2;  // variance of a process noise related to an angular velocity state
	const double Qwb_x = 0.0, Qwb_y = 0.0, Qwb_z = 0.0;  // variance of a process noise related to an angular velocity's bias state  ==>  random constant model
	const double Rg_x = 0.25, Rg_y = 0.25, Rg_z = 0.25;  // variance of a measurement noise
#elif 0
	const double beta_g_x = 0.0, beta_g_y = 0.0, beta_g_z = 0.0;  // correlation time (time constant)
	const double Qw_x = 2.0, Qw_y = 2.0, Qw_z = 2.0;  // variance of a process noise related to an angular velocity state
	const double Qwb_x = 1.0, Qwb_y = 1.0, Qwb_z = 1.0;  // variance of a process noise related to an angular velocity's bias state  ==>  random walk model
	const double Rg_x = 2.5, Rg_y = 2.5, Rg_z = 2.5;  // variance of a measurement noise
#elif 0
	const double beta_g_x = 0.0, beta_g_y = 0.0, beta_g_z = 0.0;  // correlation time (time constant)
	const double Qw_x = 0.01, Qw_y = 0.01, Qw_z = 0.01;  // variance of a process noise related to an angular velocity state
	const double Qwb_x = 0.001, Qwb_y = 0.001, Qwb_z = 0.001;  // variance of a process noise related to an angular velocity's bias state  ==>  random walk model
	const double Rg_x = 0.015, Rg_y = 0.015, Rg_z = 0.015;  // variance of a measurement noise
#elif 1
	const double beta_g_x = 0.0, beta_g_y = 0.0, beta_g_z = 0.0;  // correlation time (time constant)
	const double Qw_x = 0.01, Qw_y = 0.01, Qw_z = 10.0;  // variance of a process noise related to an angular velocity state
	const double Qwb_x = 0.001, Qwb_y = 0.001, Qwb_z = 1.0;  // variance of a process noise related to an angular velocity's bias state  ==>  random walk model
	const double Rg_x = 0.015, Rg_y = 0.015, Rg_z = 150.0;  // variance of a measurement noise
#elif 0
	const double beta_g_x = 1.0 / 100.0, beta_g_y = 1.0 / 100.0, beta_g_z = 1.0 / 100.0;  // correlation time (time constant)
	const double Qw_x = 0.1, Qw_y = 0.1, Qw_z = 0.1;  // variance of a process noise related to an angular velocity state
	const double Qwb_x = 0.1, Qwb_y = 0.1, Qwb_z = 0.1;  // variance of a process noise related to an angular velocity's bias state  ==>  colored noise model
	const double Rg_x = 0.6, Rg_y = 0.6, Rg_z = 0.6;  // variance of a measurement noise
#elif 0
	const double beta_g_x = 1.0 / 500.0, beta_g_y = 1.0 / 500.0, beta_g_z = 1.0 / 500.0;  // correlation time (time constant)
	const double Qw_x = 0.1, Qw_y = 0.1, Qw_z = 0.1;  // variance of a process noise related to an angular velocity state
	const double Qwb_x = 0.01, Qwb_y = 0.01, Qwb_z = 0.01;  // variance of a process noise related to an angular velocity's bias state  ==>  colored noise model
	const double Rg_x = 0.35, Rg_y = 0.35, Rg_z = 0.35;  // variance of a measurement noise
#endif
	const swl::GyroSystem xGyroSystem(Ts, beta_g_x, Qw_x, Qwb_x, Rg_x);
	swl::DiscreteKalmanFilter xGyroFilter(xGyroSystem, Xa0, Pa0);
	const swl::GyroSystem yGyroSystem(Ts, beta_g_y, Qw_y, Qwb_y, Rg_y);
	swl::DiscreteKalmanFilter yGyroFilter(yGyroSystem, Xa0, Pa0);
	const swl::GyroSystem zGyroSystem(Ts, beta_g_z, Qw_z, Qwb_z, Rg_z);
	swl::DiscreteKalmanFilter zGyroFilter(zGyroSystem, Xa0, Pa0);

	gsl_vector_free(Xg0);  Xg0 = NULL;
	gsl_matrix_free(Pg0);  Pg0 = NULL;

	gsl_vector *gyroBu = gsl_vector_alloc(swl::GyroSystem::stateDim);
	gsl_vector_set_zero(gyroBu);
	gsl_vector *gyroDu = gsl_vector_alloc(swl::GyroSystem::outputDim);
	gsl_vector_set_zero(gyroDu);
	gsl_vector *gyroMeasurement = gsl_vector_alloc(swl::GyroSystem::outputDim);
	gsl_vector_set_zero(gyroMeasurement);

	//
	std::list<local::Acceleration> accels;
	std::list<local::Gyro> gyros;
	read_adis16350(accels, gyros);

	//
	std::list<local::Acceleration>::iterator itAccel = accels.begin(), itAccelEnd = accels.end();
	std::list<local::Gyro>::iterator itGyro = gyros.begin(), itGyroEnd = gyros.end();

#if 1
	double prioriEstimate, posterioriEstimate;
	size_t step = 0;
	while (itAccel != itAccelEnd && itGyro != itGyroEnd)
	{
		{
			// g_x: the x-component of gravity, a_Fx: the x-component of the acceleration exerted by the robot's input force
			// Bu = Bd * (g_x + a_Fx)
			const gsl_matrix *Bd = xAccelSystem.getInputMatrix();
			// FIMXE [modify] >> 
			const double g_x = 0.0;
			const double a_Fx = 0.0;
			gsl_vector_set(accelU, 0, g_x + a_Fx);
			gsl_blas_dgemv(CblasNoTrans, 1.0, Bd, accelU, 0.0, accelBu);

			// y_tilde = a measured x-axis acceleration
			// FIMXE [modify] >> 
			const double accelActualMeasurement = 0.0;
			gsl_vector_set(accelMeasurement, 0, accelActualMeasurement);

			const bool retval = xAccelSystem.runStep(xAccelFilter, step, accelBu, accelDu, accelMeasurement, prioriEstimate, posterioriEstimate);
			assert(retval);

			const gsl_vector *x_hat = xAccelFilter.getEstimatedState();
			//const double velEstimate = gsl_vector_get(x_hat, 0);
			const double accelEstimate = gsl_vector_get(x_hat, 1);
			//const double biasEstimate = gsl_vector_get(x_hat, 2);
		}

		{
			// g_y: the y-component of gravity, a_Fy: the y-component of the acceleration exerted by the robot's input force
			// Bu = Bd * (g_y + a_Fy)
			const gsl_matrix *Bd = yAccelSystem.getInputMatrix();
			// FIMXE [modify] >> 
			const double g_y = 0.0;
			const double a_Fy = 0.0;
			gsl_vector_set(accelU, 0, g_y + a_Fy);
			gsl_blas_dgemv(CblasNoTrans, 1.0, Bd, accelU, 0.0, accelBu);

			// y_tilde = a measured y-axis acceleration
			// FIMXE [modify] >> 
			const double accelActualMeasurement = 0.0;
			gsl_vector_set(accelMeasurement, 0, accelActualMeasurement);

			const bool retval = yAccelSystem.runStep(yAccelFilter, step, accelBu, accelDu, accelMeasurement, prioriEstimate, posterioriEstimate);
			assert(retval);

			const gsl_vector *x_hat = yAccelFilter.getEstimatedState();
			//const double velEstimate = gsl_vector_get(x_hat, 0);
			const double accelEstimate = gsl_vector_get(x_hat, 1);
			//const double biasEstimate = gsl_vector_get(x_hat, 2);
		}

		{
			// g_z: the z-component of gravity, a_Fz: the z-component of the acceleration exerted by the robot's input force
			// Bu = Bd * (g_z + a_Fz)
			const gsl_matrix *Bd = zAccelSystem.getInputMatrix();
			// FIMXE [modify] >> 
			const double g_z = 0.0;
			const double a_Fz = 0.0;
			gsl_vector_set(accelU, 0, g_z + a_Fz);
			gsl_blas_dgemv(CblasNoTrans, 1.0, Bd, accelU, 0.0, accelBu);

			// y_tilde = a measured z-axis acceleration
			// FIMXE [modify] >> 
			const double accelActualMeasurement = 0.0;
			gsl_vector_set(accelMeasurement, 0, accelActualMeasurement);

			const bool retval = zAccelSystem.runStep(zAccelFilter, step, accelBu, accelDu, accelMeasurement, prioriEstimate, posterioriEstimate);
			assert(retval);

			const gsl_vector *x_hat = zAccelFilter.getEstimatedState();
			//const double velEstimate = gsl_vector_get(x_hat, 0);
			const double accelEstimate = gsl_vector_get(x_hat, 1);
			//const double biasEstimate = gsl_vector_get(x_hat, 2);
		}

		{
			// y_tilde = a measured x-axis angular velocity
			// FIMXE [modify] >> 
			const double gyroActualMeasurement = 0.0;
			gsl_vector_set(gyroMeasurement, 0, gyroActualMeasurement);

			const bool retval = xGyroSystem.runStep(xGyroFilter, step, gyroBu, gyroDu, gyroMeasurement, prioriEstimate, posterioriEstimate);
			assert(retval);

			const gsl_vector *x_hat = xGyroFilter.getEstimatedState();
			const double angularVelEstimate = gsl_vector_get(x_hat, 0);
			//const double biasEstimate = gsl_vector_get(x_hat, 1);
		}

		{
			// y_tilde = a measured y-axis angular velocity
			// FIMXE [modify] >> 
			const double gyroActualMeasurement = 0.0;
			gsl_vector_set(gyroMeasurement, 0, gyroActualMeasurement);

			const bool retval = yGyroSystem.runStep(yGyroFilter, step, gyroBu, gyroDu, gyroMeasurement, prioriEstimate, posterioriEstimate);
			assert(retval);

			const gsl_vector *x_hat = yGyroFilter.getEstimatedState();
			const double angularVelEstimate = gsl_vector_get(x_hat, 0);
			//const double biasEstimate = gsl_vector_get(x_hat, 1);
		}

		{
			// y_tilde = a measured z-axis angular velocity
			// FIMXE [modify] >> 
			const double gyroActualMeasurement = 0.0;
			gsl_vector_set(gyroMeasurement, 0, gyroActualMeasurement);

			const bool retval = zGyroSystem.runStep(zGyroFilter, step, gyroBu, gyroDu, gyroMeasurement, prioriEstimate, posterioriEstimate);
			assert(retval);

			const gsl_vector *x_hat = zGyroFilter.getEstimatedState();
			const double angularVelEstimate = gsl_vector_get(x_hat, 0);
			//const double biasEstimate = gsl_vector_get(x_hat, 1);
		}

		++itAccel;
		++itGyro;

		// advance time step
		++step;
	}
#else
	const size_t Nstep = std::min(accels.size(), gyros.size());

	std::ofstream streamAccelState("./data/accel_state.txt", std::ios::in | std::ios::trunc), streamAccelCovar("./data/accel_covar.txt", std::ios::in | std::ios::trunc), streamAccelGain("./data/accel_gain.txt", std::ios::in | std::ios::trunc);
	std::ofstream streamGyroState("./data/gyro_state.txt", std::ios::in | std::ios::trunc), streamGyroCovar("./data/gyro_covar.txt", std::ios::in | std::ios::trunc), streamGyroGain("./data/gyro_gain.txt", std::ios::in | std::ios::trunc);

	if (!streamAccelState.is_open() || !streamAccelCovar.is_open() || !streamAccelGain.is_open() ||
		!streamGyroState.is_open() || !streamGyroCovar.is_open() || !streamGyroGain.is_open())
	{
		std::cout << "file open error !!!" << std::endl;
		return;
	}

	size_t step = 0;
	while (itAccel != itAccelEnd && itGyro != itGyroEnd)
	{
#if 0  // x-axis
		const swl::AccelSystem &accelSystem = xAccelSystem;
		const swl::GyroSystem &gyroSystem = xGyroSystem;
		swl::DiscreteKalmanFilter &accelFilter = xAccelFilter;
		swl::DiscreteKalmanFilter &gyroFilter = xGyroFilter;

		// FIMXE [modify] >> 
		const double &accelVal = itAccel->x;
		const double &gyroVal = itGyro->x;
		const double g_x = 0.0;
		const double a_Fx = 0.0;
		const double accelInput = g_x + a_Fx;
#elif 0  // y-axis
		const swl::AccelSystem &accelSystem = yAccelSystem;
		const swl::GyroSystem &gyroSystem = yGyroSystem;
		swl::DiscreteKalmanFilter &accelFilter = yAccelFilter;
		swl::DiscreteKalmanFilter &gyroFilter = yGyroFilter;

		// FIMXE [modify] >> 
		const double &accelVal = itAccel->y;
		const double &gyroVal = itGyro->y;
		const double g_y = 0.0;
		const double a_Fy = 0.0;
		const double accelInput = g_y + a_Fy;
#elif 1  // z-axis
		const swl::AccelSystem &accelSystem = zAccelSystem;
		const swl::GyroSystem &gyroSystem = zGyroSystem;
		swl::DiscreteKalmanFilter &accelFilter = zAccelFilter;
		swl::DiscreteKalmanFilter &gyroFilter = zGyroFilter;

		// FIMXE [modify] >> 
		const double &accelVal = itAccel->z;
		const double &gyroVal = itGyro->z;
		const double g_z = 0.0;
		const double a_Fz = 0.0;
		const double accelInput = g_z + a_Fz;
#endif

		{
			// 0. initial estimates: x-(0) & P-(0)

			// 1. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
			// y_tilde = a measured x-axis acceleration
			gsl_vector_set(accelMeasurement, 0, accelVal);

			if (!accelFilter.updateMeasurement(step, accelMeasurement, accelDu))
				return;

			// save K(k), x(k) & P(k)
			{
				const gsl_vector *x_hat = accelFilter.getEstimatedState();
				const gsl_matrix *K = accelFilter.getKalmanGain();
				const gsl_matrix *P = accelFilter.getStateErrorCovarianceMatrix();

				streamAccelState << gsl_vector_get(x_hat, 0) << ", " << gsl_vector_get(x_hat, 1) << ", " << gsl_vector_get(x_hat, 2) << std::endl;
				streamAccelCovar << gsl_matrix_get(P, 0, 0) << ", " << gsl_matrix_get(P, 1, 1) << ", " << gsl_matrix_get(P, 2, 2) << std::endl;
				streamAccelGain << gsl_matrix_get(K, 0, 0) << ", " << gsl_matrix_get(K, 1, 0) << ", " << gsl_matrix_get(K, 2, 0) << std::endl;
			}

			// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
			// g_x: the x-component of gravity, a_Fx: the x-component of the acceleration exerted by the robot's input force
			// Bu = Bd * (g_x + a_Fx)
			const gsl_matrix *Bd = accelSystem.getInputMatrix();
			gsl_vector_set(accelU, 0, accelInput);
			gsl_blas_dgemv(CblasNoTrans, 1.0, Bd, accelU, 0.0, accelBu);

			if (!accelFilter.updateTime(step, accelBu))
				return;

			// save x-(k+1) & P-(k+1)
			{
				const gsl_vector *x_hat = accelFilter.getEstimatedState();
				const gsl_matrix *P = accelFilter.getStateErrorCovarianceMatrix();

				streamAccelState << gsl_vector_get(x_hat, 0) << ", " << gsl_vector_get(x_hat, 1) << ", " << gsl_vector_get(x_hat, 2) << std::endl;
				streamAccelCovar << gsl_matrix_get(P, 0, 0) << ", " << gsl_matrix_get(P, 1, 1) << ", " << gsl_matrix_get(P, 2, 2) << std::endl;
			}
		}

		{
			// 0. initial estimates: x-(0) & P-(0)

			// 1. measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
			// y_tilde = a measured x-axis angular velocity
			gsl_vector_set(gyroMeasurement, 0, gyroVal);

			if (!gyroFilter.updateMeasurement(step, gyroMeasurement, gyroDu))
				return;

			// save K(k), x(k) & P(k)
			{
				const gsl_vector *x_hat = gyroFilter.getEstimatedState();
				const gsl_matrix *K = gyroFilter.getKalmanGain();
				const gsl_matrix *P = gyroFilter.getStateErrorCovarianceMatrix();

				streamGyroState << gsl_vector_get(x_hat, 0) << ", " << gsl_vector_get(x_hat, 1) << std::endl;
				streamGyroCovar << gsl_matrix_get(P, 0, 0) << ", " << gsl_matrix_get(P, 1, 1) << std::endl;
				streamGyroGain << gsl_matrix_get(K, 0, 0) << ", " << gsl_matrix_get(K, 1, 0) << std::endl;
			}

			// 2. time update (prediction): x(k) & P(k)  ==>  x-(k+1) & P-(k+1)
			if (!gyroFilter.updateTime(step, gyroBu))
				return;

			// save x-(k+1) & P-(k+1)
			{
				const gsl_vector *x_hat = gyroFilter.getEstimatedState();
				const gsl_matrix *P = gyroFilter.getStateErrorCovarianceMatrix();

				streamGyroState << gsl_vector_get(x_hat, 0) << ", " << gsl_vector_get(x_hat, 1) << std::endl;
				streamGyroCovar << gsl_matrix_get(P, 0, 0) << ", " << gsl_matrix_get(P, 1, 1) << std::endl;
			}
		}

		++itAccel;
		++itGyro;

		// advance time step
		++step;
	}
#endif

	gsl_vector_free(accelU);  accelU = NULL;
	gsl_vector_free(accelBu);  accelBu = NULL;
	gsl_vector_free(accelDu);  accelDu = NULL;
	gsl_vector_free(accelMeasurement);  accelMeasurement = NULL;
	gsl_vector_free(gyroBu);  gyroBu = NULL;
	gsl_vector_free(gyroDu);  gyroDu = NULL;
	gsl_vector_free(gyroMeasurement);  gyroMeasurement = NULL;
}
