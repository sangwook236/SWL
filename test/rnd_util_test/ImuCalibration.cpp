//#include "stdafx.h"
#include "swl/Config.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <boost/math/constants/constants.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
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

void print_gsl_vector(const gsl_vector *vec)
{
	std::cout << "[ ";
	for (size_t i = 0; i < vec->size; ++i)
	{
		if (0 != i) std::cout << ' ';
		std::cout << gsl_vector_get(vec, i);
	}
	std::cout << " ]\n";
}

void print_gsl_matrix(const gsl_matrix *mat)
{
	for (size_t i = 0; i < mat->size1; ++i)
	{
		gsl_vector_view rvec = gsl_matrix_row(const_cast<gsl_matrix *>(mat), i);
		print_gsl_vector(&rvec.vector);
	}
}

void matrix_inverse(const size_t dim, const gsl_matrix *A, gsl_matrix *invA)
{
	gsl_permutation *p = gsl_permutation_alloc(dim);
	int signum;

	gsl_matrix *B = gsl_matrix_alloc(A->size1, A->size2);
	gsl_matrix_memcpy(B, A);

	// LU decomposition
	gsl_linalg_LU_decomp(B, p, &signum);
	gsl_linalg_LU_invert(B, p, invA);

	gsl_permutation_free(p);
	gsl_matrix_free(B);
}

void calculate_calibrated_acceleration(const gsl_vector *param, const gsl_vector *lg, gsl_vector *a_calibrated)
{
	const double &b_gx = gsl_vector_get(param, 0);
	const double &b_gy = gsl_vector_get(param, 1);
	const double &b_gz = gsl_vector_get(param, 2);
	const double &s_gx = gsl_vector_get(param, 3);
	const double &s_gy = gsl_vector_get(param, 4);
	const double &s_gz = gsl_vector_get(param, 5);
	const double &theta_gyz = gsl_vector_get(param, 6);
	const double &theta_gzx = gsl_vector_get(param, 7);
	const double &theta_gzy = gsl_vector_get(param, 8);

	const double &l_gx = gsl_vector_get(lg, 0);
	const double &l_gy = gsl_vector_get(lg, 1);
	const double &l_gz = gsl_vector_get(lg, 2);

	const double tan_gyz = std::tan(theta_gyz);
	const double tan_gzx = std::tan(theta_gzx);
	const double tan_gzy = std::tan(theta_gzy);
	const double cos_gyz = std::cos(theta_gyz);
	const double cos_gzx = std::cos(theta_gzx);
	const double cos_gzy = std::cos(theta_gzy);

	const double g_x = (l_gx - b_gx) / (1.0 + s_gx);
	const double g_y = tan_gyz * (l_gx - b_gx) / (1.0 + s_gx) + (l_gy - b_gy) / ((1.0 + s_gy) * cos_gyz);
	const double g_z = (tan_gzx * tan_gyz - tan_gzy / cos_gzx) * (l_gx - b_gx) / (1.0 + s_gx) +
		((l_gy - b_gy) * tan_gzx) / ((1.0 + s_gy) * cos_gyz) + (l_gz - b_gz) / ((1.0 + s_gz) * cos_gzx * cos_gzy);

	gsl_vector_set(a_calibrated, 0, g_x);
	gsl_vector_set(a_calibrated, 1, g_y);
	gsl_vector_set(a_calibrated, 2, g_z);
}

void evaluate_Fg(const gsl_vector *param, const gsl_vector *lg, const double g_true, double &Fg, gsl_vector *dFgdx, gsl_vector *dFgdl)
{
	const double &b_gx = gsl_vector_get(param, 0);
	const double &b_gy = gsl_vector_get(param, 1);
	const double &b_gz = gsl_vector_get(param, 2);
	const double &s_gx = gsl_vector_get(param, 3);
	const double &s_gy = gsl_vector_get(param, 4);
	const double &s_gz = gsl_vector_get(param, 5);
	const double &theta_gyz = gsl_vector_get(param, 6);
	const double &theta_gzx = gsl_vector_get(param, 7);
	const double &theta_gzy = gsl_vector_get(param, 8);

	const double &l_gx = gsl_vector_get(lg, 0);
	const double &l_gy = gsl_vector_get(lg, 1);
	const double &l_gz = gsl_vector_get(lg, 2);

	const double tan_gyz = std::tan(theta_gyz);
	const double tan_gzx = std::tan(theta_gzx);
	const double tan_gzy = std::tan(theta_gzy);
	const double cos_gyz = std::cos(theta_gyz);
	const double cos_gzx = std::cos(theta_gzx);
	const double cos_gzy = std::cos(theta_gzy);

	const double g_x = (l_gx - b_gx) / (1.0 + s_gx);
	const double g_y = tan_gyz * (l_gx - b_gx) / (1.0 + s_gx) + (l_gy - b_gy) / ((1.0 + s_gy) * cos_gyz);
	const double g_z = (tan_gzx * tan_gyz - tan_gzy / cos_gzx) * (l_gx - b_gx) / (1.0 + s_gx) +
		((l_gy - b_gy) * tan_gzx) / ((1.0 + s_gy) * cos_gyz) + (l_gz - b_gz) / ((1.0 + s_gz) * cos_gzx * cos_gzy);

	//
	Fg = g_x*g_x + g_y*g_y + g_z*g_z - g_true*g_true;

	//
#if 1
	const double num1 = g_x + g_y * tan_gyz + g_z * (tan_gzx * tan_gyz - tan_gzy / cos_gzx);
	const double num2 = g_y + g_z * tan_gzx;
	const double dFgdBgx = -2.0 * num1 / (1.0 + s_gx);
	const double dFgdBgy = -2.0 * num2 / ((1.0 + s_gy) * cos_gyz);
	const double dFgdBgz = -2.0 * g_z / ((1.0 + s_gz) * cos_gzx * cos_gzy);
	const double dFgdSgx = -2.0 * (l_gx - b_gx) * num1 / ((1.0 + s_gx)*(1.0 + s_gx));
	const double dFgdSgy = -2.0 * (l_gy - b_gy) * num2 / ((1.0 + s_gy)*(1.0 + s_gy) * cos_gyz);
	const double dFgdSgz = -2.0 * (l_gz - b_gz) * g_z / ((1.0 + s_gz)*(1.0 + s_gz) * cos_gzx * cos_gzy);
	const double dFgdTgyz = 2.0 * (g_y + g_z * tan_gzx) * ((l_gx - b_gx) / ((1.0 + s_gx) * cos_gyz*cos_gyz) + ((l_gy - b_gy) * tan_gyz) / (1.0 + s_gy));
	const double dFgdTgzx = 2.0 * g_x * ((tan_gyz / (cos_gzx*cos_gzx) - tan_gzx * tan_gzy) * (l_gx - b_gx) / (1.0 + s_gx) + (l_gy - b_gy) / (cos_gzx*cos_gzx * cos_gyz * (1.0 + s_gy)) + (tan_gzx * (l_gz - b_gz)) / (cos_gzy * (1.0 + s_gz)));
	const double dFgdTgzy = 2.0 * g_z * (-(l_gx - b_gx) / (cos_gzx * cos_gzy*cos_gzy * (1.0 + s_gx)) + (tan_gzy * (l_gz - b_gz)) / (cos_gzx * (1.0 + s_gz)));
	const double dFgdLgx = -dFgdBgx;
	const double dFgdLgy = -dFgdBgy;
	const double dFgdLgz = -dFgdBgz;

	gsl_vector_set(dFgdx, 0, dFgdBgx);
	gsl_vector_set(dFgdx, 1, dFgdBgy);
	gsl_vector_set(dFgdx, 2, dFgdBgz);
	gsl_vector_set(dFgdx, 3, dFgdSgx);
	gsl_vector_set(dFgdx, 4, dFgdSgy);
	gsl_vector_set(dFgdx, 5, dFgdSgz);
	gsl_vector_set(dFgdx, 6, dFgdTgyz);
	gsl_vector_set(dFgdx, 7, dFgdTgzx);
	gsl_vector_set(dFgdx, 8, dFgdTgzy);
	gsl_vector_set(dFgdl, 0, dFgdLgx);
	gsl_vector_set(dFgdl, 1, dFgdLgy);
	gsl_vector_set(dFgdl, 2, dFgdLgz);
#else
	const double dFgdBgx = -2.0 * (l_gx - b_gx) / ((1.0 + s_gx)*(1.0 + s_gx));
	const double dFgdBgy = -2.0 * (l_gy - b_gy) / ((1.0 + s_gy)*(1.0 + s_gy));
	const double dFgdBgz = -2.0 * (l_gz - b_gz) / ((1.0 + s_gz)*(1.0 + s_gz));
	const double dFgdSgx = -2.0 * (l_gx - b_gx)*(l_gx - b_gx) / ((1.0 + s_gx)*(1.0 + s_gx)*(1.0 + s_gx));
	const double dFgdSgy = -2.0 * (l_gy - b_gy)*(l_gy - b_gy) / ((1.0 + s_gy)*(1.0 + s_gy)*(1.0 + s_gy));
	const double dFgdSgz = -2.0 * (l_gz - b_gz)*(l_gz - b_gz) / ((1.0 + s_gz)*(1.0 + s_gz)*(1.0 + s_gz));
	const double dFgdLgx = -dFgdBgx;
	const double dFgdLgy = -dFgdBgy;
	const double dFgdLgz = -dFgdBgz;

	gsl_vector_set(dFgdx, 0, dFgdBgx);
	gsl_vector_set(dFgdx, 1, dFgdBgy);
	gsl_vector_set(dFgdx, 2, dFgdBgz);
	gsl_vector_set(dFgdx, 3, dFgdSgx);
	gsl_vector_set(dFgdx, 4, dFgdSgy);
	gsl_vector_set(dFgdx, 5, dFgdSgz);
	//gsl_vector_set(dFgdx, 6, 0.0);
	//gsl_vector_set(dFgdx, 7, 0.0);
	//gsl_vector_set(dFgdx, 8, 0.0);
	gsl_vector_set(dFgdl, 0, dFgdLgx);
	gsl_vector_set(dFgdl, 1, dFgdLgy);
	gsl_vector_set(dFgdl, 2, dFgdLgz);
#endif
}

void calculate_calibrated_angular_rate(const gsl_vector *param, const gsl_vector *lw, gsl_vector *w_calibrated)
{
	const double &b_wx = gsl_vector_get(param, 0);
	const double &b_wy = gsl_vector_get(param, 1);
	const double &b_wz = gsl_vector_get(param, 2);

	const double &l_wx = gsl_vector_get(lw, 0);
	const double &l_wy = gsl_vector_get(lw, 1);
	const double &l_wz = gsl_vector_get(lw, 2);

	const double w_x = l_wx - b_wx;
	const double w_y = l_wy - b_wy;
	const double w_z = l_wz - b_wz;

	gsl_vector_set(w_calibrated, 0, w_x);
	gsl_vector_set(w_calibrated, 1, w_y);
	gsl_vector_set(w_calibrated, 2, w_z);
}

void evaluate_Fw(const gsl_vector *param, const gsl_vector *lw, const double w_true, double &Fw, gsl_vector *dFwdx, gsl_vector *dFwdl)
{
	const double &b_wx = gsl_vector_get(param, 0);
	const double &b_wy = gsl_vector_get(param, 1);
	const double &b_wz = gsl_vector_get(param, 2);

	const double &l_wx = gsl_vector_get(lw, 0);
	const double &l_wy = gsl_vector_get(lw, 1);
	const double &l_wz = gsl_vector_get(lw, 2);

	const double w_x = l_wx - b_wx;
	const double w_y = l_wy - b_wy;
	const double w_z = l_wz - b_wz;

	//
	Fw = w_x*w_x + w_y*w_y + w_z*w_z - w_true*w_true;

	//
	const double dFwdBwx = -2.0 * (l_wx - b_wx);
	const double dFwdBwy = -2.0 * (l_wy - b_wy);
	const double dFwdBwz = -2.0 * (l_wz - b_wz);
	const double dFwdLwx = -dFwdBwx;
	const double dFwdLwy = -dFwdBwy;
	const double dFwdLwz = -dFwdBwz;

	gsl_vector_set(dFwdx, 0, dFwdBwx);
	gsl_vector_set(dFwdx, 1, dFwdBwy);
	gsl_vector_set(dFwdx, 2, dFwdBwz);
	gsl_vector_set(dFwdl, 0, dFwdLwx);
	gsl_vector_set(dFwdl, 1, dFwdLwy);
	gsl_vector_set(dFwdl, 2, dFwdLwz);
}

void accelerometer_calibration(const size_t Nv, const size_t Nm, const double g_true, const gsl_vector *x, const gsl_matrix *Cx, const std::vector<Acceleration> &measures, const std::vector<Acceleration> &measureVars, gsl_vector *delta_hat, gsl_matrix *Cx_hat)
{
	const double eps = 1.0e-15;

	gsl_vector *lg = gsl_vector_alloc(3);

	// TODO [check] >> Cl
	//	use the arithmetic mean of variances
	const double sigma_lgx2 = gsl_stats_mean((double *)&measureVars[0], 3, Nm);
	const double sigma_lgy2 = gsl_stats_mean((double *)&measureVars[0] + 1, 3, Nm);
	const double sigma_lgz2 = gsl_stats_mean((double *)&measureVars[0] + 2, 3, Nm);

	//
	gsl_matrix *A = gsl_matrix_alloc(Nm, Nv);
	gsl_vector *M = gsl_vector_alloc(Nm);
	gsl_vector *w = gsl_vector_alloc(Nm);

	gsl_vector *dFgdx = gsl_vector_alloc(Nv);
	gsl_vector *dFgdl = gsl_vector_alloc(3);
	double Fg = 0.0;
	for (size_t i = 0; i < Nm; ++i)
	{
		gsl_vector_set(lg, 0, measures[i].x);
		gsl_vector_set(lg, 1, measures[i].y);
		gsl_vector_set(lg, 2, measures[i].z);

		evaluate_Fg(x, lg, g_true, Fg, dFgdx, dFgdl);

		gsl_vector_memcpy(&gsl_matrix_row(A, i).vector, dFgdx);
		{
			const double &dFgdLgx = gsl_vector_get(dFgdl, 0);
			const double &dFgdLgy = gsl_vector_get(dFgdl, 1);
			const double &dFgdLgz = gsl_vector_get(dFgdl, 2);

			const double mm = sigma_lgx2 * dFgdLgx*dFgdLgx + sigma_lgy2 * dFgdLgy*dFgdLgy + sigma_lgz2 * dFgdLgz*dFgdLgz;
			assert(std::fabs(mm) > eps);
			gsl_vector_set(M, i, mm);
		}
		gsl_vector_set(w, i, Fg);
	}

	gsl_vector_free(lg);
	gsl_vector_free(dFgdx);
	gsl_vector_free(dFgdl);

	// inverse of Cx
	gsl_matrix *invCx = gsl_matrix_alloc(Nv, Nv);
	matrix_inverse(Nv, Cx, invCx);

	//
	gsl_matrix *N = gsl_matrix_alloc(Nv, Nv);
	gsl_vector *u = gsl_vector_alloc(Nv);

	for (size_t i = 0; i < Nv; ++i)
	{
		for (size_t j = 0; j < Nv; ++j)
		{
			double sum = 0.0;
			for (size_t k = 0; k < Nm; ++k)
				sum += gsl_matrix_get(A, k, i) * gsl_matrix_get(A, k, j) / gsl_vector_get(M, k);
			gsl_matrix_set(N, i, j, sum + gsl_matrix_get(invCx, i, j));
		}

		{
			double sum = 0.0;
			for (size_t k = 0; k < Nm; ++k)
				sum += gsl_matrix_get(A, k, i) * gsl_vector_get(w, k) / gsl_vector_get(M, k);
			gsl_vector_set(u, i, sum);
		}
	}

	gsl_matrix_free(A);
	gsl_vector_free(M);
	gsl_vector_free(w);
	gsl_matrix_free(invCx);

	// inverse of N
	matrix_inverse(Nv, N, Cx_hat);

	gsl_blas_dgemv(CblasNoTrans, -1.0, Cx_hat, u, 0.0, delta_hat);

	gsl_matrix_free(N);
	gsl_vector_free(u);

	//std::cout << "accelerometer delta_hat = ";
	//print_gsl_vector(delta_hat);
	//std::cout << "accelerometer Cx_hat = ";
	//print_gsl_matrix(Cx_hat);
}

void gyroscope_calibration(const size_t Nv, const size_t Nm, const double w_true, const gsl_vector *x, const gsl_matrix *Cx, const std::vector<Gyro> &measures, const std::vector<Gyro> &measureVars, gsl_vector *delta_hat, gsl_matrix *Cx_hat)
{
	const double eps = 1.0e-15;

	gsl_vector *lw = gsl_vector_alloc(3);

	// TODO [check] >> Cl
	//	use the arithmetic mean of variances
	const double sigma_lwx2 = gsl_stats_mean((double *)&measureVars[0], 3, Nm);
	const double sigma_lwy2 = gsl_stats_mean((double *)&measureVars[0] + 1, 3, Nm);
	const double sigma_lwz2 = gsl_stats_mean((double *)&measureVars[0] + 2, 3, Nm);

	//
	gsl_matrix *A = gsl_matrix_alloc(Nm, Nv);
	gsl_vector *M = gsl_vector_alloc(Nm);
	gsl_vector *w = gsl_vector_alloc(Nm);

	gsl_vector *dFwdx = gsl_vector_alloc(Nv);
	gsl_vector *dFwdl = gsl_vector_alloc(3);
	double Fw = 0.0;
	for (size_t i = 0; i < Nm; ++i)
	{
		gsl_vector_set(lw, 0, measures[i].x);
		gsl_vector_set(lw, 1, measures[i].y);
		gsl_vector_set(lw, 2, measures[i].z);

		evaluate_Fw(x, lw, w_true, Fw, dFwdx, dFwdl);

		gsl_vector_memcpy(&gsl_matrix_row(A, i).vector, dFwdx);
		{
			const double &dFwdLwx = gsl_vector_get(dFwdl, 0);
			const double &dFwdLwy = gsl_vector_get(dFwdl, 1);
			const double &dFwdLwz = gsl_vector_get(dFwdl, 2);

			const double mm = sigma_lwx2 * dFwdLwx*dFwdLwx + sigma_lwy2 * dFwdLwy*dFwdLwy + sigma_lwz2 * dFwdLwz*dFwdLwz;
			assert(std::fabs(mm) > eps);
			gsl_vector_set(M, i, mm);
		}
		gsl_vector_set(w, i, Fw);
	}

	gsl_vector_free(lw);
	gsl_vector_free(dFwdx);
	gsl_vector_free(dFwdl);

	// inverse of Cx
	gsl_matrix *invCx = gsl_matrix_alloc(Nv, Nv);
	matrix_inverse(Nv, Cx, invCx);

	//
	gsl_matrix *N = gsl_matrix_alloc(Nv, Nv);
	gsl_vector *u = gsl_vector_alloc(Nv);

	for (size_t i = 0; i < Nv; ++i)
	{
		for (size_t j = 0; j < Nv; ++j)
		{
			double sum = 0.0;
			for (size_t k = 0; k < Nm; ++k)
				sum += gsl_matrix_get(A, k, i) * gsl_matrix_get(A, k, j) / gsl_vector_get(M, k);
			gsl_matrix_set(N, i, j, sum + gsl_matrix_get(invCx, i, j));
		}

		{
			double sum = 0.0;
			for (size_t k = 0; k < Nm; ++k)
				sum += gsl_matrix_get(A, k, i) * gsl_vector_get(w, k) / gsl_vector_get(M, k);
			gsl_vector_set(u, i, sum);
		}
	}

	gsl_matrix_free(A);
	gsl_vector_free(M);
	gsl_vector_free(w);
	gsl_matrix_free(invCx);

	// inverse of N
	matrix_inverse(Nv, N, Cx_hat);

	gsl_blas_dgemv(CblasNoTrans, -1.0, Cx_hat, u, 0.0, delta_hat);

	gsl_matrix_free(N);
	gsl_vector_free(u);

	//std::cout << "gyro delta_hat = ";
	//print_gsl_vector(delta_hat);
	//std::cout << "gyro Cx_hat = ";
	//print_gsl_matrix(Cx_hat);
}

void load_imu_data(const std::string &filename, const double ref_gravity, Acceleration &meanAccel, Acceleration &varAccel, Gyro &meanGyro, Gyro &varGyro)
{
	std::ifstream stream(filename.c_str());

	// data format:
	//	Sample #,Time (sec),XgND,X Gryo,YgND,Y Gyro,ZgND,Z Gyro,XaND,X acc,YaND,Y acc,ZaND,Z acc,

	if (!stream.is_open())
	{
		std::ostringstream stream;
		stream << "file open error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	// eliminate the 1st 7 lines
	{
		std::string str;
		for (int i = 0; i < 7; ++i)
		{
			if (!stream.eof())
				std::getline(stream, str);
			else
			{
				std::cout << "file format error !!!" << std::endl;
				return;
			}
		}
	}

	//
	std::vector<Acceleration> accels;
	std::vector<Gyro> gyros;
	accels.reserve(10000);
	gyros.reserve(10000);

	double xAccel, yAccel, zAccel, xGyro, yGyro, zGyro;

	const double deg2rad = boost::math::constants::pi<double>() / 180.0;
	int dummy;
	double dummy1;
	char comma;
	while (!stream.eof())
	{
		stream >> dummy >> comma >> dummy1 >> comma >>
			dummy >> comma >> xGyro >> comma >>
			dummy >> comma >> yGyro >> comma >>
			dummy >> comma >> zGyro >> comma >>
			dummy >> comma >> xAccel >> comma >>
			dummy >> comma >> yAccel >> comma >>
			dummy >> comma >> zAccel >> comma;
		if (stream)
		{
			accels.push_back(Acceleration(xAccel * ref_gravity, yAccel * ref_gravity, zAccel * ref_gravity));  // [m/sec^2]
			gyros.push_back(Gyro(xGyro * deg2rad, yGyro * deg2rad, zGyro * deg2rad));  // [rad/sec]
		}
	}

	if (accels.empty() || gyros.empty())
	{
		std::ostringstream stream;
		stream << "data error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	// TODO [check] >>
	//	use arithmetic mean
	const size_t numAccel = accels.size();
	const size_t numGyro = gyros.size();
#if 0
	double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
	for (std::vector<Acceleration>::iterator it = accels.begin(); it != accels.end(); ++it)
	{
		sumX += it->x;
		sumY += it->y;
		sumZ += it->z;
	}
	meanAccel.x = sumX / numAccel;
	meanAccel.y = sumY / numAccel;
	meanAccel.z = sumZ / numAccel;
	varAccel.x = ...;
	varAccel.y = ...;
	varAccel.z = ...;

	sumX = sumY = sumZ = 0.0;
	for (std::vector<Gyro>::iterator it = gyros.begin(); it != gyros.end(); ++it)
	{
		sumX += it->x;
		sumY += it->y;
		sumZ += it->z;
	}
	meanGyro.x = sumX / numGyro;
	meanGyro.y = sumY / numGyro;
	meanGyro.z = sumZ / numGyro;
	varGyro.x = ...;
	varGyro.y = ...;
	varGyro.z = ...;
#else
	meanAccel.x = gsl_stats_mean((double *)&accels[0], 3, numAccel);
	meanAccel.y = gsl_stats_mean((double *)&accels[0] + 1, 3, numAccel);
	meanAccel.z = gsl_stats_mean((double *)&accels[0] + 2, 3, numAccel);
	varAccel.x = gsl_stats_variance((double *)&accels[0], 3, numAccel);
	varAccel.y = gsl_stats_variance((double *)&accels[0] + 1, 3, numAccel);
	varAccel.z = gsl_stats_variance((double *)&accels[0] + 2, 3, numAccel);

	meanGyro.x = gsl_stats_mean((double *)&gyros[0], 3, numGyro);
	meanGyro.y = gsl_stats_mean((double *)&gyros[0] + 1, 3, numGyro);
	meanGyro.z = gsl_stats_mean((double *)&gyros[0] + 2, 3, numGyro);
	varGyro.x = gsl_stats_variance((double *)&gyros[0], 3, numGyro);
	varGyro.y = gsl_stats_variance((double *)&gyros[0] + 1, 3, numGyro);
	varGyro.z = gsl_stats_variance((double *)&gyros[0] + 2, 3, numGyro);
#endif

	stream.close();
}

void load_calibration_param(const std::string &filename, const size_t numAccelParam, const size_t numGyroParam, gsl_vector *accel_param, gsl_matrix *accel_covar, gsl_vector *gyro_param, gsl_matrix *gyro_covar)
{
	std::ifstream stream(filename.c_str());
	if (!stream)
	{
		std::ostringstream stream;
		stream << "file not found at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	std::string line_str;
	double val;

	// load acceleration parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	if (!stream.eof())
	{
		std::getline(stream, line_str);

		std::istringstream sstream(line_str);
		for (size_t i = 0; i < numAccelParam; ++i)
		{
			sstream >> val;
			gsl_vector_set(accel_param, i, val);
		}
	}
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	// load covariance of acceleration parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	if (!stream.eof())
	{
		for (size_t i = 0; i < numAccelParam; ++i)
		{
			std::getline(stream, line_str);

			std::istringstream sstream(line_str);
			for (size_t j = 0; j < numAccelParam; ++j)
			{
				sstream >> val;
				gsl_matrix_set(accel_covar, i, j, val);
			}
		}
	}
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	// load gyro parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	if (!stream.eof())
	{
		std::getline(stream, line_str);

		std::istringstream sstream(line_str);
		for (size_t i = 0; i < numGyroParam; ++i)
		{
			sstream >> val;
			gsl_vector_set(gyro_param, i, val);
		}
	}
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	// load covariance of gyro parameters
	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	if (!stream.eof())
	{
		for (size_t i = 0; i < numGyroParam; ++i)
		{
			std::getline(stream, line_str);

			std::istringstream sstream(line_str);
			for (size_t j = 0; j < numGyroParam; ++j)
			{
				sstream >> val;
				gsl_matrix_set(gyro_covar, i, j, val);
			}
		}
	}
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	if (!stream.eof()) std::getline(stream, line_str);
	else
	{
		std::ostringstream stream;
		stream << "file format error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	stream.close();
}

}  // namespace local
}  // unnamed namespace

// "A new multi-position calibration method for MEMS inertial navigation systems", Z. F. Syed, P. Aggarwal, C. Goodall, X. Niu, and N. El-Sheimy,
//	Measurement Science and Technology, vol. 18, pp. 1897-1907, 2007
// "Accuracy Improvement of Low Cost INS/GPS for Land Applications", Eun-Hwan Shin, UCGE Reports, 2001
void imu_calibration()
{
	const double deg2rad = boost::math::constants::pi<double>() / 180.0;
	const double lambda = 36.368 * deg2rad;  // latitude [rad]
	const double phi = 127.364 * deg2rad;  // longitude [rad]
	const double h = 71.0;  // altitude: 71 ~ 82 [m]
	const double sin_lambda = std::sin(lambda);
	const double sin_2lambda = std::sin(2 * lambda);

	// [ref] wikipedia: Gravity of Earth
	// (latitude, longitude, altitude) = (lambda, phi, h) = (36.368, 127.364, 71.0)
	// g(lambda, h) = 9.780327 * (1 + 0.0053024 * sin(lambda)^2 - 0.0000058 * sin(2 * lambda)^2) - 3.086 * 10^-6 * h
	const double g_true = 9.780327 * (1 + 0.0053024 * sin_lambda*sin_lambda - 0.0000058 * sin_2lambda*sin_2lambda) - 3.086e-6 * h;  // [m/sec^2]

	// [ref] "The Global Positioning System and Inertial Navigation", Jay Farrell & Matthew Barth, pp. 22
	const double w_true = 7.292115e-5;  // [rad/sec]

#if 1
	const size_t numAccelParam = 9;
#else
	const size_t numAccelParam = 6;
#endif
	const size_t numGyroParam = 3;
	const size_t numMeasurement = 14;

	std::vector<local::Acceleration> meanAccels, varAccels;
	std::vector<local::Gyro> meanGyros, varGyros;
	meanAccels.reserve(numMeasurement);
	varAccels.reserve(numMeasurement);
	meanGyros.reserve(numMeasurement);
	varGyros.reserve(numMeasurement);

	local::Acceleration meanAccel(0.0, 0.0, 0.0), varAccel(0.0, 0.0, 0.0);
	local::Gyro meanGyro(0.0, 0.0, 0.0), varGyro(0.0, 0.0, 0.0);
	load_imu_data("..\\data\\adis16350_data_20100801\\01_z_pos.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\02_z_neg.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\03_x_pos.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\04_x_neg.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\05_y_pos.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\06_y_neg.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\07_pos1_1.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\08_pos1_2.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\09_pos1_3.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\10_pos1_4.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\11_pos2_1.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\12_pos2_2.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\13_pos2_3.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);
	load_imu_data("..\\data\\adis16350_data_20100801\\14_pos2_4.csv", g_true, meanAccel, varAccel, meanGyro, varGyro);
	meanAccels.push_back(meanAccel);  varAccels.push_back(varAccel);
	meanGyros.push_back(meanGyro);  varGyros.push_back(varGyro);

	//
	const std::string calibration_filename("..\\data\\adis16350_data_20100801\\imu_calibration_result.txt");
	std::ofstream stream(calibration_filename.c_str());
	if (!stream)
	{
		std::ostringstream stream;
		stream << "file open error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	//
	gsl_vector *accel_x = gsl_vector_alloc(numAccelParam);  // calibration parameters
	gsl_vector *accel_delta_hat = gsl_vector_alloc(numAccelParam);
	gsl_matrix *accel_Cx = gsl_matrix_alloc(numAccelParam, numAccelParam);
	gsl_matrix *accel_Cx_hat = gsl_matrix_alloc(numAccelParam, numAccelParam);
	gsl_vector *gyro_x = gsl_vector_alloc(numGyroParam);  // calibration parameters
	gsl_vector *gyro_delta_hat = gsl_vector_alloc(numGyroParam);
	gsl_matrix *gyro_Cx = gsl_matrix_alloc(numGyroParam, numGyroParam);
	gsl_matrix *gyro_Cx_hat = gsl_matrix_alloc(numGyroParam, numGyroParam);

	// FIXME [modify] >>
	gsl_vector_set_zero(accel_x);
	//for (size_t i = 0; i < numAccelParam; ++i)
	//	gsl_vector_set(accel_x, 0, 0.0);
	gsl_matrix_set_zero(accel_Cx);
	//gsl_matrix_set(accel_Cx, 0, 0, 0.0);
	gsl_matrix_set_identity(accel_Cx);
	gsl_vector_set_zero(gyro_x);
	//for (size_t i = 0; i < numGyroParam; ++i)
	//	gsl_vector_set(gyro_x, 0, 0.0);
	gsl_matrix_set_zero(gyro_Cx);
	//gsl_matrix_set(gyro_Cx, 0, 0, 0.0);
	gsl_matrix_set_identity(gyro_Cx);

	const size_t Niteration = 100;
	for (size_t iter = 0; iter < Niteration; ++iter)
	{
		// calibration parameters of accelerometer
		{
			accelerometer_calibration(numAccelParam, numMeasurement, g_true, accel_x, accel_Cx, meanAccels, varAccels, accel_delta_hat, accel_Cx_hat);

			// update calibration parameters of accelerometer
			for (size_t i = 0; i < accel_x->size; ++i)
				gsl_vector_set(accel_x, i, gsl_vector_get(accel_x, i) + gsl_vector_get(accel_delta_hat, i));
			gsl_matrix_memcpy(accel_Cx, accel_Cx_hat);

			// save calibration parameters of accelerometer
			//for (size_t i = 0; i < accel_x->size; ++i)
			//	stream << gsl_vector_get(accel_x, i) << (i >= accel_x->size - 1 ? "\n" : "  ");

			// use calculate_calibrated_acceleration() to get calibrated acceleration
			//calculate_calibrated_acceleration(accel_x, a_measurement, a_calibrated);
		}

		// calibration parameters of gyro
		{
			gyroscope_calibration(numGyroParam, numMeasurement, w_true, gyro_x, gyro_Cx, meanGyros, varGyros, gyro_delta_hat, gyro_Cx_hat);

			// update calibration parameters of gyro
			for (size_t i = 0; i < gyro_x->size; ++i)
				gsl_vector_set(gyro_x, i, gsl_vector_get(gyro_x, i) + gsl_vector_get(gyro_delta_hat, i));
			gsl_matrix_memcpy(gyro_Cx, gyro_Cx_hat);

			// save calibration parameters of gyro
			//for (size_t i = 0; i < gyro_x->size; ++i)
			//	stream << gsl_vector_get(gyro_x, i) << (i >= gyro_x->size - 1 ? "\n" : "  ");

			// use calculate_calibrated_angular_rate() to get calibrated angular rate
			//calculate_calibrated_angular_rate(gyro_x, w_measurement, w_calibrated);
		}
	}

	// save calibration parameters
	{
		// save calibration parameters of accelerometer
		stream << "acceleration x_hat = [" << std::endl;
		stream << '\t';
		for (size_t i = 0; i < accel_x->size; ++i)
			stream << gsl_vector_get(accel_x, i) << (i >= accel_x->size - 1 ? "\n" : "  ");
		stream << "]" << std::endl;
		stream << "acceleration Cx_hat = [" << std::endl;
		for (size_t i = 0; i < accel_Cx_hat->size1; ++i)
		{
			stream << '\t';
			for (size_t j = 0; j < accel_Cx_hat->size2; ++j)
				stream << gsl_matrix_get(accel_Cx_hat, i, j) << (j >= accel_Cx_hat->size2 - 1 ? "\n" : "  ");
		}
		stream << "]" << std::endl;

		// save calibration parameters of gyro
		stream << "gyro x_hat = [" << std::endl;
		stream << '\t';
		for (size_t i = 0; i < gyro_x->size; ++i)
			stream << gsl_vector_get(gyro_x, i) << (i >= gyro_x->size - 1 ? "\n" : "  ");
		stream << "]" << std::endl;
		stream << "gyro Cx_hat = [" << std::endl;
		for (size_t i = 0; i < gyro_Cx_hat->size1; ++i)
		{
			stream << '\t';
			for (size_t j = 0; j < gyro_Cx_hat->size2; ++j)
				stream << gsl_matrix_get(gyro_Cx_hat, i, j) << (j >= gyro_Cx_hat->size2 - 1 ? "\n" : "  ");
		}
		stream << "]" << std::endl;
	}

	gsl_vector_free(accel_x);
	gsl_vector_free(accel_delta_hat);
	gsl_matrix_free(accel_Cx);
	gsl_matrix_free(accel_Cx_hat);
	gsl_vector_free(gyro_x);
	gsl_vector_free(gyro_delta_hat);
	gsl_matrix_free(gyro_Cx);
	gsl_matrix_free(gyro_Cx_hat);

	stream.close();

	// load calibration parameters
	{
		gsl_vector *accel_param = gsl_vector_alloc(numAccelParam);
		gsl_matrix *accel_covar = gsl_matrix_alloc(numAccelParam, numAccelParam);
		gsl_vector *gyro_param = gsl_vector_alloc(numGyroParam);
		gsl_matrix *gyro_covar = gsl_matrix_alloc(numGyroParam, numGyroParam);

		local::load_calibration_param(calibration_filename, numAccelParam, numGyroParam, accel_param, accel_covar, gyro_param, gyro_covar);

		std::cout << "accelerometer x_hat = ";
		local::print_gsl_vector(accel_param);
		std::cout << "accelerometer Cx_hat = ";
		local::print_gsl_matrix(accel_covar);

		std::cout << "gyro x_hat = ";
		local::print_gsl_vector(gyro_param);
		std::cout << "gyro Cx_hat = ";
		local::print_gsl_matrix(gyro_covar);

		gsl_vector_free(accel_param);
		gsl_matrix_free(accel_covar);
		gsl_vector_free(gyro_param);
		gsl_matrix_free(gyro_covar);
	}
}
