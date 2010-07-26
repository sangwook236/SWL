#include "stdafx.h"
#include "swl/Config.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <cmath>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

void matrix_inverse(const size_t dim, gsl_matrix *A, gsl_matrix *invA)
{
	gsl_permutation *p = gsl_permutation_alloc(dim);
	int signum;

	// LU decomposition
	gsl_linalg_LU_decomp(A, p, &signum);
	gsl_linalg_LU_invert(A, p, invA);

	gsl_permutation_free(p);
}

void evaluate_Fg(const gsl_vector *x, const gsl_vector *lg, const double g_true, double &Fg, gsl_vector *dFgdx, gsl_vector *dFgdl)
{
	const double &b_gx = gsl_vector_get(x, 0);
	const double &b_gy = gsl_vector_get(x, 1);
	const double &b_gz = gsl_vector_get(x, 2);
	const double &s_gx = gsl_vector_get(x, 3);
	const double &s_gy = gsl_vector_get(x, 4);
	const double &s_gz = gsl_vector_get(x, 5);
	const double &theta_gyz = gsl_vector_get(x, 6);
	const double &theta_gzx = gsl_vector_get(x, 7);
	const double &theta_gzy = gsl_vector_get(x, 8);

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
	const double num1 = g_x + g_y * tan_gyz + g_z * (tan_gzx * tan_gzy - tan_gzy / cos_gzx);
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
	const double dFgdLgx = 2.0 * (g_x + g_y * tan_gyz + g_z * (tan_gzx * tan_gyz - tan_gzy / cos_gzx)) / (1.0 + s_gx);
	const double dFgdLgy = 2.0 * (g_y + g_z * tan_gzx) / (cos_gyz * (1.0 + s_gy));
	const double dFgdLgz = 2.0 * g_z / (cos_gzx * cos_gzy * (1.0 + s_gz));

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

void evaluate_Fw(const gsl_vector *x, const gsl_vector *lw, const double w_true, double &Fw, gsl_vector *dFwdx, gsl_vector *dFwdl)
{
	const double &b_wx = gsl_vector_get(x, 0);
	const double &b_wy = gsl_vector_get(x, 1);
	const double &b_wz = gsl_vector_get(x, 2);

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

void accelerometer_calibration()
{
#if 1
	const size_t Nv = 9;
	const size_t Nm = 9;
#else
	const size_t Nv = 6;
	const size_t Nm = 6;
#endif

	const double eps = 1.0e-15;

	gsl_vector *x = gsl_vector_alloc(Nv);
	gsl_vector *lg = gsl_vector_alloc(Nm * 3);
	// FIXME [modify] >>
	const double g_true = 9.81;
	double Fg = 0.0;
	gsl_vector *dFgdx = gsl_vector_alloc(Nv);
	gsl_vector *dFgdl = gsl_vector_alloc(Nm * 3);

	// FIXME [modify] >>
	const double sigma_lgx2 = 1.0;
	const double sigma_lgy2 = 1.0;
	const double sigma_lgz2 = 1.0;

	for (size_t i = 0; i < Nm; ++i)
	{
		// FIXME [modify] >>
		gsl_vector_set(x, 0, 0.0);
		gsl_vector_set(lg, 0, 0.0);
	}

	//
	gsl_matrix *A = gsl_matrix_alloc(Nm, Nv);
	gsl_vector *M = gsl_vector_alloc(Nm);
	gsl_vector *w = gsl_vector_alloc(Nm);

	for (size_t i = 0; i < Nm; ++i)
	{
		evaluate_Fw(x, lg, g_true, Fg, dFgdx, dFgdl);

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

	gsl_vector_free(x);
	gsl_vector_free(lg);
	gsl_vector_free(dFgdx);
	gsl_vector_free(dFgdl);

	// inverse of Cx
	gsl_matrix *invCx = gsl_matrix_alloc(Nv, Nv);
	{
		gsl_matrix *Cx = gsl_matrix_alloc(Nv, Nv);

		// FIXME [modify] >>
		gsl_matrix_set_zero(Cx);
		gsl_matrix_set(Cx, 0, 0, 0.0);

		matrix_inverse(Nv, Cx, invCx);

		gsl_matrix_free(Cx);
	}

	//
	gsl_matrix *N = gsl_matrix_alloc(Nv, Nv);
	gsl_vector *u = gsl_vector_alloc(Nv);

	for (size_t i = 0; i < Nv; ++i)
	{
		for (size_t j = 0; j < Nv; ++j)
		{
			double sum = 0.0;
			for (size_t k = 0; k < Nm; ++k)
				sum += gsl_matrix_get(A, i, k) * gsl_matrix_get(A, j, k) / gsl_vector_get(M, k);
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
	gsl_matrix *Cx_hat = gsl_matrix_alloc(Nv, Nv);
	matrix_inverse(Nv, N, Cx_hat);

	gsl_vector *delta_hat = gsl_vector_alloc(Nv);
	gsl_blas_dgemv(CblasNoTrans, -1.0, Cx_hat, u, 0.0, delta_hat);

	gsl_matrix_free(N);
	gsl_vector_free(u);

	gsl_matrix_free(Cx_hat);
	gsl_vector_free(delta_hat);
}

void gyroscope_calibration()
{
	const size_t Nv = 3;
	const size_t Nm = 3;

	const double eps = 1.0e-15;

	gsl_vector *x = gsl_vector_alloc(Nv);
	gsl_vector *lw = gsl_vector_alloc(Nm * 3);
	// FIXME [modify] >>
	const double w_true = 0.0;
	double Fw = 0.0;
	gsl_vector *dFwdx = gsl_vector_alloc(Nv);
	gsl_vector *dFwdl = gsl_vector_alloc(Nm * 3);

	// FIXME [modify] >>
	const double sigma_lwx2 = 1.0;
	const double sigma_lwy2 = 1.0;
	const double sigma_lwz2 = 1.0;

	for (size_t i = 0; i < Nm; ++i)
	{
		// FIXME [modify] >>
		gsl_vector_set(x, 0, 0.0);
		gsl_vector_set(lw, 0, 0.0);
	}

	//
	gsl_matrix *A = gsl_matrix_alloc(Nv, Nv);
	gsl_vector *M = gsl_vector_alloc(Nm);
	gsl_vector *w = gsl_vector_alloc(Nm);

	for (size_t i = 0; i < Nm; ++i)
	{
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

	gsl_vector_free(x);
	gsl_vector_free(lw);
	gsl_vector_free(dFwdx);
	gsl_vector_free(dFwdl);

	// inverse of Cx
	gsl_matrix *invCx = gsl_matrix_alloc(Nv, Nv);
	{
		gsl_matrix *Cx = gsl_matrix_alloc(Nv, Nv);

		// FIXME [modify] >>
		gsl_matrix_set_zero(Cx);
		gsl_matrix_set(Cx, 0, 0, 0.0);

		matrix_inverse(Nv, Cx, invCx);

		gsl_matrix_free(Cx);
	}

	//
	gsl_matrix *N = gsl_matrix_alloc(Nv, Nv);
	gsl_vector *u = gsl_vector_alloc(Nv);

	for (size_t i = 0; i < Nv; ++i)
	{
		for (size_t j = 0; j < Nv; ++j)
		{
			double sum = 0.0;
			for (size_t k = 0; k < Nm; ++k)
				sum += gsl_matrix_get(A, i, k) * gsl_matrix_get(A, j, k) / gsl_vector_get(M, k);
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
	gsl_matrix *Cx_hat = gsl_matrix_alloc(Nv, Nv);
	matrix_inverse(Nv, N, Cx_hat);

	gsl_vector *delta_hat = gsl_vector_alloc(Nv);
	gsl_blas_dgemv(CblasNoTrans, -1.0, Cx_hat, u, 0.0, delta_hat);

	gsl_matrix_free(N);
	gsl_vector_free(u);

	gsl_matrix_free(Cx_hat);
	gsl_vector_free(delta_hat);
}

}  // unnamed namespace

void imu_calibration()
{
	accelerometer_calibration();
	gyroscope_calibration();
}
