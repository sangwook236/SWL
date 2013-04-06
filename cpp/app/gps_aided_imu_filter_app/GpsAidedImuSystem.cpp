#include "stdafx.h"
#include "swl/Config.h"
#include "GpsAidedImuSystem.h"
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
	
GpsAidedImuSystem::GpsAidedImuSystem(const double Ts, const size_t stateDim, const size_t inputDim, const size_t outputDim, const size_t processNoiseDim, const size_t observationNoiseDim, const ImuData::Accel &initial_gravity, const ImuData::Gyro &initial_angular_velocity_of_the_earth)
: base_type(stateDim, inputDim, outputDim, processNoiseDim, observationNoiseDim),
  Ts_(Ts), f_eval_(NULL), h_eval_(NULL), initial_gravity_(initial_gravity), initial_angular_velocity_of_the_earth_(initial_angular_velocity_of_the_earth),
  p_k_N_(NULL), v_k_N_(NULL), w_k_N_(NULL), r_GPS_(NULL),
  measuredAccel_(0.0, 0.0, 0.0), measuredAngularVel_(0.0, 0.0, 0.0)
{
	//
	f_eval_ = gsl_vector_alloc(stateDim_);
	gsl_vector_set_zero(f_eval_);

	//
	h_eval_ = gsl_vector_alloc(outputDim_);
	gsl_vector_set_zero(h_eval_);

	// the time-delayed (by N samples due to sensor latency) 3D navigation-frame position vector of the vehicle
	p_k_N_ = gsl_vector_alloc(3);
	gsl_vector_set_zero(p_k_N_);
	// the time-delayed (by N samples due to sensor latency) 3D navigation-frame velocity vector of the vehicle
	v_k_N_ = gsl_vector_alloc(3);
	gsl_vector_set_zero(v_k_N_);
	// the time-delayed (by N samples due to sensor latency) 3D body-frame angular velocity vector of the vehicle
	w_k_N_ = gsl_vector_alloc(3);
	gsl_vector_set_zero(w_k_N_);

	// the location of the GPS antenna in the body frame relative to the IMU location
	// FIXME [modify] >>
	r_GPS_ = gsl_vector_alloc(3);
	gsl_vector_set_zero(r_GPS_);
}

GpsAidedImuSystem::~GpsAidedImuSystem()
{
	gsl_vector_free(f_eval_);  f_eval_ = NULL;
	gsl_vector_free(h_eval_);  h_eval_ = NULL;

	gsl_vector_free(p_k_N_);  p_k_N_ = NULL;
	gsl_vector_free(v_k_N_);  v_k_N_ = NULL;
	gsl_vector_free(w_k_N_);  w_k_N_ = NULL;

	gsl_vector_free(r_GPS_);  r_GPS_ = NULL;
}

gsl_vector * GpsAidedImuSystem::evaluatePlantEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const
{
	// update of true state w/o noise
	const double &Px = gsl_vector_get(state, 0);
	const double &Py = gsl_vector_get(state, 1);
	const double &Pz = gsl_vector_get(state, 2);
	const double &Vx = gsl_vector_get(state, 3);
	const double &Vy = gsl_vector_get(state, 4);
	const double &Vz = gsl_vector_get(state, 5);
	const double &E0 = gsl_vector_get(state, 6);
	const double &E1 = gsl_vector_get(state, 7);
	const double &E2 = gsl_vector_get(state, 8);
	const double &E3 = gsl_vector_get(state, 9);
	const double &Abp = gsl_vector_get(state, 10);
	const double &Abq = gsl_vector_get(state, 11);
	const double &Abr = gsl_vector_get(state, 12);
	const double &Wbp = gsl_vector_get(state, 13);
	const double &Wbq = gsl_vector_get(state, 14);
	const double &Wbr = gsl_vector_get(state, 15);

	const double &w0 = noise ? gsl_vector_get(noise, 0) : 0.0;
	const double &w1 = noise ? gsl_vector_get(noise, 1) : 0.0;
	const double &w2 = noise ? gsl_vector_get(noise, 2) : 0.0;
	const double &w3 = noise ? gsl_vector_get(noise, 3) : 0.0;
	const double &w4 = noise ? gsl_vector_get(noise, 4) : 0.0;
	const double &w5 = noise ? gsl_vector_get(noise, 5) : 0.0;
	const double &w6 = noise ? gsl_vector_get(noise, 6) : 0.0;
	const double &w7 = noise ? gsl_vector_get(noise, 7) : 0.0;
	const double &w8 = noise ? gsl_vector_get(noise, 8) : 0.0;
	const double &w9 = noise ? gsl_vector_get(noise, 9) : 0.0;
	const double &w10 = noise ? gsl_vector_get(noise, 10) : 0.0;
	const double &w11 = noise ? gsl_vector_get(noise, 11) : 0.0;
	const double &w12 = noise ? gsl_vector_get(noise, 12) : 0.0;
	const double &w13 = noise ? gsl_vector_get(noise, 13) : 0.0;
	const double &w14 = noise ? gsl_vector_get(noise, 14) : 0.0;
	const double &w15 = noise ? gsl_vector_get(noise, 15) : 0.0;

	//
	const double Ap_measured = measuredAccel_.x;
	const double Aq_measured = measuredAccel_.y;
	const double Ar_measured = measuredAccel_.z;
	const double Wp_measured = measuredAngularVel_.x;
	const double Wq_measured = measuredAngularVel_.y;
	const double Wr_measured = measuredAngularVel_.z;

	const double Ap = Ap_measured - Abp;  // Ap_measured - Abp - Nap
	const double Aq = Aq_measured - Abq;  // Aq_measured - Abq - Naq
	const double Ar = Ar_measured - Abr;  // Ar_measured - Abr - Nar
#if 0
	// compensate the local gravity
	const double &g_ix = initial_gravity_.x;
	const double &g_iy = initial_gravity_.y;
	const double &g_iz = initial_gravity_.z;

	const double dvdt_x = 2.0 * ((0.5 - E2*E2 - E3*E3)*Ap + (E1*E2 - E0*E3)*Aq + (E1*E3 + E0*E2)*Ar) + g_ix;
	const double dvdt_y = 2.0 * ((E1*E2 + E0*E3)*Ap + (0.5 - E1*E1 - E3*E3)*Aq + (E2*E3 - E0*E1)*Ar) + g_iy;
	const double dvdt_z = 2.0 * ((E1*E3 - E0*E2)*Ap + (E2*E3 + E0*E1)*Aq + (0.5 - E1*E1 - E2*E2)*Ar) + g_iz;
#else
	// compensate the local gravity in acceleration's measurements
	const double dvdt_x = 2.0 * ((0.5 - E2*E2 - E3*E3)*Ap + (E1*E2 - E0*E3)*Aq + (E1*E3 + E0*E2)*Ar);
	const double dvdt_y = 2.0 * ((E1*E2 + E0*E3)*Ap + (0.5 - E1*E1 - E3*E3)*Aq + (E2*E3 - E0*E1)*Ar);
	const double dvdt_z = 2.0 * ((E1*E3 - E0*E2)*Ap + (E2*E3 + E0*E1)*Aq + (0.5 - E1*E1 - E2*E2)*Ar);
#endif

#if 0
	// compensate the initial angular velocity of the Earth
	const double &Wc_ix = initial_angular_velocity_of_the_earth_.x;
	const double &Wc_iy = initial_angular_velocity_of_the_earth_.y;
	const double &Wc_iz = initial_angular_velocity_of_the_earth_.z;
	const double Wc_ip = 2.0 * ((0.5 - E2*E2 - E3*E3)*Wc_ix + (E1*E2 + E0*E3)*Wc_iy + (E1*E3 - E0*E2)*Wc_iz);
	const double Wc_iq = 2.0 * ((E1*E2 - E0*E3)*Wc_ix + (0.5 - E1*E1 - E3*E3)*Wc_iy + (E2*E3 + E0*E1)*Wc_iz);
	const double Wc_ir = 2.0 * ((E1*E3 + E0*E2)*Wc_ix + (E2*E3 - E0*E1)*Wc_iy + (0.5 - E1*E1 - E2*E2)*Wc_iz);

	// FIXME [check] >> 
	const double Wp = Wp_measured - Wbp - Wc_ip;  // Wp_measured - Wbp - Wc_ip - Nwp
	const double Wq = Wq_measured - Wbq - Wc_iq;  // Wq_measured - Wbq - Wc_iq - Nwq
	const double Wr = Wr_measured - Wbr - Wc_ir;  // Wr_measured - Wbr - Wc_ir - Nwr
#else
	// compensate the initial angular velocity of the Earth in angular velocity's measurements
	// FIXME [check] >> 
	const double Wp = Wp_measured - Wbp;  // Wp_measured - Wbp - Nwp
	const double Wq = Wq_measured - Wbq;  // Wq_measured - Wbq - Nwq
	const double Wr = Wr_measured - Wbr;  // Wr_measured - Wbr - Nwr
#endif

	const double dPhi = Wp * Ts_;
	const double dTheta = Wq * Ts_;
	const double dPsi = Wr * Ts_;
	const double s = 0.5 * std::sqrt(dPhi*dPhi + dTheta*dTheta + dPsi*dPsi);
	const double e_norm = std::sqrt(E0*E0 + E1*E1 + E2*E2 + E3*E3);
	const double lambda = 1.0 - e_norm*e_norm;
	// TODO [check] >>
	const double eta_dt = 0.95;  // eta * dt < 1.0

	const double eps = 1.0e-10;
	const double coeff1 = std::cos(s) + eta_dt * lambda;
	const double coeff2 = std::fabs(s) <= eps ? 0.0 : (0.5 * std::sin(s) / s);

	const double f0 = Px + Vx * Ts_;
	const double f1 = Py + Vy * Ts_;
	const double f2 = Pz + Vz * Ts_;
	const double f3 = Vx + dvdt_x * Ts_;
	const double f4 = Vy + dvdt_y * Ts_;
	const double f5 = Vz + dvdt_z * Ts_;
	const double f6 = coeff1 * E0 - coeff2 * (dPhi*E1 + dTheta*E2 + dPsi*E3);
	const double f7 = coeff1 * E1 - coeff2 * (-dPhi*E0 - dPsi*E2 + dTheta*E3);
	const double f8 = coeff1 * E2 - coeff2 * (-dTheta*E0 + dPsi*E1 - dPhi*E3);
	const double f9 = coeff1 * E3 - coeff2 * (-dPsi*E0 - dTheta*E1 + dPhi*E2);
	const double f10 = Abp + w10 * Ts_;
	const double f11 = Abq + w11 * Ts_;
	const double f12 = Abr + w12 * Ts_;
	const double f13 = Wbp + w13 * Ts_;
	const double f14 = Wbq + w14 * Ts_;
	const double f15 = Wbr + w15 * Ts_;

	gsl_vector_set(f_eval_, 0, f0);
	gsl_vector_set(f_eval_, 1, f1);
	gsl_vector_set(f_eval_, 2, f2);
	gsl_vector_set(f_eval_, 3, f3);
	gsl_vector_set(f_eval_, 4, f4);
	gsl_vector_set(f_eval_, 5, f5);
	gsl_vector_set(f_eval_, 6, f6);
	gsl_vector_set(f_eval_, 7, f7);
	gsl_vector_set(f_eval_, 8, f8);
	gsl_vector_set(f_eval_, 9, f9);
	gsl_vector_set(f_eval_, 10, f10);
	gsl_vector_set(f_eval_, 11, f11);
	gsl_vector_set(f_eval_, 12, f12);
	gsl_vector_set(f_eval_, 13, f13);
	gsl_vector_set(f_eval_, 14, f14);
	gsl_vector_set(f_eval_, 15, f15);

	return f_eval_;
}

gsl_vector * GpsAidedImuSystem::evaluateMeasurementEquation(const size_t step, const gsl_vector *state, const gsl_vector *input, const gsl_vector *noise) const 
{
	const double &Px = gsl_vector_get(state, 0);
	const double &Py = gsl_vector_get(state, 1);
	const double &Pz = gsl_vector_get(state, 2);
	const double &Vx = gsl_vector_get(state, 3);
	const double &Vy = gsl_vector_get(state, 4);
	const double &Vz = gsl_vector_get(state, 5);
	const double &E0 = gsl_vector_get(state, 6);
	const double &E1 = gsl_vector_get(state, 7);
	const double &E2 = gsl_vector_get(state, 8);
	const double &E3 = gsl_vector_get(state, 9);
	const double &Abp = gsl_vector_get(state, 10);
	const double &Abq = gsl_vector_get(state, 11);
	const double &Abr = gsl_vector_get(state, 12);
	const double &Wbp = gsl_vector_get(state, 13);
	const double &Wbq = gsl_vector_get(state, 14);
	const double &Wbr = gsl_vector_get(state, 15);

	const double &v0 = noise ? gsl_vector_get(noise, 0) : 0.0;
	const double &v1 = noise ? gsl_vector_get(noise, 1) : 0.0;
	const double &v2 = noise ? gsl_vector_get(noise, 2) : 0.0;
	const double &v3 = noise ? gsl_vector_get(noise, 3) : 0.0;
	const double &v4 = noise ? gsl_vector_get(noise, 4) : 0.0;
	const double &v5 = noise ? gsl_vector_get(noise, 5) : 0.0;

#if 0
	// FIXME [add] >> save current states: p_k, v_k, w_k

	// use time-delayed data by N samples due to sensor latency
	const double &p_k_N_x = gsl_vector_get(p_k_N_, 0);
	const double &p_k_N_y = gsl_vector_get(p_k_N_, 1);
	const double &p_k_N_z = gsl_vector_get(p_k_N_, 2);
	const double &v_k_N_x = gsl_vector_get(v_k_N_, 0);
	const double &v_k_N_y = gsl_vector_get(v_k_N_, 1);
	const double &v_k_N_z = gsl_vector_get(v_k_N_, 2);
	const double &w_k_N_p = gsl_vector_get(w_k_N_, 0);
	const double &w_k_N_q = gsl_vector_get(w_k_N_, 1);
	const double &w_k_N_r = gsl_vector_get(w_k_N_, 2);
#else
	// use current states ==> ignore sensor latency
	const double &p_k_N_x = Px;
	const double &p_k_N_y = Py;
	const double &p_k_N_z = Pz;
	const double &v_k_N_x = Vx;
	const double &v_k_N_y = Vy;
	const double &v_k_N_z = Vz;
	// FIXME [modify] >> now assume no rotational motion
	const double &w_k_N_p = 0.0;
	const double &w_k_N_q = 0.0;
	const double &w_k_N_r = 0.0;
#endif

	const double &r_GPS_p = gsl_vector_get(r_GPS_, 0);
	const double &r_GPS_q = gsl_vector_get(r_GPS_, 1);
	const double &r_GPS_r = gsl_vector_get(r_GPS_, 2);
	const double r_GPS_x = 2.0 * ((0.5 - E2*E2 - E3*E3)*r_GPS_p + (E1*E2 - E0*E3)*r_GPS_q + (E1*E3 + E0*E2)*r_GPS_r);
	const double r_GPS_y = 2.0 * ((E1*E2 + E0*E3)*r_GPS_p + (0.5 - E1*E1 - E3*E3)*r_GPS_q + (E2*E3 - E0*E1)*r_GPS_r);
	const double r_GPS_z = 2.0 * ((E1*E3 - E0*E2)*r_GPS_p + (E2*E3 + E0*E1)*r_GPS_q + (0.5 - E1*E1 - E2*E2)*r_GPS_r);

	// v_GPS = w_GPS x r_GPS
	const double v_GPS_p = w_k_N_q * r_GPS_r - w_k_N_r * r_GPS_q;
	const double v_GPS_q = w_k_N_r * r_GPS_p - w_k_N_p * r_GPS_r;
	const double v_GPS_r = w_k_N_p * r_GPS_q - w_k_N_q * r_GPS_p;
	const double v_GPS_x = 2.0 * ((0.5 - E2*E2 - E3*E3)*v_GPS_p + (E1*E2 - E0*E3)*v_GPS_q + (E1*E3 + E0*E2)*v_GPS_r);
	const double v_GPS_y = 2.0 * ((E1*E2 + E0*E3)*v_GPS_p + (0.5 - E1*E1 - E3*E3)*v_GPS_q + (E2*E3 - E0*E1)*v_GPS_r);
	const double v_GPS_z = 2.0 * ((E1*E3 - E0*E2)*v_GPS_p + (E2*E3 + E0*E1)*v_GPS_q + (0.5 - E1*E1 - E2*E2)*v_GPS_r);

	gsl_vector_set(h_eval_, 0, p_k_N_x + r_GPS_x + v0);
	gsl_vector_set(h_eval_, 1, p_k_N_y + r_GPS_y + v1);
	gsl_vector_set(h_eval_, 2, p_k_N_z + r_GPS_z + v2);
	gsl_vector_set(h_eval_, 3, v_k_N_x + v_GPS_x + v3);
	gsl_vector_set(h_eval_, 4, v_k_N_y + v_GPS_y + v4);
	gsl_vector_set(h_eval_, 5, v_k_N_z + v_GPS_z + v5);

	return h_eval_;
}

}  // namespace swl
