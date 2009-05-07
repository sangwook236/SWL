#include "swl/view/ViewMatrix.h"
#include <memory>
#include <cmath>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class ViewMatrix2

ViewMatrix2::ViewMatrix2()
{
	identity();
}

ViewMatrix2::ViewMatrix2(const value_type mat[6])
{
	memcpy(tmat_, mat, sizeof(value_type) * 6);
}

ViewMatrix2::ViewMatrix2(const value_type x[2], const value_type y[2], const value_type t[2])
{
	tmat_[0] = x[0];  tmat_[1] = y[0];  tmat_[2] = t[0];
	tmat_[3] = x[1];  tmat_[4] = y[1];  tmat_[5] = t[1];
}

ViewMatrix2::ViewMatrix2(const ViewMatrix2& rhs)
{
	memcpy(tmat_, rhs.tmat_, sizeof(value_type) * 6);
}

ViewMatrix2& ViewMatrix2::operator=(const ViewMatrix2& rhs)
{
	if (this == &rhs) return *this;
	memcpy(tmat_, rhs.tmat_, sizeof(value_type) * 6);
	return *this;
}

void ViewMatrix2::set(const value_type mat[6])
{
	memcpy(tmat_, mat, sizeof(value_type) * 6);
}

void ViewMatrix2::get(value_type mat[6]) const
{
	memcpy(mat, tmat_, sizeof(value_type) * 6);
}

void ViewMatrix2::translate(const value_type dx, const value_type dy)
{
	tmat_[2] += dx;
	tmat_[5] += dy;
}

void ViewMatrix2::rotate(const value_type rad)
{
	const value_type s = sin(rad);
	const value_type c = cos(rad);

	const value_type a11 = tmat_[0];
	const value_type a12 = tmat_[1];
	const value_type a13 = tmat_[2];
	const value_type a21 = tmat_[3];
	const value_type a22 = tmat_[4];
	const value_type a23 = tmat_[5];

	tmat_[0] = a11 * c - a21 * s;
	tmat_[1] = a12 * c - a22 * s;
	tmat_[2] = a13 * c - a23 * s;
	tmat_[3] = a11 * s + a21 * c;
	tmat_[4] = a12 * s + a22 * c;
	tmat_[5] = a13 * s + a23 * c;
}

void ViewMatrix2::scale(const value_type sx, const value_type sy)
{
	tmat_[0] *= sx;
	tmat_[1] *= sx;
	tmat_[2] *= sx;
	tmat_[3] *= sy;
	tmat_[4] *= sy;
	tmat_[5] *= sy;
}

void ViewMatrix2::identity()
{
	//tmat_[0] = 1.0f; tmat_[1] = 0.0f; tmat_[2] = 0.0f;
	//tmat_[3] = 0.0f; tmat_[4] = 1.0f; tmat_[5] = 0.0f;

	memset(tmat_, 0, sizeof(value_type) * 6);
	tmat_[0] = tmat_[4] = 1.0f;
}

ViewMatrix2 ViewMatrix2::operator*(const ViewMatrix2& rhs) const
{
	value_type mat[6] = { 0., };

	mat[0] = tmat_[0] * rhs.tmat_[0] + tmat_[1] * rhs.tmat_[3];
	mat[1] = tmat_[0] * rhs.tmat_[1] + tmat_[1] * rhs.tmat_[4];
	mat[2] = tmat_[0] * rhs.tmat_[2] + tmat_[1] * rhs.tmat_[5] + tmat_[2];
	mat[3] = tmat_[3] * rhs.tmat_[0] + tmat_[4] * rhs.tmat_[3];
	mat[4] = tmat_[3] * rhs.tmat_[1] + tmat_[4] * rhs.tmat_[4];
	mat[5] = tmat_[3] * rhs.tmat_[2] + tmat_[4] * rhs.tmat_[5] + tmat_[5];

	return ViewMatrix2(mat);
}

bool ViewMatrix2::transformPoint(const value_type x, const value_type y, value_type& xt, value_type& yt) const
{
	xt = tmat_[0] * x + tmat_[1] * y + tmat_[2];
	yt = tmat_[3] * x + tmat_[4] * y + tmat_[5];

	return true;
}

bool ViewMatrix2::inverseTransformPoint(const value_type x, const value_type y, value_type& xt, value_type& yt, const value_type tol /*= 1.0e-7f*/) const
{
	const value_type det = tmat_[0] * tmat_[4] - tmat_[3] * tmat_[1];
	if (fabs(det) <= tol) return false;

	xt = (tmat_[4] * x - tmat_[1] * y + tmat_[1] * tmat_[5] - tmat_[2] * tmat_[4]) / det;
	yt = (-tmat_[3] * x + tmat_[0] * y - tmat_[0] * tmat_[5] + tmat_[2] * tmat_[3]) / det;

	return true;
}


//--------------------------------------------------------------------------
//  class ViewMatrix3

ViewMatrix3::ViewMatrix3()
{
	identity();
}

ViewMatrix3::ViewMatrix3(const value_type mat[12])
{
	memcpy(tmat_, mat, sizeof(value_type) * 12);
}

ViewMatrix3::ViewMatrix3(const value_type x[3], const value_type y[3], const value_type z[3], const value_type t[3])
{
	tmat_[0] = x[0];  tmat_[1] = y[0];  tmat_[2] = z[0];  tmat_[3] = t[0];
	tmat_[4] = x[1];  tmat_[5] = y[1];  tmat_[6] = z[1];  tmat_[7] = t[1];
	tmat_[8] = x[2];  tmat_[9] = y[2];  tmat_[10] = z[2];  tmat_[11] = t[2];
}

ViewMatrix3::ViewMatrix3(const ViewMatrix3& rhs)
{
	memcpy(tmat_, rhs.tmat_, sizeof(value_type) * 12);
}

ViewMatrix3& ViewMatrix3::operator=(const ViewMatrix3& rhs)
{
	if (this == &rhs) return *this;
	memcpy(tmat_, rhs.tmat_, sizeof(value_type) * 12);
	return *this;
}

void ViewMatrix3::set(const value_type mat[12])
{
	memcpy(tmat_, mat, sizeof(value_type) * 12);
}

void ViewMatrix3::get(value_type mat[12]) const
{
	memcpy(mat, tmat_, sizeof(value_type) * 12);
}

void ViewMatrix3::translate(const value_type dx, const value_type dy, const value_type dz)
{
	tmat_[3] += dx;
	tmat_[7] += dy;
	tmat_[11] += dz;
}

void ViewMatrix3::rotateX(const value_type rad)
{
	const value_type s = sin(rad);
	const value_type c = cos(rad);

	const value_type a21 = tmat_[4];
	const value_type a22 = tmat_[5];
	const value_type a23 = tmat_[6];
	const value_type t2 = tmat_[7];
	const value_type a31 = tmat_[8];
	const value_type a32 = tmat_[9];
	const value_type a33 = tmat_[10];
	const value_type t3 = tmat_[11];

	tmat_[4] = a21 * c - a31 * s;
	tmat_[5] = a22 * c - a32 * s;
	tmat_[6] = a23 * c - a33 * s;
	tmat_[7] = t2 * c - t3 * s;
	tmat_[8] = a21 * s + a31 * c;
	tmat_[9] = a22 * s + a32 * c;
	tmat_[10] = a23 * s + a33 * c;
	tmat_[11] = t2 * s + t3 * c;
}

void ViewMatrix3::rotateY(const value_type rad)
{
	const value_type s = sin(rad);
	const value_type c = cos(rad);

	const value_type a11 = tmat_[0];
	const value_type a12 = tmat_[1];
	const value_type a13 = tmat_[2];
	const value_type t1 = tmat_[3];
	const value_type a31 = tmat_[8];
	const value_type a32 = tmat_[9];
	const value_type a33 = tmat_[10];
	const value_type t3 = tmat_[11];

	tmat_[0] = a11 * c + a31 * s;
	tmat_[1] = a12 * c + a32 * s;
	tmat_[2] = a13 * c + a33 * s;
	tmat_[3] = t1 * c + t3 * s;
	tmat_[8] = -a11 * s + a31 * c;
	tmat_[9] = -a12 * s + a32 * c;
	tmat_[10] = -a13 * s + a33 * c;
	tmat_[11] = -t1 * s + t3 * c;
}

void ViewMatrix3::rotateZ(const value_type rad)
{
	const value_type s = sin(rad);
	const value_type c = cos(rad);

	const value_type a11 = tmat_[0];
	const value_type a12 = tmat_[1];
	const value_type a13 = tmat_[2];
	const value_type t1 = tmat_[3];
	const value_type a21 = tmat_[4];
	const value_type a22 = tmat_[5];
	const value_type a23 = tmat_[6];
	const value_type t2 = tmat_[7];

	tmat_[0] = a11 * c - a21 * s;
	tmat_[1] = a12 * c - a22 * s;
	tmat_[2] = a13 * c - a23 * s;
	tmat_[3] = t1 * c - t2 * s;
	tmat_[4] = a11 * s + a21 * c;
	tmat_[5] = a12 * s + a22 * c;
	tmat_[6] = a13 * s + a23 * c;
	tmat_[7] = t1 * s + t2 * c;
}

void ViewMatrix3::scale(const value_type sx, const value_type sy, const value_type sz)
{
	tmat_[0] *= sx;
	tmat_[1] *= sx;
	tmat_[2] *= sx;
	tmat_[3] *= sx;
	tmat_[4] *= sy;
	tmat_[5] *= sy;
	tmat_[6] *= sy;
	tmat_[7] *= sy;
	tmat_[8] *= sz;
	tmat_[9] *= sz;
	tmat_[10] *= sz;
	tmat_[11] *= sz;
}

void ViewMatrix3::identity()
{
	//tmat_[0] = 1.0f; tmat_[1] = 0.0f; tmat_[2] = 0.0f; tmat_[3] = 0.0f;
	//tmat_[4] = 0.0f; tmat_[5] = 1.0f; tmat_[6] = 0.0f; tmat_[7] = 0.0f;
	//tmat_[8] = 0.0f; tmat_[9] = 0.0f; tmat_[10] = 1.0f; tmat_[11] = 0.0f;

	memset(tmat_, 0, sizeof(value_type) * 12);
	tmat_[0] = tmat_[5] = tmat_[10] = 1.0f;
}

ViewMatrix3 ViewMatrix3::operator*(const ViewMatrix3& rhs) const
{
	value_type mat[12] = { 0., };

	mat[0] = tmat_[0] * rhs.tmat_[0] + tmat_[1] * rhs.tmat_[4] + tmat_[2] * rhs.tmat_[8];
	mat[1] = tmat_[0] * rhs.tmat_[1] + tmat_[1] * rhs.tmat_[5] + tmat_[2] * rhs.tmat_[9];
	mat[3] = tmat_[0] * rhs.tmat_[2] + tmat_[1] * rhs.tmat_[6] + tmat_[2] * rhs.tmat_[10];
	mat[4] = tmat_[0] * rhs.tmat_[3] + tmat_[1] * rhs.tmat_[7] + tmat_[2] * rhs.tmat_[11] + tmat_[3];

	mat[5] = tmat_[4] * rhs.tmat_[0] + tmat_[5] * rhs.tmat_[4] + tmat_[6] * rhs.tmat_[8];
	mat[6] = tmat_[4] * rhs.tmat_[1] + tmat_[5] * rhs.tmat_[5] + tmat_[6] * rhs.tmat_[9];
	mat[7] = tmat_[4] * rhs.tmat_[2] + tmat_[5] * rhs.tmat_[6] + tmat_[6] * rhs.tmat_[10];
	mat[8] = tmat_[4] * rhs.tmat_[3] + tmat_[5] * rhs.tmat_[7] + tmat_[6] * rhs.tmat_[11] + tmat_[7];

	mat[9] = tmat_[8] * rhs.tmat_[0] + tmat_[9] * rhs.tmat_[4] + tmat_[10] * rhs.tmat_[8];
	mat[10] = tmat_[8] * rhs.tmat_[1] + tmat_[9] * rhs.tmat_[5] + tmat_[10] * rhs.tmat_[9];
	mat[11] = tmat_[8] * rhs.tmat_[2] + tmat_[9] * rhs.tmat_[6] + tmat_[10] * rhs.tmat_[10];
	mat[12] = tmat_[8] * rhs.tmat_[3] + tmat_[9] * rhs.tmat_[7] + tmat_[10] * rhs.tmat_[11] + tmat_[11];

	return ViewMatrix3(mat);
}

bool ViewMatrix3::transformPoint(const value_type x, const value_type y, const value_type z, value_type& xt, value_type& yt, value_type& zt) const
{
	xt = tmat_[0] * x + tmat_[1] * y + tmat_[2] * z + tmat_[3];
	yt = tmat_[4] * x + tmat_[5] * y + tmat_[6] * z + tmat_[7];
	zt = tmat_[8] * x + tmat_[9] * y + tmat_[10] * z + tmat_[11];

	return true;
}

bool ViewMatrix3::inverseTransformPoint(const value_type x, const value_type y, const value_type z, value_type& xt, value_type& yt, value_type& zt, const value_type tol /*= 1.0e-7f*/) const
{
	const value_type& a11 = tmat_[0];
	const value_type& a12 = tmat_[1];
	const value_type& a13 = tmat_[2];
	const value_type& t1 = tmat_[3];
	const value_type& a21 = tmat_[4];
	const value_type& a22 = tmat_[5];
	const value_type& a23 = tmat_[6];
	const value_type& t2 = tmat_[7];
	const value_type& a31 = tmat_[8];
	const value_type& a32 = tmat_[9];
	const value_type& a33 = tmat_[10];
	const value_type& t3 = tmat_[11];

	const value_type det = a11*a22*a33 - a11*a32*a23 - a21*a12*a33 + a21*a32*a13 + a31*a12*a23 - a31*a22*a13;
	if (fabs(det) <= tol) return false;

	xt = ((a22*a33-a32*a23) * x + (-a12*a33+a32*a13) * y + (a12*a23-a22*a13) * z + (-a12*a23*t3 + a12*a33*t2 + a22*a13*t3 - a22*a33*t1 - a32*a13*t2 + a32*a23*t1)) / det;
	yt = ((-a21*a33+a31*a23) * x + (a11*a33-a31*a13) * y + (-a11*a23+a21*a13) * z + (a11*a23*t3 - a11*a33*t2 - a21*a13*t3 + a21*a33*t1 + a31*a13*t2 - a31*a23*t1)) / det;
	zt = ((a21*a32-a31*a22) * x + (-a11*a32+a31*a12) * y + (a11*a22-a21*a12) * z + (-a11*a22*t3 + a11*a32*t2 + a21*a12*t3 - a21*a32*t1 - a31*a12*t2 + a31*a22*t1)) / det;

	return true;
}

}  // namespace swl
