#include "swl/oglview/OglCamera.h"
#include <GL/glut.h>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class OglCamera

OglCamera::OglCamera()
: base_type()
{}

OglCamera::OglCamera(const OglCamera& rhs)
: base_type(rhs)
{}

OglCamera::~OglCamera()  {}

OglCamera& OglCamera::operator=(const OglCamera& rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type&>(*this) = rhs;
	return *this;
}

/*
void OglCamera::write(std::ostream& stream)
{
	beginWrite(stream);
		// write version
		stream << ' ' << OglCamera::getVersion();

		// write base class
		OglCamera::base_type::write(stream);

		// write self
	endWriteEndl(stream);
}

void OglCamera::read(std::istream& stream)
{
	beginAssert(stream);
		// read version
		unsigned int iVersion;
		stream >> iVersion;

		// read base class
		OglCamera::base_type::read(stream);

		// read self
		unsigned int iOldVersion = setReadVersion(iVersion);
		switch (iVersion)
		{
		case 20021008:
			read20021008(stream);
			break;
		default:
			UtBsErrorHandler::throwFileReadError(ClassName(), iVersion);
		}
		setReadVersion(iOldVersion);
	endAssert(stream);
}

void OglCamera::read20021008(std::istream& stream)
{
}
*/

bool OglCamera::doUpdateFrustum()
{
	int iOldMatrixMode;
	::glGetIntegerv(GL_MATRIX_MODE, &iOldMatrixMode);
	if (iOldMatrixMode != GL_PROJECTION) ::glMatrixMode(GL_PROJECTION);
	::glLoadIdentity();

	const Region2<double> rctViewRegion = getRevisedRegion();

	if (isPerspective_)
	{
		// object: 설정된 view region이 near clipping plane상의 영역이 아닌
		//         reference object position을 지나는 평면상의 영역을 의미하도록 하기 위해
		// merit: perspective & orthographic projection의 전환시 화면에 그려지는 image의 size 변화가 적다
		//        view bound의 설정이 쉽다
		// demerit: eyeDistance_에 의해 near clipping plane의 size가 변경되므로
		//          object coordinates <==> window coordinates 으로의 좌표변경시 이들의 영향을 고려하여야 한다
		if (nearPlane_ > 0.0 && farPlane_ > 0.0 && nearPlane_ < farPlane_)
		{
			double dRatio = calcResizingRatio();
			::glFrustum(
				rctViewRegion.left * dRatio, rctViewRegion.right * dRatio,
				rctViewRegion.bottom * dRatio, rctViewRegion.top * dRatio,
				nearPlane_, farPlane_
			);
		}
		else isValid_ = false;
	}
	else
	{
		if (nearPlane_ < farPlane_)
			::glOrtho(
				rctViewRegion.left, rctViewRegion.right,
				rctViewRegion.bottom, rctViewRegion.top,
				nearPlane_, farPlane_
			);
		else isValid_ = false;
	}

	//lookAt();

	if (iOldMatrixMode != GL_PROJECTION) ::glMatrixMode(iOldMatrixMode);
	return true;
}

inline bool OglCamera::doUpdateViewport()
{
	::glViewport(viewport_.left, viewport_.bottom, viewport_.getWidth(), viewport_.getHeight());
	return doUpdateFrustum();
}

inline void OglCamera::lookAt()
{
	//doUpdateFrustum();
	::gluLookAt(eyePosX_, eyePosY_, eyePosZ_, eyePosX_+eyeDirX_, eyePosY_+eyeDirY_, eyePosZ_+eyeDirZ_, upDirX_, upDirY_, upDirZ_);
}

// projection transformation: an eye coordinates(before projection)  ==>  a clip coordinates(after projection)
// ==> OpenGL Red Book p. 96
bool OglCamera::doMapEyeToClip(const double ptEye[3], double ptClip[3]) const
{
	// rojection transformation: an eye coordinates  ==>  a clip coordinates
	Region2<double> rctViewRegion = getRevisedRegion();

	//--S [] 2001/08/08: Sang-Wook Lee
	// doUpdateFrustum()에서 ::glFrustum()을 호출하기 위해 clipping region을 수정하고 있기 때문에
	if (isPerspective())
		rctViewRegion *= calcResizingRatio();
	//--E [] 2001/08/08

	if (!rctViewRegion.isValid() || nearPlane_ >= farPlane_)
		return false;

	// from OpenGL Red Book
	if (isPerspective_)
	{
		double dW = -ptEye[2];
		if (dW <= 0.0 || nearPlane_ <= 0.0) return false;

		ptClip[0] = (2.0*nearPlane_*ptEye[0] + (rctViewRegion.right+rctViewRegion.left)*ptEye[2]) / (rctViewRegion.right - rctViewRegion.left);
		ptClip[1] = (2.0*nearPlane_*ptEye[1] + (rctViewRegion.top+rctViewRegion.bottom)*ptEye[2]) / (rctViewRegion.top - rctViewRegion.bottom);
		ptClip[2] = (-2.0*farPlane_*nearPlane_ - (farPlane_+nearPlane_)*ptEye[2]) / (farPlane_ - nearPlane_);

		// clip coordinates
		// because of perspective, -1.0 <= x, y <= 1.0 at near plane
		ptClip[0] /= dW;
		ptClip[1] /= dW;
		// -1.0 <= z <= 1.0
		ptClip[2] /= dW;
	}
	else
	{
		// clip coordinates
		// because of orthographic, -1.0 <= x, y & z <= 1.0
		ptClip[0] = (2.0*ptEye[0] - (rctViewRegion.right+rctViewRegion.left)) / (rctViewRegion.right - rctViewRegion.left);
		ptClip[1] = (2.0*ptEye[1] - (rctViewRegion.top+rctViewRegion.bottom)) / (rctViewRegion.top - rctViewRegion.bottom);
		ptClip[2] = (-2.0*ptEye[2] - (farPlane_+nearPlane_)) / (farPlane_ - nearPlane_);
	}

	return true;
}

// viewport transformation: a clip coordinates  ==>  a window coordinates
bool OglCamera::doMapClipToWindow(const double ptClip[3], double ptWin[3]) const
{
	// viewport transformation: a clip coordinates  ==>  a window coordinates
	const Region2<double> rctViewRegion = getRevisedRegion();
/*
	//--S [] 2001/08/08: Sang-Wook Lee
	// doUpdateFrustum()에서 ::glFrustum()을 호출하기 위해 clipping region을 수정하고 있기 때문에
	if (isPerspective())
		rctViewRegion *= calcResizingRatio();
	//--E [] 2001/08/08
*/
	double dNCx = ptClip[0] * rctViewRegion.getWidth() * 0.5 + rctViewRegion.getCenterX();
	double dNCy = ptClip[1] * rctViewRegion.getHeight() * 0.5 + rctViewRegion.getCenterY();

	int iX, iY;
	if (base_type::mapNcToVc(dNCx, dNCy, iX, iY))
	{
		ptWin[0] = double(iX);
		ptWin[1] = double(iY);
		ptWin[2] = ptClip[2];  // -1.0 <= z <= 1.0
		return true;
	}
	else return false;
}

// inverse viewport transformation: a window coordinates  ==>  a clip coordinates
bool OglCamera::doMapWindowToClip(const double ptWin[3], double ptClip[3]) const
{
	// inverse viewport transformation: a window coordinates  ==>  a clip coordinates
	const Region2<double> rctViewRegion = getRevisedRegion();
/*
	//--S [] 2001/08/08: Sang-Wook Lee
	// doUpdateFrustum()에서 ::glFrustum()을 호출하기 위해 clipping region을 수정하고 있기 때문에
	if (isPerspective())
		rctViewRegion *= calcResizingRatio();
	//--E [] 2001/08/08
*/
	double dNCx, dNCy;
	if (rctViewRegion.isValid() && base_type::mapVcToNc(int(ptWin[0]), int(ptWin[1]), dNCx, dNCy))
	{
		// if perspective, -1.0 <= x, y <= 1.0 at near plane
		// if orthographic, -1.0 <= x, y <= 1.0
		ptClip[0] = (dNCx - rctViewRegion.getCenterX()) / (rctViewRegion.getWidth() * 0.5);
		ptClip[1] = (dNCy - rctViewRegion.getCenterY()) / (rctViewRegion.getHeight() * 0.5);
		// -1.0 <= z <= 1.0
		ptClip[2] = ptWin[2];

		return true;
	}
	else return false;
}

// inverse projection transformation: a clip coordinates(after projection)  ==>  an eye coordinates(before projection)
// ==> OpenGL Red Book p. 96
bool OglCamera::doMapClipToEye(const double ptClip[3], double ptEye[3]) const
{
	// inverse projection transformation: a clip coordinates  ==>  an eye coordinates
	Region2<double> rctViewRegion = getRevisedRegion();

	//--S [] 2001/08/08: Sang-Wook Lee
	// doUpdateFrustum()에서 ::glFrustum()을 호출하기 위해 clipping region을 수정하고 있기 때문에
	if (isPerspective())
		rctViewRegion *= calcResizingRatio();
	//--E [] 2001/08/08

	if (!rctViewRegion.isValid() || nearPlane_ >= farPlane_)
		return false;

	// from OpenGL Red Book
	if (isPerspective_)
	{
		if (nearPlane_ <= 0.0) return false;

		ptEye[0] = ((rctViewRegion.right-rctViewRegion.left)*ptClip[0] + (rctViewRegion.right+rctViewRegion.left)) / (2.0 * nearPlane_);
		ptEye[1] = ((rctViewRegion.top-rctViewRegion.bottom)*ptClip[1] + (rctViewRegion.top+rctViewRegion.bottom)) / (2.0 * nearPlane_);
		ptEye[2] = -1.0;
		const double dW = (-(farPlane_-nearPlane_)*ptClip[2] + (farPlane_+nearPlane_)) / (2.0 * farPlane_ * nearPlane_);
		const double dEPS = 1.0e-10;
		if (-dEPS <= dW && dW <= dEPS) return false;

		ptEye[0] /= dW;
		ptEye[1] /= dW;
		ptEye[2] /= dW;
	}
	else
	{
		ptEye[0] = ((rctViewRegion.right-rctViewRegion.left)*ptClip[0] + (rctViewRegion.right+rctViewRegion.left)) * 0.5;
		ptEye[1] = ((rctViewRegion.top-rctViewRegion.bottom)*ptClip[1] + (rctViewRegion.top+rctViewRegion.bottom)) * 0.5;
		//ptEye[2] = (-(farPlane_-nearPlane_)*ptClip[2] + (farPlane_+nearPlane_)) * 0.5;
		ptEye[2] = (-(farPlane_-nearPlane_)*ptClip[2] - (farPlane_+nearPlane_)) * 0.5;
	}

	return true;
}

double OglCamera::calcResizingRatio() const
{
	const double dEPS = 1.0e-10;
	double dRatio = (-dEPS <= eyeDistance_ && eyeDistance_ <= dEPS) ? nearPlane_ : nearPlane_ / eyeDistance_;
	checkLimit(dRatio);
	return dRatio;
}

}  // namespace swl
