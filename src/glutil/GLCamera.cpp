#include "swl/Config.h"
#include "swl/glutil/GLCamera.h"
#include <GL/glut.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class GLCamera

GLCamera::GLCamera()
: base_type()
{}

GLCamera::GLCamera(const GLCamera &rhs)
: base_type(rhs)
{}

GLCamera::~GLCamera()  {}

GLCamera & GLCamera::operator=(const GLCamera &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type&>(*this) = rhs;
	return *this;
}

/*
void GLCamera::write(std::ostream &stream)
{
	beginWrite(stream);
		// write version
		stream << ' ' << GLCamera::getVersion();

		// write base class
		GLCamera::base_type::write(stream);

		// write self
	endWriteEndl(stream);
}

void GLCamera::read(std::istream &stream)
{
	beginAssert(stream);
		// read version
		unsigned int iVersion;
		stream >> iVersion;

		// read base class
		GLCamera::base_type::read(stream);

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

void GLCamera::read20021008(std::istream& stream)
{
}
*/

bool GLCamera::doUpdateFrustum()
{
	int oldMatrixMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);
	if (oldMatrixMode != GL_PROJECTION) glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	const Region2<double> &rctViewRegion = getCurrentViewRegion();

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
			const double dRatio = calcResizingRatio();
			glFrustum(
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
			glOrtho(
				rctViewRegion.left, rctViewRegion.right,
				rctViewRegion.bottom, rctViewRegion.top,
				nearPlane_, farPlane_
			);
		else isValid_ = false;
	}

	//lookAt();

	if (oldMatrixMode != GL_PROJECTION) glMatrixMode(oldMatrixMode);
	return true;
}

inline bool GLCamera::doUpdateViewport()
{
	glViewport(viewport_.left, viewport_.bottom, viewport_.getWidth(), viewport_.getHeight());
	return doUpdateFrustum();
}

inline void GLCamera::lookAt()
{
	//doUpdateFrustum();
	gluLookAt(eyePosX_, eyePosY_, eyePosZ_, eyePosX_+eyeDirX_, eyePosY_+eyeDirY_, eyePosZ_+eyeDirZ_, upDirX_, upDirY_, upDirZ_);
}

// projection transformation: an eye coordinates(before projection)  ==>  a clip coordinates(after projection)
// ==> OpenGL Red Book p. 96
bool GLCamera::doMapEyeToClip(const double ptEye[3], double ptClip[3]) const
{
	// rojection transformation: an eye coordinates  ==>  a clip coordinates
	Region2<double> rctViewRegion = getCurrentViewRegion();

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
		const double dW = -ptEye[2];
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
bool GLCamera::doMapClipToWindow(const double ptClip[3], double ptWin[3]) const
{
	// viewport transformation: a clip coordinates  ==>  a window coordinates
	const Region2<double> rctViewRegion = getCurrentViewRegion();
/*
	//--S [] 2001/08/08: Sang-Wook Lee
	// doUpdateFrustum()에서 ::glFrustum()을 호출하기 위해 clipping region을 수정하고 있기 때문에
	if (isPerspective())
		rctViewRegion *= calcResizingRatio();
	//--E [] 2001/08/08
*/
	const double dNCx = ptClip[0] * rctViewRegion.getWidth() * 0.5 + rctViewRegion.getCenterX();
	const double dNCy = ptClip[1] * rctViewRegion.getHeight() * 0.5 + rctViewRegion.getCenterY();

	int iX, iY;
	if (base_type::mapCanvasToWindow(dNCx, dNCy, iX, iY))
	{
		ptWin[0] = double(iX);
		ptWin[1] = double(iY);
		ptWin[2] = ptClip[2];  // -1.0 <= z <= 1.0
		return true;
	}
	else return false;
}

// inverse viewport transformation: a window coordinates  ==>  a clip coordinates
bool GLCamera::doMapWindowToClip(const double ptWin[3], double ptClip[3]) const
{
	// inverse viewport transformation: a window coordinates  ==>  a clip coordinates
	const Region2<double> rctViewRegion = getCurrentViewRegion();
/*
	//--S [] 2001/08/08: Sang-Wook Lee
	// doUpdateFrustum()에서 glFrustum()을 호출하기 위해 clipping region을 수정하고 있기 때문에
	if (isPerspective())
		rctViewRegion *= calcResizingRatio();
	//--E [] 2001/08/08
*/
	double dNCx, dNCy;
	if (rctViewRegion.isValid() && base_type::mapWindowToCanvas(int(ptWin[0]), int(ptWin[1]), dNCx, dNCy))
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
bool GLCamera::doMapClipToEye(const double ptClip[3], double ptEye[3]) const
{
	// inverse projection transformation: a clip coordinates  ==>  an eye coordinates
	Region2<double> rctViewRegion = getCurrentViewRegion();

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

double GLCamera::calcResizingRatio() const
{
	const double dEPS = 1.0e-10;
	double dRatio = (-dEPS <= eyeDistance_ && eyeDistance_ <= dEPS) ? nearPlane_ : nearPlane_ / eyeDistance_;
	checkLimit(dRatio);
	return dRatio;
}

}  // namespace swl
