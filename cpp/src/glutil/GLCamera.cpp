#include "swl/Config.h"
#include "swl/glutil/GLCamera.h"
#include "swl/math/MathConstant.h"
#include <GL/glut.h>
#include <cmath>
#include <cstring>


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
	GLint oldMatrixMode;
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

bool GLCamera::rotateViewAboutAxis(const EAxis eAxis, const int iX1, const int iY1, const int iX2, const int iY2)
{
#if 1
	const double dDeltaX = double(iX2 - iX1) / zoomFactor_;
	const double dDeltaY = double(iY1 - iY2) / zoomFactor_;  // upward y-axis
	//const double dDeltaY = double(iY2 - iY1) / zoomFactor_;  // downward y-axis

	const bool isPositiveDir = std::fabs(dDeltaX) > std::fabs(dDeltaY) ? dDeltaX >= 0.0 : dDeltaY >= 0.0;
	double rotAxis[3] = { 0.0, 0.0, 0.0 };  // rotational axis
	switch (eAxis)
	{
	case XAXIS:
		rotAxis[0] = isPositiveDir ? 1.0 : -1.0;
		break;
	case YAXIS:
		rotAxis[1] = isPositiveDir ? 1.0 : -1.0;
		break;
	case ZAXIS:
		rotAxis[2] = isPositiveDir ? 1.0 : -1.0;
		break;
	default:
		return false;
	}

	const double rotAngle = -MathConstant::_2_PI * std::sqrt(dDeltaX*dDeltaX + dDeltaY*dDeltaY) / base_type::getViewRegion().getDiagonal();
	double mRot[9] = { 0.0, };
	if (!calcRotationMatrix(rotAngle, rotAxis, mRot)) return false;

	// the direction from the reference point to the eye point
	double vV[3] = { 0.0, };
	memcpy(vV, eyeDir_, 3 * sizeof(double));
	productMatrixAndVector(mRot, vV, eyeDir_);
	eyePosX_ = refObjX_ - eyeDistance_ * eyeDirX_;  eyePosY_ = refObjY_ - eyeDistance_ * eyeDirY_;  eyePosZ_ = refObjZ_ - eyeDistance_ * eyeDirZ_;
	memcpy(vV, upDir_, 3 * sizeof(double));
	productMatrixAndVector(mRot, vV, upDir_);

	return true;
#else
	const double eps = 1.0e-20;

	GLint oldMatrixMode = 0;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(GL_MODELVIEW);

	glPushMatrix();
		glLoadIdentity();
		// 1. need to load current modelview matrix
		//   e.g.) glLoadMatrixd(modelviewMatrix);
		// 2. need to be thought of viewing transformation
		//   e.g.) camera->lookAt();
		lookAt();

		double modelview[16] = { 0.0, }, projection[16] = { 0.0, };
		int viewport[4] = { 0, };
		double depthRange[2] = { 0.0, 1.0 };
		glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
		glGetDoublev(GL_PROJECTION_MATRIX, projection);
		glGetIntegerv(GL_VIEWPORT, viewport);
		glGetDoublev(GL_DEPTH_RANGE, depthRange);

		double pt1[3] = { 0.0, }, pt2[3] = { 0.0, };
		const bool ret = gluUnProject(double(iX1), double(viewport[3] - iY1), depthRange[0], modelview, projection, viewport, &pt1[0], &pt1[1], &pt1[2]) == GL_TRUE &&
			gluUnProject(double(iX2), double(viewport[3] - iY2), depthRange[0], modelview, projection, viewport, &pt2[0], &pt2[1], &pt2[2]) == GL_TRUE;
	glPopMatrix();

	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(oldMatrixMode);
	if (!ret) return false;

	double vec1[3] = { pt1[0] - refObj_[0], pt1[1] - refObj_[1], pt1[2] - refObj_[2] };
	double vec2[3] = { pt2[0] - refObj_[0], pt2[1] - refObj_[1], pt2[2] - refObj_[2] };

	normalizeVector(vec1);
	normalizeVector(vec2);

	double rotAxis[3] = { 0.0, 0.0, 0.0 };  // rotational axis
	double rotAngle = 0.0;
	switch (eAxis)
	{
	case XAXIS:
		{
			rotAxis[0] = 1.0;
			const double norm1 = std::sqrt(vec1[1]*vec1[1] + vec1[2]*vec1[2]);
			const double norm2 = std::sqrt(vec2[1]*vec2[1] + vec2[2]*vec2[2]);
			if (norm1 <= eps || norm2 <= eps) return false;
			rotAngle = std::acos((vec1[1] * vec2[1] + vec1[2] * vec2[2]) / (norm1 * norm2));
		}
		break;
	case YAXIS:
		{
			rotAxis[1] = 1.0;
			const double norm1 = std::sqrt(vec1[0]*vec1[0] + vec1[2]*vec1[2]);
			const double norm2 = std::sqrt(vec2[0]*vec2[0] + vec2[2]*vec2[2]);
			if (norm1 <= eps || norm2 <= eps) return false;
			rotAngle = std::acos((vec1[0] * vec2[0] + vec1[2] * vec2[2]) / (norm1 * norm2));
		}
		break;
	case ZAXIS:
		{
			rotAxis[2] = 1.0;
			const double norm1 = std::sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1]);
			const double norm2 = std::sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1]);
			if (norm1 <= eps || norm2 <= eps) return false;
			rotAngle = std::acos((vec1[0] * vec2[0] + vec1[1] * vec2[1]) / (norm1 * norm2));
		}
		break;
	default:
		return false;
	}

	double vec3[3] = { 0.0, };
	crossVector(vec1, vec2, vec3);
	if (!normalizeVector(vec3)) return false;

	const double betweenAngle = std::acos(vec3[0] * rotAxis[0] + vec3[1] * rotAxis[1] + vec3[2] * rotAxis[2]);
	const bool isPositiveDir = betweenAngle < PI * 0.5; 

	double mRot[9] = { 0.0, };
	if (!calcRotationMatrix(isPositiveDir ? -rotAngle : rotAngle, rotAxis, mRot)) return false;

	// the direction from the reference point to the eye point
	double vV[3] = { 0.0, };
	memcpy(vV, eyeDir_, 3 * sizeof(double));
	productMatrixAndVector(mRot, vV, eyeDir_);
	eyePosX_ = refObjX_ - eyeDistance_ * eyeDirX_;  eyePosY_ = refObjY_ - eyeDistance_ * eyeDirY_;  eyePosZ_ = refObjZ_ - eyeDistance_ * eyeDirZ_;
	memcpy(vV, upDir_, 3 * sizeof(double));
	productMatrixAndVector(mRot, vV, upDir_);

	return true;
#endif
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
	// doUpdateFrustum()에서 glFrustum()을 호출하기 위해 clipping region을 수정하고 있기 때문에
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
	// doUpdateFrustum()에서 glFrustum()을 호출하기 위해 clipping region을 수정하고 있기 때문에
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
