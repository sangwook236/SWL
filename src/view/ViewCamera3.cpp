#include "swl/Config.h"
#include "swl/view/ViewCamera3.h"
#include "swl/math/MathConstant.h"
#include <stdexcept>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#define  _TRANSFORM_SCENE_IN_KERNEL_SC_VIEWER_CAMERA_3D_CPP_  1

namespace swl {

//--------------------------------------------------------------------------
// 3-dimensional camera for view

ViewCamera3::ViewCamera3()
: base_type(),
  isValid_(true), isPerspective_(true),
  eyeDistance_(100.0),
  nearPlane_(0.0), farPlane_(10.0),
  refObjX_(0.0), refObjY_(0.0), refObjZ_(0.0),
  eyePosX_(0.0), eyePosY_(0.0), eyePosZ_(100.0),
  eyeDirX_(0.0), eyeDirY_(0.0), eyeDirZ_(-1.0),
  upDirX_(0.0), upDirY_(1.0), upDirZ_(0.0)
{
	isValid_ &= normalizeVector(eyeDir_) & normalizeVector(upDir_);
	updateEyePosition();
}

ViewCamera3::ViewCamera3(const ViewCamera3 &rhs)
: base_type(rhs),
  isValid_(rhs.isValid_), isPerspective_(rhs.isPerspective_),
  eyeDistance_(rhs.eyeDistance_),
  nearPlane_(rhs.nearPlane_), farPlane_(rhs.farPlane_),
  refObjX_(rhs.refObjX_), refObjY_(rhs.refObjY_), refObjZ_(rhs.refObjZ_),
  eyePosX_(rhs.eyePosX_), eyePosY_(rhs.eyePosY_), eyePosZ_(rhs.eyePosZ_),
  eyeDirX_(rhs.eyeDirX_), eyeDirY_(rhs.eyeDirY_), eyeDirZ_(rhs.eyeDirZ_),
  upDirX_(rhs.upDirX_), upDirY_(rhs.upDirY_), upDirZ_(rhs.upDirZ_)
{}

ViewCamera3::~ViewCamera3()  {}

ViewCamera3 & ViewCamera3::operator=(const ViewCamera3 &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;

	isValid_ = rhs.isValid_;
	isPerspective_ = rhs.isPerspective_;
	eyeDistance_ = rhs.eyeDistance_;
	nearPlane_ = rhs.nearPlane_;
	farPlane_ = rhs.farPlane_;
	for (int i = 0; i < 3; ++i)
	{
		refObj_[i] = rhs.refObj_[i];
		eyePos_[i] = rhs.eyePos_[i];
		eyeDir_[i] = rhs.eyeDir_[i];
		upDir_[i] = rhs.upDir_[i];
	}

	return *this;
}

/*
void ViewCamera3::write(std::ostream &stream)
{
	beginWrite(stream);
		// write version
		stream << ViewCamera3::getVersion() << " ";

		// write base class
		ViewCamera3::base_type::write(stream);

		// write self
		stream << isValid_ << ' ' << isPerspective_ << ' ' <<
			eyeDistance_ << ' ' << nearPlane_ <<' ' << farPlane_ << ' ' <<
			refObjX_ << ' ' << refObjY_ << ' ' << refObjZ_ << ' ' <<
			eyePosX_ << ' ' << eyePosY_ << ' ' << eyePosZ_ << ' ' <<
			eyeDirX_ << ' ' << eyeDirY_ << ' ' << eyeDirZ_ << ' ' <<
			upDirX_ << ' ' << upDirY_ << ' ' << upDirZ_ << '\n'; 
	endWriteEndl(stream);
}

void ViewCamera3::read(std::istream &stream)
{
	beginAssert(stream);
		// read version
		unsigned int iVersion;
		stream >> iVersion;

		// read base class
		ViewCamera3::base_type::read(stream);

		// read self
		unsigned int iOldVersion = SetReadVersion(iVersion);
		switch (iVersion)
		{
		case 20021008:
			read20021008(stream);
			break;
		default:
			//UtBsErrorHandler::throwFileReadError(ClassName(), iVersion);
			// version error
			static char err_msg[256];			
			sprintf(err_msg, "Cannot open file !\nVersion mismatch error!\n1) class name : %s\n2) class version : %d", ClassName().c_str() , iVersion);
			throw err_msg;	
		}
		setReadVersion(iOldVersion);
	endAssert(stream);
}

void ViewCamera3::Read20021008(std::istream &stream)
{
	stream >> isValid_ >> isPerspective_ >>
		eyeDistance_ >> nearPlane_ >> farPlane_ >>
		refObjX_ >> refObjY_ >> refObjZ_ >>
		eyePosX_ >> eyePosY_ >> eyePosZ_ >>
		eyeDirX_ >> eyeDirY_ >> eyeDirZ_ >>
		upDirX_ >> upDirY_ >> upDirZ_; 
}
*/

bool ViewCamera3::setViewBound(const double dLeft, const double dBottom, const double dRight, const double dTop, const double dNear, const double dFar)
{
	nearPlane_ = dNear <= dFar ? dNear : dFar;
	farPlane_ = dNear > dFar ? dNear : dFar;

	//-- [] 2001/05/22: Sang-Wook Lee
	return base_type::setViewBound(dLeft, dBottom, dRight, dTop) &&
		   doUpdateFrustum();
/*
	return base_type::setViewBound(dLeft, dBottom, dRight, dTop) &&
		   SetObjectPosition((dLeft+dRight)*0.5, (dBottom+dTop)*0.5, (dNear+dFar)*0.5);
*/
}

bool ViewCamera3::setViewport(const Region2<int> &rViewport)
{
	return base_type::setViewport(rViewport) &&
		   doUpdateViewport();
}

inline bool ViewCamera3::setViewRegion(const Region2<double> &rRct)
{
	// resize a viewing region && move the position of eye point
#if defined(_TRANSFORM_SCENE_IN_KERNEL_SC_VIEWER_CAMERA_3D_CPP_)
	return base_type::setViewRegion(rRct) &&
		   doUpdateFrustum();
#else  // _TRANSFORM_SCENE_IN_KERNEL_SC_VIEWER_CAMERA_3D_CPP_
	Point2<double> m_rctViewRegion.Center() - rRct.Center();
	return doUpdateZoomFactor() &&
		   translateScene(ptDelta.x, ptDelta.y) &&
		   doUpdateFrustum();
#endif  // _TRANSFORM_SCENE_IN_KERNEL_SC_VIEWER_CAMERA_3D_CPP_
}

inline bool ViewCamera3::moveViewRegion(const double dDeltaX, const double dDeltaY)
{
	// move the position of eye point
#if defined(_TRANSFORM_SCENE_IN_KERNEL_SC_VIEWER_CAMERA_3D_CPP_)
	return base_type::moveViewRegion(dDeltaX, dDeltaY) &&
		   doUpdateFrustum();
#else  // _TRANSFORM_SCENE_IN_KERNEL_SC_VIEWER_CAMERA_3D_CPP_
	return translateScene(dDeltaX, dDeltaY) &&
		   doUpdateFrustum();
#endif  // _TRANSFORM_SCENE_IN_KERNEL_SC_VIEWER_CAMERA_3D_CPP_
}

inline bool ViewCamera3::rotateViewRegion(const double dDeltaX, const double dDeltaY)
{
	// rotate the position of eye point, the direction of sight and up direction
	return rotateScene(dDeltaX, dDeltaY) &&
		   doUpdateFrustum();
}

inline bool ViewCamera3::scaleViewRegion(const double dFactor)
{
	return base_type::scaleViewRegion(dFactor) &&
		   doUpdateFrustum();
}

inline bool ViewCamera3::restoreViewRegion()
// zoom all the scene
{
	//-- [] 2001/05/22: Sang-Wook Lee
/*
	return base_type::restoreViewRegion() &&
		   updateEyePosition() &&
		   doUpdateFrustum();
*/
	return base_type::restoreViewRegion() &&
		   updateEyePosition();
}

bool ViewCamera3::setEyePose(const double dDirX, const double dDirY, const double dDirZ, const double dUpX, const double dUpY, const double dUpZ, const bool bUpdateViewpoint /*= true*/)
{
	eyeDirX_ = dDirX;  eyeDirY_ = dDirY;  eyeDirZ_ = dDirZ;
	upDirX_ = dUpX;  upDirY_ = dUpY;  upDirZ_ = dUpZ;

	isValid_ &= normalizeVector(eyeDir_) & normalizeVector(upDir_);
	if (!isValid_)  return false;

	for (int i = 0; i < 3; ++i)  eyePos_[i] = refObj_[i] - eyeDistance_ * eyeDir_[i];

	return bUpdateViewpoint ? doUpdateFrustum() : true;
}

bool ViewCamera3::setEyeFrame(const ViewMatrix3 &rFrame, const bool bUpdateViewpoint /*= true*/)
{
	double mat[12] = { 0., };
	rFrame.get(mat);

	for (int i = 0; i < 3; ++i)
	{
		upDir_[i] = mat[i*4 + 1];
		eyeDir_[i] = -mat[i*4 + 2];
		eyePos_[i] = mat[i*4 + 3];
	}

	return bUpdateViewpoint ? doUpdateFrustum() : true;
}

ViewMatrix3 ViewCamera3::getEyeFrame() const
{
	const double vYe[] = { upDirX_, upDirY_, upDirZ_ };
	const double vZe[] = { -eyeDirX_, -eyeDirY_, -eyeDirZ_ };
	const double vTe[] = { eyePosX_, eyePosY_, eyePosZ_ };
	double vXe[3] = { 0., };

	crossVector(vYe, vZe, vXe);
	return ViewMatrix3(vXe, vYe, vZe, vTe);
}

bool ViewCamera3::transform(const ViewMatrix3 &rdT, const bool bUpdateViewpoint /*= true*/)
{
	return setEyeFrame(rdT * getEyeFrame(), bUpdateViewpoint);
}

inline bool ViewCamera3::updateEyeDistance(const bool bUpdateViewpoint /*= true*/)
{
	eyeDistance_ = std::sqrt(std::pow(refObj_[0]-eyePos_[0], 2.0) + std::pow(refObj_[1]-eyePos_[1], 2.0) + std::pow(refObj_[2]-eyePos_[2], 2.0));
	return bUpdateViewpoint ? doUpdateFrustum() : true;  //-- [] 2001/05/22: Sang-Wook Lee
}

inline bool ViewCamera3::updateEyePosition(const bool bUpdateViewpoint /*= true*/)
{
	if (eyeDistance_ <= 0.0)  return isValid_ = false;
	for (int i = 0; i < 3; ++i)  eyePos_[i] = refObj_[i] - eyeDistance_ * eyeDir_[i];
	//return true;  //-- [] 2001/05/22: Sang-Wook Lee
	return bUpdateViewpoint ? doUpdateFrustum() : true;
}

inline bool ViewCamera3::updateEyeDirection(const bool bUpdateViewpoint /*= true*/)
{
	for (int i = 0; i < 3; ++i)  eyeDir_[i] = refObj_[i] - eyePos_[i];
	isValid_ &= normalizeVector(eyeDir_);
	//return isValid_;  //-- [] 2001/05/22: Sang-Wook Lee
	return !isValid_ ? false : bUpdateViewpoint ? doUpdateFrustum() : true;
}

bool ViewCamera3::translateScene(const double dDeltaX, const double dDeltaY)
// translate a scene along screen coordinate frames
{
	double vXs[3];
	crossVector(eyeDir_, upDir_, vXs);

	for (int i = 0; i < 3; ++i)
		eyePos_[i] -= dDeltaX * vXs[i] + dDeltaY * upDir_[i];

	return true;
}

bool ViewCamera3::rotateScene(const double dDeltaX, const double dDeltaY)
// rotate a scene about screen coordinate frames
{
	const double vYs[3] = { upDirX_, upDirY_, upDirZ_ };
	const double vZs[3] = { -eyeDirX_, -eyeDirY_, -eyeDirZ_ };
	double vXs[3];  crossVector(vYs, vZs, vXs);

	double vV[3] = { 0.0, };
	for (int i = 0; i < 3; ++i)  vV[i] = dDeltaX * vXs[i] + dDeltaY * vYs[i];
	isValid_ &= normalizeVector(vV);
	if (!isValid_)
	{
		isValid_ = true;
		return false;
	}

	double vU[3] = { 0.0, };  // rotational axis
	crossVector(vZs, vV, vU);

	const double dRad = -MathConstant::_4_PI * std::sqrt(dDeltaX*dDeltaX + dDeltaY*dDeltaY) / base_type::getViewRegion().getDiagonal();
	double mRot[9] = { 0.0, };
	if (!calcRotationMatrix(dRad, vU, mRot)) return false;

	// the direction from the reference point to the eye point
	memcpy(vV, eyeDir_, 3 * sizeof(double));
	productMatrixAndVector(mRot, vV, eyeDir_);
	eyePosX_ = refObjX_ - eyeDistance_ * eyeDirX_;  eyePosY_ = refObjY_ - eyeDistance_ * eyeDirY_;  eyePosZ_ = refObjZ_ - eyeDistance_ * eyeDirZ_;
	productMatrixAndVector(mRot, vYs, upDir_);

	return true;
}

bool ViewCamera3::rotateViewAboutAxis(const EAxis eAxis, const int iX1, const int iY1, const int iX2, const int iY2)
{
	// FIXME [implement] >>
	throw std::runtime_error("not yet implemented");
}

bool ViewCamera3::translateEye(const ViewCamera3::EAxis eAxis, const double dDelta)
// translate an eye point along eye coordinate frames
{
	switch (eAxis)
	{
	case XAXIS:
		{
			double vXe[3];  crossVector(upDir_, eyeDir_, vXe);
			for (int i = 0; i < 3; ++i)  eyePos_[i] += dDelta * vXe[i];
		}
		break;
	case YAXIS:
		for (int i = 0; i < 3; ++i)  eyePos_[i] -= dDelta * upDir_[i];
		break;
	case ZAXIS:
		for (int i = 0; i < 3; ++i)  eyePos_[i] += dDelta * eyeDir_[i];
		break;
	default:
		return false;
	}

	return doUpdateFrustum();
}

bool ViewCamera3::rotateEye(const ViewCamera3::EAxis eAxis, const double dRad)
// rotate an eye point about eye coordinate frames
{
	double mRot[9];

	switch (eAxis)
	{
	case XAXIS:
		{
			const double vYe[3] = { upDirX_, upDirY_, upDirZ_ };
			const double vZe[3] = { eyeDirX_, eyeDirY_, eyeDirZ_ };
			double vXe[3];
			crossVector(vYe, vZe, vXe);

			if (!calcRotationMatrix(dRad, vXe, mRot)) return false;
			productMatrixAndVector(mRot, vYe, upDir_);
			productMatrixAndVector(mRot, vZe, eyeDir_);
		}
		break;
	case YAXIS:
		{
			if (!calcRotationMatrix(dRad, upDir_, mRot)) return false;
			const double vZe[3] = { eyeDirX_, eyeDirY_, eyeDirZ_ };
			productMatrixAndVector(mRot, vZe, eyeDir_);
		}
		break;
	case ZAXIS:
		{
			if (!calcRotationMatrix(dRad, eyeDir_, mRot)) return false;
			const double vYe[3] = { upDirX_, upDirY_, upDirZ_ };
			productMatrixAndVector(mRot, vYe, upDir_);
		}
		break;
	default:
		return false;
	}

	return doUpdateFrustum();
}

bool ViewCamera3::normalizeVector(double vV[3]) const
// make unit vector
{
	const double dEPS = 1.0e-10;

	const double dNorm = normVector(vV);
	if (dNorm < dEPS) return false;

	// normalize vector
	for (int i = 0; i < 3; ++i) vV[i] /= dNorm;
	return true;
}

inline void ViewCamera3::crossVector(const double vV1[3], const double vV2[3], double vV[3]) const
// calculate cross product of two vectors, V = V1 * V2 = cross( V1, V2 )
{
	vV[0] = vV1[1] * vV2[2] - vV1[2] * vV2[1];
	vV[1] = vV1[2] * vV2[0] - vV1[0] * vV2[2];
	vV[2] = vV1[0] * vV2[1] - vV1[1] * vV2[0];
}

inline double ViewCamera3::normVector(const double vV[3]) const
{  return std::sqrt(vV[0]*vV[0] + vV[1]*vV[1] + vV[2]*vV[2]);  }

bool ViewCamera3::calcRotationMatrix(const double dRad, const double vV[3], double mRot[9]) const
//						   [  e0  e3  e6  ]
// R  =  [  x  y  z  ]  =  [  e1  e4  e7  ]
//						   [  e2  e5  e8  ]
{
	const double c = std::cos(dRad), s = std::sin(dRad), v = 1 - c;
	mRot[0] = vV[0]*vV[0]*v+c;			mRot[3] = vV[0]*vV[1]*v-vV[2]*s;	mRot[6] = vV[0]*vV[2]*v+vV[1]*s;
	mRot[1] = vV[0]*vV[1]*v+vV[2]*s;	mRot[4] = vV[1]*vV[1]*v+c;			mRot[7] = vV[1]*vV[2]*v-vV[0]*s;
	mRot[2] = vV[0]*vV[2]*v-vV[1]*s;	mRot[5] = vV[1]*vV[2]*v+vV[0]*s;	mRot[8] = vV[2]*vV[2]*v+c;
	if (!normalizeVector(&mRot[0]) || !normalizeVector(&mRot[3]) || !normalizeVector(&mRot[6]))
		return false;
	return true;
}

inline void ViewCamera3::productMatrixAndVector(const double mRot[9], const double vVi[3], double vVo[3]) const
{
	vVo[0] = mRot[0] * vVi[0] + mRot[3] * vVi[1] + mRot[6] * vVi[2];
	vVo[1] = mRot[1] * vVi[0] + mRot[4] * vVi[1] + mRot[7] * vVi[2];
	vVo[2] = mRot[2] * vVi[0] + mRot[5] * vVi[1] + mRot[8] * vVi[2];
}

// map an object coordinates(before projection) to a window coordinates(after projection)  ==>  OpenGL Red Book p. 96
bool ViewCamera3::mapObjectToWindow(const double ptObj[3], double ptWin[3]) const
{
	// 1. modeling & viewing transformation: an object coordinates  ==>  an eye coordinates
	double ptEye[3] = { 0., };  // an eye coordinates
	if (!doMapObjectToEye(ptObj, ptEye))
	{
		memset(ptWin, 0, sizeof(double) * 3);
		return false;
	}

	// 2. projection transformation: an eye coordinates  ==>  a clip coordinates
	// from OpenGL Red Book
	double ptClip[3] = { 0., };  // a clip coordinates
	if (!doMapEyeToClip(ptEye, ptClip))
	{
		memset(ptWin, 0, sizeof(double) * 3);
		return false;
	}

	// 3. viewport transformation: a clip coordinates  ==>  a window coordinates
	if (!doMapClipToWindow(ptClip, ptWin))
	{
		memset(ptWin, 0, sizeof(double) * 3);
		return false;
	}

	return true;
}

// map a window coordinates(after projection) to an object coordinates(before projection)  ==>  OpenGL Red Book p. 96
bool ViewCamera3::mapWindowToObject(const double ptWin[3], double ptObj[3]) const
{
	// 1. inverse viewport transformation: a window coordinates  ==>  a clip coordinates
	double ptClip[3] = { 0., };  // a clip coordinates
	if (!doMapWindowToClip(ptWin, ptClip))
	{
		memset(ptObj, 0, sizeof(double) * 3);
		return false;
	}

	// 2. inverse projection transformation: a clip coordinates  ==>  an eye coordinates
	// from OpenGL Red Book
	double ptEye[3] = { 0., };  // an eye coordinates
	if (!doMapClipToEye(ptClip, ptEye))
	{
		memset(ptObj, 0, sizeof(double) * 3);
		return false;
	}

	// 3. inverse modeling & viewing transformation: an eye coordinates  ==>  an object coordinates
	if (!doMapEyeToObject(ptEye, ptObj))
	{
		memset(ptObj, 0, sizeof(double) * 3);
		return false;
	}

	return true;
}

bool ViewCamera3::doMapObjectToEye(const double ptObj[3], double ptEye[3]) const
{
	// modeling & viewing transformation: an object coordinates  ==>  an eye coordinates
	return getEyeFrame().inverseTransformPoint(ptObj[0], ptObj[1], ptObj[2], ptEye[0], ptEye[1], ptEye[2]);
}

bool ViewCamera3::doMapEyeToClip(const double ptEye[3], double ptClip[3]) const
{
	// projection transformation: an eye coordinates  ==>  a clip coordinates
	const Region2<double> rctViewRegion = getCurrentViewRegion();
	if (rctViewRegion.left >= rctViewRegion.right ||
		rctViewRegion.bottom >= rctViewRegion.top ||
		nearPlane_ >= farPlane_)
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
		ptClip[0] /= dW;  // -1.0 <= x <= 1.0 at near plane
		ptClip[1] /= dW;  // -1.0 <= y <= 1.0 at near plane
		ptClip[2] /= dW;  // -1.0 <= z <= 1.0
	}
	else
	{
		// clip coordinates: -1.0 <= x, y & z <= 1.0
		ptClip[0] = (2.0*ptEye[0] - (rctViewRegion.right+rctViewRegion.left)) / (rctViewRegion.right - rctViewRegion.left);
		ptClip[1] = (2.0*ptEye[1] - (rctViewRegion.top+rctViewRegion.bottom)) / (rctViewRegion.top - rctViewRegion.bottom);
		ptClip[2] = (-2.0*ptEye[2] - (farPlane_+nearPlane_)) / (farPlane_ - nearPlane_);
	}
	
	return true;
}

bool ViewCamera3::doMapClipToWindow(const double ptClip[3], double ptWin[3]) const
{
	// viewport transformation: a clip coordinates  ==>  a window coordinates
	const Region2<double> rctViewRegion = getCurrentViewRegion();
	const double dNCx = ptClip[0] * rctViewRegion.getWidth() * 0.5;
	const double dNCy = ptClip[1] * rctViewRegion.getHeight() * 0.5;

	int iX, iY;
	if (base_type::mapCanvasToWindow(dNCx, dNCy, iX, iY))
	{
		ptWin[0] = double(iX);
		ptWin[1] = double(iY);
		ptWin[2] = ptClip[2];  // -1.0 <= z <= 1.0
		return true;
	}
	else  return false;
}

bool ViewCamera3::doMapWindowToClip(const double ptWin[3], double ptClip[3]) const
{
	// inverse viewport transformation: a window coordinates  ==>  a clip coordinates
	const Region2<double> rctViewRegion = getCurrentViewRegion();
	double dNCx, dNCy;
	if (rctViewRegion.isValid() && base_type::mapWindowToCanvas(int(ptWin[0]), int(ptWin[1]), dNCx, dNCy))
	{
		ptClip[0] = dNCx / (rctViewRegion.getWidth() * 0.5);
		ptClip[1] = dNCy / (rctViewRegion.getHeight() * 0.5);
		ptClip[2] = ptWin[2];  // -1.0 <= z <= 1.0

		return true;
	}
	else  return false;

}

bool ViewCamera3::doMapClipToEye(const double ptClip[3], double ptEye[3]) const
{
	// inverse projection transformation: a clip coordinates  ==>  an eye coordinates
	const Region2<double> rctViewRegion = getCurrentViewRegion();
	if (rctViewRegion.left >= rctViewRegion.right
		|| rctViewRegion.bottom >= rctViewRegion.top
		|| nearPlane_ >= farPlane_)
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
		ptEye[0] = ((rctViewRegion.right-rctViewRegion.left)*ptClip[0] + (rctViewRegion.right+rctViewRegion.left)) / 2.0;
		ptEye[1] = ((rctViewRegion.top-rctViewRegion.bottom)*ptClip[1] + (rctViewRegion.top+rctViewRegion.bottom)) / 2.0;
		//ptEye[2] = (-(farPlane_-nearPlane_)*ptClip[2] + (farPlane_+nearPlane_)) / 2.0;
		ptEye[2] = (-(farPlane_-nearPlane_)*ptClip[2] - (farPlane_+nearPlane_)) / 2.0;
	}

	return true;
}

bool ViewCamera3::doMapEyeToObject(const double ptEye[3], double ptObj[3]) const
{
	// inverse modeling & viewing transformation: an eye coordinates  ==>  an object coordinates
	return getEyeFrame().transformPoint(ptEye[0], ptEye[1], ptEye[2], ptObj[0], ptObj[1], ptObj[2]);
}

}  // namespace swl
