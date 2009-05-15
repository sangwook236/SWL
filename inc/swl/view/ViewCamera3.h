#if !defined(__SWL_VIEW__VIEW_CAMERA3__H_)
#define __SWL_VIEW__VIEW_CAMERA3__H_ 1


#include "swl/view/ViewCamera2.h"
#include "swl/view/ViewMatrix.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// 3-dimensional camera for view

class SWL_VIEW_API ViewCamera3: public ViewCamera2
{
public:
	typedef ViewCamera2 base_type;

public:
	enum EAxis { XAXIS, YAXIS, ZAXIS };

public:
	ViewCamera3();
	ViewCamera3(const ViewCamera3 &rhs);
	virtual ~ViewCamera3();

	ViewCamera3 & operator=(const ViewCamera3 &rhs);

public:
	/// virtual copy constructor
	/*virtual*/ ViewCamera2 * cloneCamera() const
	{  return new ViewCamera3(*this);  }

	///*virtual*/ void write(::std::ostream& stream);
	///*virtual*/ void read(::std::istream& stream);

	/// set the size of viewing volume: (left, bottom, right, top) is set wrt a eye coordinates frame
    /*virtual*/ bool setViewBound(double dLeft, double dBottom, double dRight, double dTop, double dNear, double dFar);
	/// get the size of viewing volume
    /*virtual*/ void getViewBound(double& rdLeft, double& rdBottom, double& rdRight, double& rdTop, double& rdNear, double& rdFar) const
	{
		base_type::getViewBound(rdLeft, rdBottom, rdRight, rdTop);
		rdNear = nearPlane_;  rdFar = farPlane_;
	}

	/// set the size of viewport
    /*virtual*/ bool setViewport(int iLeft, int iBottom, int iRight, int iTop)
	{  return setViewport(Region2<int>(iLeft, iBottom, iRight, iTop));  }
    /*virtual*/ bool setViewport(const Point2<int> &rPt1, Point2<int> &rPt2)
	{  return setViewport(Region2<int>(rPt1, rPt2));  }
    /*virtual*/ bool setViewport(const Region2<int> &rViewport);

	/// set a viewing region
	/*virtual */ bool setViewRegion(double dX1, double dY1, double dX2, double dY2)
	{  return setViewRegion(Region2<double>(dX1, dY1, dX2, dY2));  }
    /*virtual */ bool setViewRegion(const Point2<double> &rPt1, Point2<double> &rPt2)
	{  return setViewRegion(Region2<double>(rPt1, rPt2));  }
	/*virtual */ bool setViewRegion(const Region2<double> &rRct);

	/// move a viewing region
	/*virtual*/ bool moveViewRegion(double dDeltaX, double dDeltaY);
	/// rotate a viewing region
	/*virtual*/ bool rotateViewRegion(double dDeltaX, double dDeltaY);
	/// scale a viewing region
	/*virtual*/ bool scaleViewRegion(double dFactor);

	/// restore a viewing region
	/*virtual*/ bool restoreViewRegion();

	///
	bool isValid() const  {  return isValid_;  }
	void resetValid()  {  isValid_ = true;  }

	/// which projection mode is perspective or orthographic
    bool isPerspective() const  {  return isPerspective_;  }
    void setPerspective(bool bIsPerspective = true)
	{
		if (isPerspective_ != bIsPerspective)
		{
			isPerspective_ = bIsPerspective;
			/*return*/ doUpdateFrustum();
		}
	}

	///
	bool setEyeDistance(double dEyeDistance, const bool bUpdateViewpoint = true)
	{
		eyeDistance_ = dEyeDistance;
		return updateEyePosition(bUpdateViewpoint);
	}
	double getEyeDistance() const  {  return eyeDistance_;  }

	///
	bool setObjectPosition(double dObjX, double dObjY, double dObjZ, const bool bUpdateViewpoint = true)
	{
		refObjX_ = dObjX;  refObjY_ = dObjY;  refObjZ_ = dObjZ;
		//--S [] 2001/05/22: Sang-Wook Lee
		//updateEyeDistance( false );
		//return updateEyeDirection( bUpdateViewpoint );
		return updateEyePosition(bUpdateViewpoint);
		//--E [] 2001/05/22
	}
	void getObjectPosition(double& rdObjX, double& rdObjY, double& rdObjZ) const
	{  rdObjX = refObjX_;  rdObjY = refObjY_;  rdObjZ = refObjZ_;  }

	///
	bool setEyePose(double dDirX, double dDirY, double dDirZ, double dUpX, double dUpY, double dUpZ, const bool bUpdateViewpoint = true);
	void getEyePose(double& rdDirX, double& rdDirY, double& rdDirZ, double& rdUpX, double& rdUpY, double& rdUpZ) const
	{
		rdDirX = eyeDirX_;  rdDirY = eyeDirY_;  rdDirZ = eyeDirZ_;
		rdUpX = upDirX_;  rdUpY = upDirY_;  rdUpZ = upDirZ_;
	}
	void getEyeDirection(double& rdDirX, double& rdDirY, double& rdDirZ) const
	{  rdDirX = eyeDirX_;  rdDirY = eyeDirY_;  rdDirZ = eyeDirZ_;  }

	///
	bool setEyePosition(double dEyeX, double dEyeY, double dEyeZ, const bool bUpdateViewpoint = true)
	{
		eyePosX_ = dEyeX;  eyePosY_ = dEyeY;  eyePosZ_ = dEyeZ;
		updateEyeDistance(false);
		return updateEyeDirection(bUpdateViewpoint);
	}
	void getEyePosition(double& rdEyeX, double& rdEyeY, double& rdEyeZ) const
	{  rdEyeX = eyePosX_;  rdEyeY = eyePosY_;  rdEyeZ = eyePosZ_;  }

	///
	bool setEyeUp(double dUpX, double dUpY, double dUpZ, const bool bUpdateViewpoint = true)
	{
		upDirX_ = dUpX;  upDirY_ = dUpY;  upDirZ_ = dUpZ;
		isValid_ &= normalizeVector(upDir_);
		//return isValid_;  //-- [] 2001/05/22: Sang-Wook Lee
		return !isValid_ ? false : bUpdateViewpoint ? doUpdateFrustum() : true;
	}
	void getEyeUp(double& rdUpX, double& rdUpY, double& rdUpZ) const
	{  rdUpX = upDirX_;  rdUpY = upDirY_;  rdUpZ = upDirZ_;  }

	/// set an eye coordinates frame wrt a world coordinates frame
	bool setEyeFrame(const ViewMatrix3& rFrame, const bool bUpdateViewpoint = true);
	/// get an eye coordinates frame wrt a world coordinates frame
	ViewMatrix3 getEyeFrame() const;

	/// transform an eye coordinates frame wrt a world coordinates frame, Te' = rdT * Te
	bool transform(const ViewMatrix3& rdT, const bool bUpdateViewpoint = true);

	///
	bool setViewDepth(double dNear, double dFar, const bool bUpdateViewpoint = true)
	{
		nearPlane_ = dNear;  farPlane_ = dFar;
		isValid_ &= (nearPlane_ < farPlane_) && (!isPerspective_ || (nearPlane_ > 0.0 && farPlane_ > 0.0));
		//return isValid_;  //-- [] 2001/05/22: Sang-Wook Lee
		return !isValid_ ? false : bUpdateViewpoint ? doUpdateFrustum() : true;
	}
	void getViewDepth(double& rdNear, double& rdFar) const
	{  rdNear = nearPlane_;  rdFar = farPlane_;  }

	/// transform an eye wrt eye coordinate frames
	virtual bool translateEye(EAxis eAxis, double dDelta);
	virtual bool rotateEye(EAxis eAxis, double dRad);

	/// map an object coordinates(before projection) to a window coordinates(after projection)
	/// ==> OpenGL Red Book(p. 96)
	bool mapObjectToWindow(const double ptObj[3], double ptWin[3]) const;
	/// map a window coordinates(after projection) to an object coordinates(before projection)
	/// ==> OpenGL Red Book(p. 96)
	bool mapWindowToObject(const double ptWin[3], double ptObj[3]) const;

	/// update the camera
	/*virtual*/ bool update()  {  return doUpdateFrustum();  }

protected:
	///
	virtual bool doUpdateFrustum()  {  return true;  }
	virtual bool doUpdateViewport()  {  return doUpdateFrustum();  }

	/// modeling & viewing transformation: an object coordinates  ==>  an eye coordinates
	virtual bool doMapObjectToEye(const double ptObj[3], double ptEye[3]) const;
	/// projection transformation: an eye coordinates  ==>  a clip coordinates
	virtual bool doMapEyeToClip(const double ptEye[3], double ptClip[3]) const;
	/// viewport transformation: a clip coordinates  ==>  a window coordinates
	virtual bool doMapClipToWindow(const double ptClip[3], double ptWin[3]) const;
	/// inverse viewport transformation: a window coordinates  ==>  a clip coordinates
	virtual bool doMapWindowToClip(const double ptWin[3], double ptClip[3]) const;
	/// inverse projection transformation: a clip coordinates  ==>  an eye coordinates
	virtual bool doMapClipToEye(const double ptClip[3], double ptEye[3]) const;
	/// inverse modeling & viewing transformation: an eye coordinates  ==>  an object coordinates
	virtual bool doMapEyeToObject(const double ptEye[3], double ptObj[3]) const;

	///
	bool updateEyeDistance(const bool bUpdateViewpoint = true);
	bool updateEyePosition(const bool bUpdateViewpoint = true);
	bool updateEyeDirection(const bool bUpdateViewpoint = true);

	/// transform a scene wrt screen coordinate frames
	virtual bool translateScene(double dDeltaX, double dDeltaY);
	virtual bool rotateScene(double dDeltaX, double dDeltaY);

	/// vector & matrix arithmetic
	bool normalizeVector(double vV[3]) const;
	void crossVector(const double vV1[3], const double vV2[3], double vV[3]) const;
	double normVector(const double vV[3]) const;
	bool calcRotationMatrix(double dRad, const double vV[3], double mRot[9]) const;
	void productMatrixAndVector(const double mRot[9], const double vVi[3], double vVo[3]) const;

private:
	///
	//void Read20021008(::std::istream& stream);

protected:
	/// check if the camera's state is valid or not
	bool isValid_;
	/// check if projection mode is perspective or orthographic
    bool isPerspective_;

	/// the positive distance from the eye point(viewpoint) to the reference(object) point
	/// by default, the distance is 100.0
	double eyeDistance_;

	/// the distance from the eye point(viewpoint) to the near & far clipping planes of viewing volume
    double nearPlane_, farPlane_;    

	/// the position of the reference(object) point wrt the world coordinate system
	/// by default, the position is (0.0, 0.0, 0.0)
    union
	{
        struct { double refObjX_, refObjY_, refObjZ_; };
        double refObj_[3];
    };
	/// the position of the eye point(viewpoint)
    union
	{
        struct { double eyePosX_, eyePosY_, eyePosZ_; };
        double eyePos_[3];
    };
	/// the direction of the line of sight from the eye point(viewpoint) to the reference(object) point wrt the world coordinate system, unit vector
    union
	{
        struct { double eyeDirX_, eyeDirY_, eyeDirZ_; };
        double eyeDir_[3];
    };
	/// the direction of the up vector wrt the world coordinate system, unit vector
    union
	{
        struct { double upDirX_, upDirY_, upDirZ_; };
        double upDir_[3];
    };
};

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_CAMERA3__H_
