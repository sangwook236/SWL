#if !defined(__SWL_VIEW__VIEW_CAMERA2__H_)
#define __SWL_VIEW__VIEW_CAMERA2__H_ 1


#include "swl/view/ExportView.h"
#include "swl/common/Region.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// 2-dimensional camera for view

class SWL_VIEW_API ViewCamera2
{
public:
	//typedef ViewCamera2			base_type;
	/// the type of a normalized coordinates ( canvas coordinates )
	typedef double					nc_type;
	/// the type of a viewport coordinates ( screen coordinates )
	typedef int						vc_type;

public:
	ViewCamera2();
	ViewCamera2(const ViewCamera2& rhs);
	virtual ~ViewCamera2();

	ViewCamera2& operator=(const ViewCamera2& rhs);

public:
	/// virtual copy constructor
	virtual ViewCamera2* cloneCamera() const
	{  return new ViewCamera2(*this);  }

	///*virtual*/ void write(::std::ostream& stream);
	///*virtual*/ void read(::std::istream& stream);

	///
	double getZoomFactor() const  {  return zoomFactor_;  }

	/// set the size of viewing bound
    virtual bool setViewBound(double dLeft, double dBottom, double dRight, double dTop)
	{  return setViewBound(Region2<double>(dLeft, dBottom, dRight, dTop) );  }
    virtual bool setViewBound(const Point2<double>& rPt1, Point2<double>& rPt2)
	{  return setViewBound(Region2<double>(rPt1, rPt2));  }
    virtual bool setViewBound(const Region2<double>& rViewBound);
	/// get the size of viewing bound
    virtual void getViewBound(double& rdLeft, double& rdBottom, double& rdRight, double& rdTop) const
	{
		rdLeft = viewBound_.left;  rdBottom = viewBound_.bottom;
		rdRight = viewBound_.right;  rdTop = viewBound_.top;
	}
    virtual Region2<double> getViewBound() const  {  return viewBound_;  }

	///
    virtual bool setViewBound(double dLeft, double dBottom, double dRight, double dTop, double dNear, double dFar)
	{  return setViewBound(Region2<double>(dLeft, dBottom, dRight, dTop));  }
    virtual void getViewBound(double& rdLeft, double& rdBottom, double& rdRight, double& rdTop, double& rdNear, double& rdFar) const
	{
		getViewBound(rdLeft, rdBottom, rdRight, rdTop);
		rdNear = rdFar = 0.0;
	}

	/// set the size of viewport
    virtual bool setViewport(int iLeft, int iBottom, int iRight, int iTop)
	{  return setViewport(Region2<int>(iLeft, iBottom, iRight, iTop));  }
    virtual bool setViewport(const Point2<int>& rPt1, Point2<int>& rPt2)
	{  return setViewport(Region2<int>(rPt1, rPt2));  }
    virtual bool setViewport(const Region2<int>& rViewport);
	/// get the size of viewport
    virtual void getViewport(int& riLeft, int& riBottom, int& riRight, int& riTop) const
	{
		riLeft = viewport_.left;  riBottom = viewport_.bottom;
		riRight = viewport_.right;  riTop = viewport_.top;
	}
    virtual Region2<int> getViewport() const  {  return viewport_;  }

	/// set a viewing region
	virtual bool setViewRegion(double dX1, double dY1, double dX2, double dY2)
	{  return setViewRegion(Region2<double>(dX1, dY1, dX2, dY2));  }
    virtual bool setViewRegion(const Point2<double>& rPt1, Point2<double>& rPt2)
	{  return setViewRegion(Region2<double>(rPt1, rPt2));  }
	virtual bool setViewRegion(const Region2<double>& rRct);
	/// get the size of viewport
    virtual void getViewRegion(double& rdX1, double& rdY1, double& rdX2, double& rdY2) const
	{
		rdX1 = viewRegion_.left;  rdY1 = viewRegion_.bottom;
		rdX2 = viewRegion_.right;  rdY2 = viewRegion_.top;
	}
    virtual Region2<double> getViewRegion() const  {  return viewRegion_;  }

	///
	virtual Region2<double> getRevisedRegion() const;

	/// resize a viewing region
	virtual bool resizeViewRegion(double dWidth, double dHeight);
	/// move a viewing region
	virtual bool moveViewRegion(double dDeltaX, double dDeltaY);
	virtual bool moveViewRegion(const Point2<double>& rDelta)
	{  return moveViewRegion(rDelta.x, rDelta.y);  }
	/// rotate a viewing region
	virtual bool rotateViewRegion(double dDeltaX, double dDeltaY)  {  return true;  }
	virtual bool rotateViewRegion(const Point2<double>& rDelta)
	{  return rotateViewRegion(rDelta.x, rDelta.y);  }
	/// scale a viewing region
	virtual bool scaleViewRegion(double dFactor);

	/// restore a viewing region
	virtual bool restoreViewRegion();

 	/// set a view
	virtual bool setView(int iX1, int iY1, int iX2, int iY2);
	/// move a view
	virtual bool moveView(int iDeltaX, int iDeltaY);
	/// rotate a view
	virtual bool rotateView(int iDeltaX, int iDeltaY);

	/// look at the view region from the eye position along the eye direction and up direction
	/// this function is needed usually in 3D, but not 2D 
	virtual void lookAt()  {}

	/// map a window(screen, viewport) coordinates to a canvas(normalized, object) coordinates
    virtual bool mapWindowToCanvas(int iVCx, int iVCy, double& rNCx, double& rNCy) const;
    virtual bool mapWindowToCanvas(const Point2<int>& rVC, Point2<double>& rNC) const;
    /// map a canvas(normalized, object) coordinates to a window(screen, viewport) coordinates
    virtual bool mapCanvasToWindow(double dNCx, double dNCy, int& rVCx, int& rVCy) const;
    virtual bool mapCanvasToWindow(const Point2<double>& rNC, Point2<int>& rVC) const;

	/// for scrolling window
	virtual bool getHorizontalRatio(float& rfPosRatio, float& rfWidthRatio);
	virtual bool getVerticalRatio(float& rfPosRatio, float& rfHeightRatio);

	virtual bool scrollHorizontally(float fRatio);
	virtual bool scrollVertically(float fRatio);

	/// update the camera
	virtual bool update()  {  return true;  }

protected:
	///
	virtual bool doUpdateZoomFactor();

	///
	void checkLimit(double& rdValue) const;

	///
	double round(double dValue) const;

private:
	///
	//void read20021008(::std::istream& stream);

protected:
	/// 1. take the direction of axes into consideration
	///   -. in case of upward y-axis        -. in case of downward y-axis
	///
	///                   (right,top)                  +x          
	///             глгнгнгнгнгл               глгнгнгнг╛
	///             г№        г№               г№
	///   +yб№      г№        г№               г№ (left,bottom)
	///     г№      глгнгнгнгнгл               г№       глгнгнгнгнгл
	///     г№ (left,bottom)                 +yб¤       г№        г№
	///     г№                                          г№        г№
	///     глгнгнгнг╛                                  глгнгнгнгнгл
	///             +x                                        (right,top)
	///
	/// 2. conclusion
	///	left <= right  &&  bottom <= top, always

	/// -. axis direction
	///	: x-axis ==> rightward,  y-axis ==> upward

    /// the size of viewing bound expressed in a normalized coordinates
    Region2<double> viewBound_;
    /// the size of viewport expressed in a viewport coordinates
    Region2<int> viewport_;

    /// the size of viewing region expressed in a normalized coordinates
    Region2<double> viewRegion_;

	/// zoom factor
	double zoomFactor_;
};


//-----------------------------------------------------------------------------------------
//

template<typename T>
ViewCamera2* createCamera()
{  return new T();  }

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_CAMERA2__H_
