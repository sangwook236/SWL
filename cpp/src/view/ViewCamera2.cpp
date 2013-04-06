#include "swl/Config.h"
#include "swl/view/ViewCamera2.h"
#include <cmath>
#include <cassert>
#include <limits>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// 2-dimensional camera for view

ViewCamera2::ViewCamera2()
: viewBound_(), viewport_(), viewRegion_(), zoomFactor_(1.0)
{}

ViewCamera2::ViewCamera2(const ViewCamera2 &rhs)
: viewBound_(rhs.viewBound_), viewport_(rhs.viewport_),
  viewRegion_(rhs.viewRegion_), zoomFactor_(rhs.zoomFactor_)
{}

ViewCamera2::~ViewCamera2()  {}

ViewCamera2 & ViewCamera2::operator=(const ViewCamera2 &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;

	viewBound_ = rhs.viewBound_;
	viewport_ = rhs.viewport_;
	viewRegion_ = rhs.viewRegion_;
	zoomFactor_ = rhs.zoomFactor_;
	
	return *this;
}

/*
void ViewCamera2::write(std::ostream &stream)
{
	beginWrite(stream);
		// write version
		stream << ViewCamera2::getVersion() << " ";

		// write base class
		//ViewCamera2::base_type::write(stream);

		// write self
		stream << viewBound_.left << ' ' <<viewBound_.bottom << ' '	<< viewBound_.right << ' ' <<viewBound_.top << ' ' <<
			viewport_.left << ' ' << viewport_.bottom << ' ' << viewport_.right << ' ' <<viewport_.top << ' ' <<
			viewRegion_.left << ' ' << viewRegion_.bottom << ' ' << viewRegion_.right << ' ' << viewRegion_.top << '\n';
	endWriteEndl(stream);
}

void ViewCamera2::read(std::istream &stream)
{
	beginAssert(stream);
		// read version
		unsigned int iVersion;
		stream >> iVersion;

		// read base class
		//ViewCamera2::base_type::read(stream);

		// read self
		unsigned int iOldVersion = setReadVersion(iVersion);
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

void ViewCamera2::read20021008(std::istream &stream)
{
	stream >> viewBound_.left >> viewBound_.bottom >> viewBound_.right >> viewBound_.top >>
		viewport_.left >> viewport_.bottom >> viewport_.right >> viewport_.top >>
		viewRegion_.left >> viewRegion_.bottom >> viewRegion_.right >> viewRegion_.top;
	doUpdateZoomFactor();
}
*/
inline bool ViewCamera2::setViewBound(const Region2<double> &rViewBound)
{
	if (!rViewBound.isValid())  return false;

	viewBound_ = viewRegion_ = rViewBound;

	// update zoom factor
	return doUpdateZoomFactor();
}

inline bool ViewCamera2::setViewport(const Region2<int> &rViewport)
{
	if (!rViewport.isValid())  return false;

	viewport_ = rViewport;

	// update zoom factor
	return doUpdateZoomFactor();
}

bool ViewCamera2::doUpdateZoomFactor()
{
	// calculate a ratio of aspect ratio of viewport to that of viewing bound
	if (!viewport_.isValid() || !viewRegion_.isValid())  return false;

	const double dWidthVP = double(viewport_.getWidth()), dHeightVP = double(viewport_.getHeight());
	const double dWidthVR = viewRegion_.getWidth(), dHeightVR = viewRegion_.getHeight();
    double dRatioAR = (dWidthVP / dHeightVP) / (dWidthVR / dHeightVR);
	checkLimit(dRatioAR);

	// resize viewing region
	zoomFactor_ = dRatioAR > 1.0 ? dWidthVP / (dWidthVR * dRatioAR) : dHeightVP * dRatioAR / dHeightVR;
	checkLimit(zoomFactor_);

	return true;
}

inline bool ViewCamera2::setViewRegion(const Region2<double> &rRct)
{
	if (!rRct.isValid())  return false;

	// move the center of a viewing region
	viewRegion_ = rRct;

	// update zoom factor
	return doUpdateZoomFactor();
}

inline bool ViewCamera2::resizeViewRegion(const double dWidth, const double dHeight)
{
	viewRegion_.changeSize(dWidth, dHeight);
	// update zoom factor
	return doUpdateZoomFactor();
}

inline bool ViewCamera2::moveViewRegion(const double dDeltaX, const double dDeltaY)
{
	// caution: the size of a viewing region isn't changed
	viewRegion_ -= Point2<double>(dDeltaX, dDeltaY);
	return true;
}

inline bool ViewCamera2::scaleViewRegion(const double dFactor)
{
/*
	const double dEPS = 1.0e-10;
	if (dFactor <= -dEPS || dFactor >= dEPS) zoomFactor_ /= dFactor;
	checkLimit(zoomFactor_);

	return true;
*/
	viewRegion_.changeSize(viewRegion_.getWidth() * dFactor, viewRegion_.getHeight() * dFactor);
	// update zoom factor
	return doUpdateZoomFactor();
}

inline bool ViewCamera2::restoreViewRegion()
{
	viewRegion_ = viewBound_;
	// update zoom factor
	return doUpdateZoomFactor();
}

inline bool ViewCamera2::setView(const int iX1, const int iY1, const int iX2, const int iY2)
{
	const Region2<int> rctView(iX1, iY1, iX2, iY2);
	if (!rctView.isValid())  return false;

	Point2<double> ptNC1, ptNC2;
	if (!mapWindowToCanvas(Point2<int>(rctView.left, rctView.bottom), ptNC1) || !mapWindowToCanvas(Point2<int>(rctView.right, rctView.top), ptNC2))
		return false;

	// move the center of a viewing region
	return setViewRegion(ptNC1, ptNC2);
}

inline bool ViewCamera2::moveView(const int iDeltaX, const int iDeltaY)
{
	return moveViewRegion(double(iDeltaX)/zoomFactor_, double(iDeltaY)/zoomFactor_);
}

inline bool ViewCamera2::rotateView(const int iDeltaX, const int iDeltaY)
{
	// rotate the position of eye point, the direction of sight and up direction
	return rotateViewRegion(double(iDeltaX)/zoomFactor_, double(iDeltaY)/zoomFactor_);
}

inline bool ViewCamera2::mapWindowToCanvas(const int iVCx, const int iVCy, double &rNCx, double &rNCy) const
{
	rNCx = double(iVCx - viewport_.getCenterX()) / zoomFactor_ + viewRegion_.getCenterX();

	// upward y-axis
	rNCy = double(iVCy - viewport_.getCenterY()) / zoomFactor_ + viewRegion_.getCenterY();
	//rNCy = double(iVCy - viewport_.top) / zoomFactor_ + viewRegion_.top;
	//rNCy = double(iVCy - viewport_.bottom) / zoomFactor_ + viewRegion_.bottom;

	// downward y-axis
	//rNCy = double(viewport_.getCenterY() - iVCy) / zoomFactor_ + viewRegion_.getCenterY();
	//rNCy = double(viewport_.top - iVCy) / zoomFactor_ + viewRegion_.top;
	//rNCy = double(viewport_.bottom - iVCy) / zoomFactor_ + viewRegion_.bottom;

	return true;
}

inline bool ViewCamera2::mapWindowToCanvas(const Point2<int> &rVC, Point2<double> &rNC) const
{
	rNC.x = double(rVC.x - viewport_.getCenterX()) / zoomFactor_ + viewRegion_.getCenterX();
	
	// upward y-axis
	rNC.y = double(rVC.y - viewport_.getCenterY()) / zoomFactor_ + viewRegion_.getCenterY();
	//rNC.y = double(rVC.y - viewport_.top) / zoomFactor_ + viewRegion_.top;
	//rNC.y = double(rVC.y - viewport_.bottom) / zoomFactor_ + viewRegion_.bottom;
	
	// downward y-axis
	//rNC.y = double(viewport_.getCenterY() - rVC.y) / zoomFactor_ + viewRegion_.getCenterY();
	//rNC.y = double(viewport_.top - rVC.y) / zoomFactor_ + viewRegion_.top;
	//rNC.y = double(viewport_.bottom - rVC.y) / zoomFactor_ + viewRegion_.bottom;

	return true;
}

inline bool ViewCamera2::mapCanvasToWindow(const double dNCx, const double dNCy, int &rVCx, int &rVCy) const
{
	rVCx = (int)round((dNCx - viewRegion_.getCenterX()) * zoomFactor_) + viewport_.getCenterX();

	// upward y-axis
	rVCy = (int)round((dNCy - viewRegion_.getCenterY()) * zoomFactor_) + viewport_.getCenterY();
	//rVCy = (int)round((dNCy - viewRegion_.top) * zoomFactor_) + viewport_.top;
	//rVCy = (int)round((dNCy - viewRegion_.bottom) * zoomFactor_) + viewport_.bottom;

	// downward y-axis
	//rVCy = (int)round((viewRegion_.getCenterY() - dNCy) * zoomFactor_) + viewport_.getCenterY();
	//rVCy = (int)round((viewRegion_.top - dNCy) * zoomFactor_) + viewport_.top;
	//rVCy = (int)round((viewRegion_.bottom - dNCy) * zoomFactor_) + viewport_.bottom;

    return true;
}

inline bool ViewCamera2::mapCanvasToWindow(const Point2<double> &rNC, Point2<int> &rVC) const
{
	rVC.x = (int)round((rNC.x - viewRegion_.getCenterX()) * zoomFactor_) + viewport_.getCenterX();

	// upward y-axis 
	rVC.y = (int)round((rNC.y - viewRegion_.getCenterY()) * zoomFactor_) + viewport_.getCenterY();
	//rVC.y = (int)round((rNC.y - viewRegion_.top) * zoomFactor_) + viewport_.top;
	//rVC.y = (int)round((rNC.y - viewRegion_.bottom) * zoomFactor_) + viewport_.bottom;

	// downward y-axis
	//rVC.y = (int)round((viewRegion_.getCenterY() - rNC.y) * zoomFactor_) + viewport_.getCenterY();
	//rVC.y = (int)round((viewRegion_.top - rNC.y) * zoomFactor_) + viewport_.top;
	//rVC.y = (int)round((viewRegion_.bottom - rNC.y) * zoomFactor_) + viewport_.bottom;

    return true;
}

bool ViewCamera2::getHorizontalRatio(float &rfPosRatio, float &rfWidthRatio)
// 0 <= rfPosRatio <= 1  &&  0 <= rfWidthRatio <= 1
// 0 <= rfPosRatio-rfWidthRatio/2 <= 1  &&  0 <= rfPosRatio+rfWidthRatio/2 <= 1
{
	if (!viewBound_.isValid() || !viewRegion_.isValid())
	{
		rfPosRatio = 0.0f;
		rfWidthRatio = 1.0f;
		return false;
	}
/*
	// intersect rectangles               
	const Region2<double> rctIntersection(viewBound_ & viewRegion_);

	if (rctIntersection.isValid())
	{
		rfWidthRatio = rctIntersection.getWidth() / viewBound_.getWidth();
		rfPosRatio = (rctIntersection.left - viewBound_.left) / viewBound_.getWidth();
	}
	else if (viewBound_.left < viewRegion_.left && viewRegion_.left < viewBound_.right)
	{
		rfWidthRatio = (viewBound_.right - viewRegion_.left) / viewBound_.getWidth();
		rfPosRatio = 0.0f;
	}
	else if (viewBound_.left < viewRegion_.right && viewRegion_.right < viewBound_.right)
	{
		rfWidthRatio = (viewRegion_.right - viewBound_.left) / viewBound_.getWidth();
		rfPosRatio = 1.0f - rfWidthRatio;
	}
	else
	{
		//rfWidthRatio = viewRegion_.getWidth() / viewBound_.getWidth();
		//if (rfWidthRatio > 1.0f)  rfWidthRatio = 1.0f;
		rfWidthRatio = 0.05f;
		if (viewBound_.right < viewRegion_.left)
			rfPosRatio = 1.0f - rfWidthRatio;
		else
			rfPosRatio = 0.0f;
	}
*/
	if (viewBound_.left < viewRegion_.left && viewRegion_.right < viewBound_.right)
	{
		rfWidthRatio = float(viewRegion_.getWidth() / viewBound_.getWidth());
		rfPosRatio = float((viewRegion_.left - viewBound_.left) / viewBound_.getWidth());
	}
	else
	{
		// union rectangles               
		const Region2<double> rctUnion(viewBound_ | viewRegion_);
		rfWidthRatio = float(viewRegion_.getWidth() / rctUnion.getWidth());
		rfPosRatio = float((viewRegion_.left - rctUnion.left) / rctUnion.getWidth());
	}

	return true;
}

bool ViewCamera2::getVerticalRatio(float &rfPosRatio, float &rfHeightRatio)
{
	if (!viewBound_.isValid() || !viewRegion_.isValid())
	{
		rfPosRatio = 0.0f;
		rfHeightRatio = 1.0f;
		return false;
	}
/*
	// intersect rectangles               
	const Region2<double> rctIntersection(viewBound_ & viewRegion_);

	if (rctIntersection.isValid())
	{
		rfHeightRatio = rctIntersection.getHeight() / viewBound_.getHeight();
		rfPosRatio = (rctIntersection.bottom - viewBound_.bottom) / viewBound_.getHeight();
	}
	else if (viewBound_.bottom < viewRegion_.bottom && viewRegion_.bottom < viewBound_.top)
	{
		rfHeightRatio = (viewBound_.top - viewRegion_.bottom) / viewBound_.getHeight();
		rfPosRatio = 0.0f;
	}
	else if (viewBound_.bottom < viewRegion_.top && viewRegion_.top < viewBound_.top)
	{
		rfHeightRatio = (viewRegion_.top - viewBound_.bottom) / viewBound_.getHeight();
		rfPosRatio = 1.0f - rfHeightRatio;
	}
	else
	{
		//rfHeightRatio = viewRegion_.getHeight() / viewBound_.getHeight();
		//if (viewRegion_ > 1.0f)  rfHeightRatio = 1.0f;
		rfHeightRatio = 0.05f;
		if (viewBound_.top < viewRegion_.bottom)
			rfPosRatio = 0.0f;
		else
			rfPosRatio = 1.0f - rfHeightRatio;
	}
*/
	if (viewBound_.bottom < viewRegion_.bottom && viewRegion_.top < viewBound_.top)
	{
		rfHeightRatio = float(viewRegion_.getHeight() / viewBound_.getHeight());
		rfPosRatio = float((viewRegion_.bottom - viewBound_.bottom) / viewBound_.getHeight());
	}
	else
	{
		// union rectangles               
		const Region2<double> rctUnion(viewBound_ | viewRegion_);
		rfHeightRatio = float(viewRegion_.getHeight() / rctUnion.getHeight());
		rfPosRatio = float((viewRegion_.bottom - rctUnion.bottom) / rctUnion.getHeight());
	}

	return true;
}

inline bool ViewCamera2::scrollHorizontally(const float fRatio)
// if fRatio < 0, scroll rightwards
//           > 0, scroll leftwards
// if | fRatio | < 1, scroll partial page 
//               = 1, scroll one page 
//               > 1, scroll more a page 
{
	viewRegion_.moveCenter(viewRegion_.getWidth() * double(fRatio), 0.0);
	return true;
}

inline bool ViewCamera2::scrollVertically(const float fRatio)
// if fRatio < 0, scroll downwards
//           > 0, scroll upwards
// if | fRatio | < 1, scroll partial page 
//               = 1, scroll one page 
//               > 1, scroll more a page 
{
	viewRegion_.moveCenter(0.0, viewRegion_.getHeight() * double(fRatio));
	return true;
}

inline void ViewCamera2::checkLimit(double &rdValue) const
{
	const double dEPS = 1.0e-7;
	if (rdValue < dEPS) rdValue = dEPS;
	else if (rdValue > std::numeric_limits<double>::max())
		rdValue = std::numeric_limits<double>::max();
}

inline double ViewCamera2::round(const double dValue) const
{	return (int)std::floor(dValue + 0.5);  }

Region2<double> ViewCamera2::getCurrentViewRegion() const
{
	if (!viewport_.isValid() || !viewRegion_.isValid())
		return viewRegion_;

	const double dWidthVP = double(viewport_.getWidth()), dHeightVP = double(viewport_.getHeight());
	const double dWidthVR = viewRegion_.getWidth(), dHeightVR = viewRegion_.getHeight();
    double dRatioAR = (dWidthVP / dHeightVP) / (dWidthVR / dHeightVR);
	checkLimit(dRatioAR);

	// resize viewing region
	Region2<double> rctViewRegion(viewRegion_);
	if (dRatioAR > 1.0)
		rctViewRegion.changeSize(dWidthVR * dRatioAR, dHeightVR);
	else
		rctViewRegion.changeSize(dWidthVR, dHeightVR / dRatioAR);

	return rctViewRegion;
}

}  // namespace swl
