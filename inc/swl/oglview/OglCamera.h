#if !defined(__SWL_OGL_VIEW__OGL_CAMERA__H_)
#define __SWL_OGL_VIEW__OGL_CAMERA__H_ 1


#include "swl/oglview/ExportOglView.h"
#include "swl/view/ViewCamera3.h"


namespace swl {

//-----------------------------------------------------------------------------------------
//   class OglCamera

class SWL_OGL_VIEW_API OglCamera: public ViewCamera3
{
public :
	typedef ViewCamera3 base_type;

public :
	OglCamera();
	OglCamera(const OglCamera &rhs);
	virtual ~OglCamera();
	
	OglCamera & operator=(const OglCamera &rhs);

public :
	/// virtual copy constructor
	/*virtual*/ ViewCamera2 * cloneCamera() const
	{  return new OglCamera(*this);  }

	///*virtual*/ void Write(std::ostream &stream);
	///*virtual*/ void Read(std::istream &stream);

	/// look at the view region from the eye position along the eye direction and up direction
	/*virtual*/ void lookAt();

	///
	double calcResizingRatio() const;

protected :
	///
	/*virtual*/ bool doUpdateFrustum();
	/*virtual*/ bool doUpdateViewport();

	/// projection transformation: an eye coordinates  ==>  a clip coordinates
	/*virtual*/ bool doMapEyeToClip(const double ptEye[3], double ptClip[3]) const;
	/// viewport transformation: a clip coordinates  ==>  a window coordinates
	/*virtual*/ bool doMapClipToWindow(const double ptClip[3], double ptWin[3]) const;
	/// inverse viewport transformation: a window coordinates  ==>  a clip coordinates
	/*virtual*/ bool doMapWindowToClip(const double ptWin[3], double ptClip[3]) const;
	/// inverse projection transformation: a clip coordinates  ==>  an eye coordinates
	/*virtual*/ bool doMapClipToEye(const double ptClip[3], double ptEye[3]) const;

private :
	///
	//void read20021008(::std::istream& stream);
};

}  // namespace swl


#endif  //  __SWL_OGL_VIEW__OGL_CAMERA__H_
