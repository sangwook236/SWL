#if !defined(__SWL_VIEW__VIEW_CONTEXT__H_)
#define __SWL_VIEW__VIEW_CONTEXT__H_ 1


#include "swl/common/Region.h"


namespace swl {

//-----------------------------------------------------------------------------------
// 

struct ViewContext
{
public:
	//typedef ViewContext base_type;

protected:
	ViewContext(const Region2<int>& drawRegion)
	: drawRegion_(drawRegion), isActivated_(false), isDrawing_(false)
	{}
public:
	virtual ~ViewContext()
	{}

private:
	ViewContext(const ViewContext&);
	ViewContext& operator=(const ViewContext&);

public:
	/// redraw the context
	virtual bool redraw() = 0;
	/// resize the context
	virtual bool resize(const Region2<int>& drawRegion)
	{
		drawRegion_ = drawRegion;
		return true;
	}

	/// activate the context
	virtual bool activate() = 0;
	/// de-activate the context
	virtual bool deactivate() = 0;

	/// get the context activation flag
	bool isActivated() const  {  return isActivated_;  }

	/// get the drawing flag
	bool isDrawing() const  {  return isDrawing_;  }

	/// get the native context
	virtual void * getNativeContext() = 0;
	virtual const void * const getNativeContext() const = 0;

	/// get the drawing region
	const Region2<int> & getRegion() const  {  return drawRegion_;  }

protected:
	/// set the context activation flag
	void setActivation(const bool isActivated)  {  isActivated_ = isActivated;  }

	/// set the drawing flag
	void setDrawing(const bool isDrawing)  {  isDrawing_ = isDrawing;  }

protected:
	/// a drawing region
	Region2<int> drawRegion_;

private:
	/// a context activation flag
	bool isActivated_;

	/// a flag to check whether drawing
	bool isDrawing_;
};

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_CONTEXT__H_
