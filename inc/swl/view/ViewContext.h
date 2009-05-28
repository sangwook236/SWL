#if !defined(__SWL_VIEW__VIEW_CONTEXT__H_)
#define __SWL_VIEW__VIEW_CONTEXT__H_ 1


#include "swl/common/Region.h"
#include <boost/any.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
// 

struct ViewContext
{
public:
	enum EContextMode { CM_VIEWING = 0, CM_PRINTING };

public:
	struct Guard;
	friend struct Guard;
	struct Guard
	{
	public:
		//typedef Guard base_type;
		typedef ViewContext context_type;

	public:
		Guard(context_type &context)
		: context_(context)
		{  context_.activate();  }
		~Guard()
		{  context_.deactivate();  }

	private:
		context_type &context_;
	};

public:
	//typedef ViewContext base_type;
	typedef Guard guard_type;

protected:
	explicit ViewContext(const Region2<int> &drawRegion, const bool isOffScreenUsed, const EContextMode contextMode)
	: drawRegion_(drawRegion), isOffScreenUsed_(isOffScreenUsed), contextMode_(contextMode), isActivated_(false), isDrawing_(false)
	{}
public:
	virtual ~ViewContext()
	{}

private:
	ViewContext(const ViewContext &);
	ViewContext & operator=(const ViewContext &);

public:
	/// swap buffers
	virtual bool swapBuffer() = 0;
	/// resize the context
	virtual bool resize(const int x1, const int y1, const int x2, const int y2)
	{
		if (isActivated()) return false;
		drawRegion_ = Region2<int>(x1, y1, x2, y2);
		return true;
	}

	/// get the off-screen flag
	bool isOffScreenUsed() const  {  return isOffScreenUsed_;  }

	/// get the context mode
	EContextMode getContextMode() const  {  return contextMode_;  }

	/// get the context activation flag
	bool isActivated() const  {  return isActivated_;  }

	/// get the drawing flag
	bool isDrawing() const  {  return isDrawing_;  }

	/// get the native context
	virtual boost::any getNativeContext() = 0;
	virtual const boost::any getNativeContext() const = 0;

	/// get the drawing context region
	const Region2<int> & getRegion() const  {  return drawRegion_;  }

	/// set the viewing region
	void setViewingRegion(const Region2<double> &viewingRegion)  {  viewingRegion_ = viewingRegion;  }
	/// get the viewing region
	const Region2<double> & getViewingRegion() const  {  return viewingRegion_;  }

protected:
	/// set the context activation flag
	void setActivation(const bool isActivated)  {  isActivated_ = isActivated;  }

	/// set the drawing flag
	void setDrawing(const bool isDrawing)  {  isDrawing_ = isDrawing;  }

private:
	/// activate the context
	virtual bool activate() = 0;
	/// de-activate the context
	virtual bool deactivate() = 0;

protected:
	/// a drawing context region
	Region2<int> drawRegion_;
	/// a viewing region
	Region2<double> viewingRegion_;

private:
	/// a flag to check whether off-screen buffer is used
	const bool isOffScreenUsed_;

	/// a context mode
	const EContextMode contextMode_;

	/// a context activation flag
	bool isActivated_;

	/// a flag to check whether drawing is proceeding
	bool isDrawing_;
};

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_CONTEXT__H_
