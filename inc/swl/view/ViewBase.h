#if !defined(__SWL_VIEW__VIEW_BASE__H_)
#define __SWL_VIEW__VIEW_BASE__H_ 1


#include <boost/smart_ptr.hpp>
#include <stack>


namespace swl {

struct ViewContext;
class ViewCamera2;
class ViewCamera3;

//-----------------------------------------------------------------------------------
// 

struct IView
{
public:
	//typedef IView base_type;

public:
	virtual ~IView()  {}

public:
	virtual bool raiseDrawEvent(const bool isContextActivated) = 0;

	virtual bool initializeView() = 0;
	virtual bool resizeView(const int x1, const int y1, const int x2, const int y2) = 0;
};

//-----------------------------------------------------------------------------------
// 

template<typename Context, typename Camera>
struct ViewBase: public IView
{
public:
	typedef IView base_type;
	typedef Context context_type;
	typedef Camera camera_type;

public:
	ViewBase()  {}
	virtual ~ViewBase()  {}

public:
	/// context stack
	void pushContext(const boost::shared_ptr<context_type> &context)
	{
		contextStack_.push(context);
	}
	void popContext()
	{
		contextStack_.pop();
	}
	const boost::shared_ptr<context_type> & topContext() const
	{
		return contextStack_.empty() ? nullContext_ : contextStack_.top();
	}
	void clearContextStack()
	{
		while (!contextStack_.empty())
			contextStack_.pop();
	}

	size_t getContextStackSize() const
	{
		return contextStack_.size();
	}
	bool isContextStackEmpty() const
	{
		return contextStack_.empty();
	}

	///  camera stack
	void pushCamera(const boost::shared_ptr<camera_type> &camera)
	{
		cameraStack_.push(camera);
	}
	void popCamera()
	{
		cameraStack_.pop();
	}
	const boost::shared_ptr<camera_type> & topCamera() const
	{
		return cameraStack_.empty() ? nullCamera_ : cameraStack_.top();
	}
	void clearCameraStack()
	{
		while (!cameraStack_.empty())
			cameraStack_.pop();
	}

	size_t getCameraStackSize() const
	{
		return cameraStack_.size();
	}
	bool isCameraStackEmpty() const
	{
		return cameraStack_.empty();
	}

private:
	std::stack<const boost::shared_ptr<context_type> > contextStack_;
	std::stack<const boost::shared_ptr<camera_type> > cameraStack_;

	const boost::shared_ptr<context_type> nullContext_;
	const boost::shared_ptr<camera_type> nullCamera_;
};

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_BASE__H_
