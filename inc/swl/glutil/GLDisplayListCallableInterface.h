#if !defined(__SWL_GL_UTIL__DISPLAY_LIST_CALLABE_INTERFACE__H_)
#define __SWL_GL_UTIL__DISPLAY_LIST_CALLABE_INTERFACE__H_ 1


#include "swl/glutil/ExportGLUtil.h"
#include <stack>


namespace swl {

//-----------------------------------------------------------------------------------------
// struct GLDisplayListCallableInterface: mix-in style class

struct SWL_GL_UTIL_API GLDisplayListCallableInterface
{
public:
	struct Guard;
	friend struct Guard;
	struct Guard
	{
	public:
		//typedef Guard base_type;
		typedef GLDisplayListCallableInterface guardable_type;

	public:
		Guard(guardable_type &guardable)
		: guardable_(guardable)
		{  guardable_.pushDisplayList();  }
		~Guard()
		{  guardable_.popDisplayList();  }

	private:
		guardable_type &guardable_;
	};

public:
	//typedef GLDisplayListCallableInterface base_type;
	typedef Guard guard_type;

protected:
	GLDisplayListCallableInterface(const unsigned int displayListCount);
	GLDisplayListCallableInterface(const GLDisplayListCallableInterface &rhs);
	virtual ~GLDisplayListCallableInterface();

private:
	GLDisplayListCallableInterface & operator=(const GLDisplayListCallableInterface &rhs);

public:
	//
	virtual bool createDisplayList() = 0;
	virtual void callDisplayList() const = 0;

	//
	bool pushDisplayList();
	bool popDisplayList();
	bool isDisplayListUsed() const  {  return !displayListNameBaseStack_.empty() && 0u != displayListNameBaseStack_.top();  }

	unsigned int getDisplayListNameBase() const  {  return displayListNameBaseStack_.empty() ? 0u : displayListNameBaseStack_.top();  }
	unsigned int getDisplayListCount() const  {  return displayListCount_;  }

private:
	//
	std::stack<unsigned int> displayListNameBaseStack_;
	const unsigned int displayListCount_;
};

}  // namespace swl


#endif  // __SWL_GL_UTIL__DISPLAY_LIST_CALLABE_INTERFACE__H_
