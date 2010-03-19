#if !defined(__SWL_GL_UTIL__DISPLAY_LIST_CALLABE_INTERFACE__H_)
#define __SWL_GL_UTIL__DISPLAY_LIST_CALLABE_INTERFACE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct IGLDisplayListCallable: mix-in style class

struct IGLDisplayListCallable
{
protected:
	virtual ~IGLDisplayListCallable()  {}

public:
	//
	virtual bool createDisplayList() = 0;
	virtual void callDisplayList() const = 0;
};

}  // namespace swl


#endif  // __SWL_GL_UTIL__DISPLAY_LIST_CALLABE_INTERFACE__H_
