#if !defined(__SWL_BASE__MVC_VIEW_H_)
#define __SWL_BASE__MVC_VIEW_H_ 1


#include "swl/base/IObserver.h"


namespace swl {

//--------------------------------------------------------------------------
//  class MvcView

class SWL_BASE_API MvcView: public IObserver
{
public:
	typedef IObserver base_type;

protected:
	MvcView()  {}
public:
	virtual ~MvcView()  {}

private:
	MvcView(const MvcView &);
	MvcView& operator=(const MvcView &);

public:
};

}  // namespace swl


#endif  // __SWL_BASE__MVC_VIEW_H_
