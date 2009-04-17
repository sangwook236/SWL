#if !defined(__SWL_COMMON__MVC_VIEW_H_)
#define __SWL_COMMON__MVC_VIEW_H_ 1


#include "swl/common/IObserver.h"


namespace swl {

//--------------------------------------------------------------------------
//  class MvcView

class SWL_COMMON_API MvcView: public IObserver
{
public:
	typedef IObserver base_type;

protected:
	MvcView()  {}
public:
	virtual ~MvcView()  {}

private:
	MvcView(const MvcView&);
	MvcView& operator=(const MvcView&);

public:
};

}  // namespace swl


#endif  // __SWL_COMMON__MVC_VIEW_H_
