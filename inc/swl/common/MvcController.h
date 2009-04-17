#if !defined(__SWL_COMMON__MVC_CONTROLLER__H_)
#define __SWL_COMMON__MVC_CONTROLLER__H_ 1


#include "swl/common/INotifier.h"


namespace swl {

//--------------------------------------------------------------------------
//  class MvcController

class MvcController: public Notifier
{
public:
	typedef Notifier base_type;

protected:
	MvcController()
	: base_type()
	{}
public:
	virtual ~MvcController()  {}

private:
	MvcController(const MvcController&);
	MvcController& operator=(const MvcController&);

public:
};

}  // namespace swl


#endif  // __SWL_COMMON__MVC_CONTROLLER__H_
