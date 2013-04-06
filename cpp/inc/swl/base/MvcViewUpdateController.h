#if !defined(__SWL_BASE__MVC_VIEW_UPDATE_CONTROLLER_H_)
#define __SWL_BASE__MVC_VIEW_UPDATE_CONTROLLER_H_ 1


#include "swl/base/MvcController.h"


//--------------------------------------------------------------------------
//  class MvcViewUpdateController

class SWL_BASE_API MvcViewUpdateController: public MvcController
{
public:
	typedef MvcController base_type;

protected:
	MvcViewUpdateController()  {}
public:
	virtual ~MvcViewUpdateController()  {}

private:
	MvcViewUpdateController(const MvcViewUpdateController &);
	MvcViewUpdateController& operator=(const MvcViewUpdateController &);

public:
};


#endif  // __SWL_BASE__MVC_VIEW_UPDATE_CONTROLLER_H_
