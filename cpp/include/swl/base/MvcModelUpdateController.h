#if !defined(__SWL_BASE__MVC_MODEL_UPDATE_CONTROLLER_H_)
#define __SWL_BASE__MVC_MODEL_UPDATE_CONTROLLER_H_ 1


#include "swl/base/MvcController.h"


//--------------------------------------------------------------------------
//  class MvcModelUpdateController

class SWL_BASE_API MvcModelUpdateController: public MvcController
{
public:
	typedef MvcController base_type;

protected:
	MvcModelUpdateController()  {}
public:
	virtual ~MvcModelUpdateController()  {}

private:
	MvcModelUpdateController(const MvcModelUpdateController &);
	MvcModelUpdateController& operator=(const MvcModelUpdateController &);

public:
};


#endif  // __SWL_BASE__MVC_MODEL_UPDATE_CONTROLLER_H_
