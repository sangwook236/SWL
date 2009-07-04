#if !defined(__SWL_COMMON__MVC_MODEL_UPDATE_CONTROLLER_H_)
#define __SWL_COMMON__MVC_MODEL_UPDATE_CONTROLLER_H_ 1


#include "swl/common/MvcController.h"


//--------------------------------------------------------------------------
//  class MvcModelUpdateController

class SWL_COMMON_API MvcModelUpdateController: public MvcController
{
/**@name Type Definitions */
//@{
public:
	typedef MvcController base_type;
//@}

/**@name Constructors, Destructor & Assignment Operator */
//@{
protected:
	MvcModelUpdateController()  {}
public:
	virtual ~MvcModelUpdateController()  {}

private:
	MvcModelUpdateController(const MvcModelUpdateController&);
	MvcModelUpdateController& operator=(const MvcModelUpdateController&);
//@}

/**@name Public Member Functions */
//@{
public:
//@}
};


#endif  // __SWL_COMMON__MVC_MODEL_UPDATE_CONTROLLER_H_
