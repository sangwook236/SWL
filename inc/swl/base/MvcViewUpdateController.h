#if !defined(__SWL_COMMON__MVC_VIEW_UPDATE_CONTROLLER_H_)
#define __SWL_COMMON__MVC_VIEW_UPDATE_CONTROLLER_H_ 1


#include "swl/common/MvcController.h"


//--------------------------------------------------------------------------
//  class MvcViewUpdateController

class SWL_COMMON_API MvcViewUpdateController: public MvcController
{
/**@name Type Definitions */
//@{
public:
	typedef MvcController base_type;
//@}

/**@name Constructors, Destructor & Assignment Operator */
//@{
protected:
	MvcViewUpdateController()  {}
public:
	virtual ~MvcViewUpdateController()  {}

private:
	MvcViewUpdateController(const MvcViewUpdateController&);
	MvcViewUpdateController& operator=(const MvcViewUpdateController&);
//@}

/**@name Public Member Functions */
//@{
public:
//@}
};


#endif  // __SWL_COMMON__MVC_VIEW_UPDATE_CONTROLLER_H_
