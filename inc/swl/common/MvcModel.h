#if !defined(__SWL_COMMON__MVC_MODEL__H_)
#define __SWL_COMMON__MVC_MODEL__H_ 1


#include "swl/common/IObserver.h"


namespace swl {

//--------------------------------------------------------------------------
//  class MvcModel

class SWL_COMMON_API MvcModel: public IObserver
{
public:
	typedef IObserver base_type;

protected:
	MvcModel()  {}
public:
	virtual ~MvcModel()  {}

private:
	MvcModel(const MvcModel&);
	MvcModel& operator=(const MvcModel&);

public:
};

}  // namespace swl


#endif  // __SWL_COMMON__MVC_MODEL__H_
