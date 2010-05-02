#include "swl/Config.h"
#include "swl/graphics/ObjectPickerMgr.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// class ObjectPickerMgr

/*static*/ boost::scoped_ptr<ObjectPickerMgr> ObjectPickerMgr::singleton_;

/*static*/ ObjectPickerMgr & ObjectPickerMgr::getInstance()
{
	if (!singleton_)
		singleton_.reset(new ObjectPickerMgr());

	return *singleton_;
}

/*static*/ void ObjectPickerMgr::clearInstance()
{
	singleton_.reset();
}

}  // namespace swl
