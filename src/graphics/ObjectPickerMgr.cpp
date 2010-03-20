#include "swl/Config.h"
#include "swl/graphics/ObjectPickerMgr.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// class ObjectPickerMgr

/*static*/ ObjectPickerMgr *ObjectPickerMgr::singleton_ = NULL;

/*static*/ ObjectPickerMgr & ObjectPickerMgr::getInstance()
{
	if (NULL == singleton_)
		singleton_ = new ObjectPickerMgr();

	return *singleton_;
}

/*static*/ void ObjectPickerMgr::clearInstance()
{
	if (singleton_)
	{
		delete singleton_;
		singleton_ = NULL;
	}
}

}  // namespace swl
