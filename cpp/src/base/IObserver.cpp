#include "swl/Config.h"
#include "swl/base/IObserver.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class IObserver

IObserver::~IObserver()
{
}

}  // namespace swl
