#include "swl/Config.h"
#include "swl/base/IObserver.h"


#if defined(_DEBUG)
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
