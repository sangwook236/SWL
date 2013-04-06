#include "swl/common/IObserver.h"


#if defined(_MSC_VER) && defined(_DEBUG)
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
