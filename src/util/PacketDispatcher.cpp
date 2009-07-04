#include "swl/Config.h"
#include "swl/util/PacketDispatcher.h"
#include "swl/util/GuardedBuffer.h"


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl  {

//-----------------------------------------------------------------------------------
//	packet dispatcher

PacketDispatcher::PacketDispatcher()
{
}

PacketDispatcher::~PacketDispatcher()
{
}

}  // namespace swl
