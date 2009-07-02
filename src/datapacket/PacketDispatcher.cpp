#include "swl/datapacket/PacketDispatcher.h"
#include "swl/util/GuardedBuffer.h"


#if defined(_MSC_VER) && defined(_DEBUG)
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
