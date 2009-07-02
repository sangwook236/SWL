#if !defined(__SWL_DATA_PACKET__PACKET_DISPATCHER__H_)
#define __SWL_DATA_PACKET__PACKET_DISPATCHER__H_ 1


#include "swl/datapacket/ExportDataPacket.h"


namespace swl {

template<class T> class GuardedBuffer;
typedef GuardedBuffer<unsigned char> GuardedByteBuffer;

//-----------------------------------------------------------------------------------
//	packet dispatcher

struct SWL_DATA_PACKET_API PacketDispatcher
{
public:
	//typedef PacketDispatcher base_type;

protected:
	PacketDispatcher();
public:
	virtual ~PacketDispatcher();

public:
	virtual bool dispatch(GuardedByteBuffer& byteBuffer) const = 0;
};

}  // namespace swl


#endif  //  __SWL_DATA_PACKET__PACKET_DISPATCHER__H_
