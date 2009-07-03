#if !defined(__SWL_UTIL__PACKET_DISPATCHER__H_)
#define __SWL_UTIL__PACKET_DISPATCHER__H_ 1


#include "swl/util/ExportUtil.h"


namespace swl {

template<class T> class GuardedBuffer;
typedef GuardedBuffer<unsigned char> GuardedByteBuffer;

//-----------------------------------------------------------------------------------
//	packet dispatcher

struct SWL_UTIL_API PacketDispatcher
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


#endif  //  __SWL_UTIL__PACKET_DISPATCHER__H_
