#if !defined(__SWL_UTIL__PACKET_DISPATCHER_INTERFACE__H_)
#define __SWL_UTIL__PACKET_DISPATCHER_INTERFACE__H_ 1


namespace swl {

template<class T> class GuardedBuffer;
typedef GuardedBuffer<unsigned char> GuardedByteBuffer;

//-----------------------------------------------------------------------------------
//	packet dispatcher interface

/**
 *	@brief  통신 관련 application 개발시 수신되는 packet을 처리하기 위한 abstract class (interface).
 *
 *	수신된 통신 data를 GuardedByteBuffer class의 객체에 저장하고, 이를 dispatch() 함수에 인자로 넘겨준다.
 *	dispatch() 함수 내에서는 GuardedByteBuffer class 객체에 저장되어 있는 packet이 통신 규약에 맞는지 점검하고
 *	유효한 packet이라면 이를 처리하도록 한다.
 */
struct IPacketDispatcher
{
public:
	//typedef IPacketDispatcher base_type;

public:
	/**
	 *	@brief  인자로 넘겨진 GuardedByteBuffer 내의 data를 확인하여 통신 규약에 부합하는지 점검하고 처리.
	 *	@param[in,out]  byteBuffer  통신 과정에서 수신된 data를 저장하고 있는 data buffer.
	 *	@return  올바른 packet이 수신되었는지를 반환.
	 *
	 *	인자로 넘겨진 GuardedByteBuffer class의 객체에 저장되어 있는 data로부터 packet를 뽑아내고
	 *	뽑아낸 packet이 통신 규약에 맞는지 점검하여 올바른 packet이라면 dispatch를 한다.
	 *
	 *	수신된 packet이 통신 규약에 맞다면 true를, 그렇지 않다면 false를 리턴한다.
	 */
	virtual bool dispatch(GuardedByteBuffer &byteBuffer) const = 0;
};

}  // namespace swl


#endif  //  __SWL_UTIL__PACKET_DISPATCHER_INTERFACE__H_
