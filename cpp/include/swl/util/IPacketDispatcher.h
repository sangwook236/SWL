#if !defined(__SWL_UTIL__PACKET_DISPATCHER_INTERFACE__H_)
#define __SWL_UTIL__PACKET_DISPATCHER_INTERFACE__H_ 1


namespace swl {

template<class T> class GuardedBuffer;
typedef GuardedBuffer<unsigned char> GuardedByteBuffer;

//-----------------------------------------------------------------------------------
//	packet dispatcher interface

/**
 *	@brief  ��� ���� application ���߽� ���ŵǴ� packet�� ó���ϱ� ���� abstract class (interface).
 *
 *	���ŵ� ��� data�� GuardedByteBuffer class�� ��ü�� �����ϰ�, �̸� dispatch() �Լ��� ���ڷ� �Ѱ��ش�.
 *	dispatch() �Լ� �������� GuardedByteBuffer class ��ü�� ����Ǿ� �ִ� packet�� ��� �Ծ࿡ �´��� �����ϰ�
 *	��ȿ�� packet�̶�� �̸� ó���ϵ��� �Ѵ�.
 */
struct IPacketDispatcher
{
public:
	//typedef IPacketDispatcher base_type;

public:
	/**
	 *	@brief  ���ڷ� �Ѱ��� GuardedByteBuffer ���� data�� Ȯ���Ͽ� ��� �Ծ࿡ �����ϴ��� �����ϰ� ó��.
	 *	@param[in,out]  byteBuffer  ��� �������� ���ŵ� data�� �����ϰ� �ִ� data buffer.
	 *	@return  �ùٸ� packet�� ���ŵǾ������� ��ȯ.
	 *
	 *	���ڷ� �Ѱ��� GuardedByteBuffer class�� ��ü�� ����Ǿ� �ִ� data�κ��� packet�� �̾Ƴ���
	 *	�̾Ƴ� packet�� ��� �Ծ࿡ �´��� �����Ͽ� �ùٸ� packet�̶�� dispatch�� �Ѵ�.
	 *
	 *	���ŵ� packet�� ��� �Ծ࿡ �´ٸ� true��, �׷��� �ʴٸ� false�� �����Ѵ�.
	 */
	virtual bool dispatch(GuardedByteBuffer &byteBuffer) const = 0;
};

}  // namespace swl


#endif  //  __SWL_UTIL__PACKET_DISPATCHER_INTERFACE__H_
