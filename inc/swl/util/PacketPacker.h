#if !defined(__SWL_UTIL__PACKET_PACKER__H_ )
#define __SWL_UTIL__PACKET_PACKER__H_ 1


#include "swl/util/ExportUtil.h"
#include "swl/DisableCompilerWarning.h"
#include <list>
#include <vector>


namespace swl {

//-----------------------------------------------------------------------------------
//  byte-based packet packer

/**
 *	@brief  통신 관련 application 개발시 송신되는 packet을 구성하기 위해 제공되는 utility class.
 *
 *	통신 application에서 사용되는 송신 packet을 쉽게 만들 수 있는 기능을 제공해 주는 class이다.
 *
 *	사용 예) <br>
 *	>>  const size_t packetLen = 10; <br>
 *	>>  const bool isLittleEndian = false; <br>
 *	>>  PacketPacker packer(packetLen, isLittleEndian); <br>
 *	>> <br>
 *	>>  if (!packer.initialize()) <br>
 *	>>  { <br>
 *	>>      // do something <br>
 *	>>  } <br>
 *	>> <br>
 *	>>  // do something by using putChar(), putShort(), putInt(), ..., putText(), fillChar() <br>
 *	>> <br>
 *	>>  if (!packer.finalize()) <br>
 *	>>  { <br>
 *	>>      // do something <br>
 *	>>  } <br>
 *	>> <br>
 *	>>  // do something by using getPacket() and getPacketSize();
 */
//template<class T>
class SWL_UTIL_API PacketPacker
{
public:
	//typedef PacketPacker				base_type;
	/**
	 *	@brief  통신을 위해 사용하는 자료형으로 unsigned char를 지정.
	 *
	 *	Type definition을 이용해 value_type을 지정하므로, 향후 class 내부에서 실제적으로 사용하는 data 형이 바뀌더라도
	 *	외부 프로그램에 영향을 덜 주게 된다.
	 */
	//typedef T							value_type;
	typedef unsigned char				value_type;
	typedef std::list<value_type>		buffer_type;

public:
	/**
	 *	@brief  [ctor] contructor.
	 *	@param[in]  packetSize  통신 규약의 길이. packetSize가 0이라면 가변 길이 protocol이 됨.
	 *	@param[in]  isLittleEndian  multi-byte data가 little-endian (reverse order)으로 packing되는지 지정.
	 *
	 *	packet 생성을 위해 필요한 정보를 입력 받는다.
	 */
	PacketPacker(const size_t packetSize, const bool isLittleEndian);
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	해당 class로부터 자식 class 파생이 가능하도록 virtual로 선언되어 있다.
	 */
	virtual ~PacketPacker();

private:
	PacketPacker(const PacketPacker &);
	PacketPacker & operator=(const PacketPacker &);

public:
	/**
	 *	@brief  packet을 little-endian으로 생성하는지 big-endian으로 생성하는지 반환.
	 *	@return  packet을 little-endian으로 생성되도록 설정되었다면 true 반환.
	 *
	 *	packet을 little-endian으로 생성하도록 설정되었다면 true, 그렇지 않다면 false를 반환한다.
	 */
	bool isLittleEndian() const  {  return isLittleEndian_;  }

	/**
	 *	@brief  packet packer에 의해 생성 packet를 반환.
	 *	@param[in]  isComplete  반환해야 할 packet이 완전한 것인지 지정. true라면 해당 packet이 완성되었음을 의미.
	 *	@return  packet packer에 의해 생성된 packet pointer를 반환.
	 *
	 *	packet이 완성된 형태(isComplete이 true라면)이고 고정 길이 protocol(isFixedSize()가 true라면)인 경우
	 *	현재까지 packet packer에 저장되어 있는 packet의 길이는 ctor에서 지정한 packet size와 동일하여야 한다.
	 *	만약 이를 만족하지 않는다면, 반환되는 packet pointer는 NULL이 된다.
	 *
	 *	packet 생성을 위한 data buffer가 비어 있거나 메모리가 부족하다면 NULL을 반환한다.
	 *
	 *	반환된 packet pointer는 이 함수를 호출한 측에서 array delete (e.g. delete [] packet_pointer;)를 이용하여 삭제하여야 한다.
	 */
	value_type * getPacket(bool isComplete = true) const;
	/**
	 *	@brief  함수를 호출하는 시점까지 packet packer에 pack된 data size.
	 *	@return  packet packer에 저장되어 있는 data size를 반환.
	 *
	 *	packet을 고정 길이던지 가변 길이던지와 무관하게 호출 시점에 저장되어 있는 packet data의 길이를 반환한다.
	 */
	size_t getPacketSize() const  {  return dataBuf_.size();  }

	/**
	 *	@brief  packer packer를 초기화.
	 *	@param[in]  doesClearData  해당 값이 true라면 초기화 과정에서 현재 packet packer에 저장되어 있는 data를 삭제.
	 *	@return  초기화가 정상적으로 이루어졌다면 true를 반환.
	 *
	 *	packer packer class의 설정 값들을 초기화 시킨다.
	 */
	bool initialize(bool doesClearData = true);
	/**
	 *	@brief  packer packer의 packing 과정을 마무리 지음.
	 *	@return  마무리 과정이 정상적으로 종료되었다며 true를 반환.
	 *
	 *	packet packer에 저장된 data가 없다면 false를 반환한다.
	 *
	 *	고정 길이 protocol인 경우, 함수 호출 시점에 packet packer에 저장되어 있는 data의 길이와 ctor에서 지정한 packet 길이와 같아야 한다.
	 *	이를 만족하지 않는다면, false를 리턴하며 함수를 종료한다.
	 *
	 *	이 함수가 호출된 이후에는 packet packer에 더이상 data를 pack할 수 없다.
	 *	만약 추가적으로 data를 저장한다면 그 결과는 예측할 수 없게 된다.
	 */
	bool finalize();

	/**
	 *	@brief  packet packer에 char 형의 data를 pack함.
	 *	@param[in]  data  packet packer에 저장될 data.
	 */
	void putChar(char data);
	/**
	 *	@brief  packet packer에 short 형의 data를 pack함.
	 *	@param[in]  data  packet packer에 저장될 data.
	 */
	void putShort(short data);
	/**
	 *	@brief  packet packer에 int 형의 data를 pack함.
	 *	@param[in]  data  packet packer에 저장될 data.
	 */
	void putInt(int data);
	/**
	 *	@brief  packet packer에 long 형의 data를 pack함.
	 *	@param[in]  data  packet packer에 저장될 data.
	 */
	void putLong(long data);
	/**
	 *	@brief  packet packer에 64-bit int 형의 data를 pack함.
	 *	@param[in]  data  packet packer에 저장될 data.
	 */
	void putInt64(__int64 data);
	/**
	 *	@brief  packet packer에 float 형의 data를 pack함.
	 *	@param[in]  data  packet packer에 저장될 data.
	 */
	void putFloat(float data);
	/**
	 *	@brief  packet packer에 double 형의 data를 pack함.
	 *	@param[in]  data  packet packer에 저장될 data.
	 */
	void putDouble(double data);
	/**
	 *	@brief  packet packer에 long double 형의 data를 pack함.
	 *	@param[in]  data  packet packer에 저장될 data.
	 */
	void putLDouble(long double data);
	/**
	 *	@brief  packet packer에 c-style string(text) 형의 data를 pack함.
	 *	@param[in]  data  packet packer에 저장될 data의 pointer.
	 *	@param[in]  size  packet packer에 저장될 data의 길이.
	 */
	void putText(const char* data, const size_t size);

	/**
	 *	@brief  packet packer에 인자로 주어진 char 형의 data를 size 개수 만큼 pack함.
	 *	@param[in]  data  packet packer에 반복 저장될 data.
	 *	@param[in]  size  packet packer에 저장될 data의 개수.
	 */
	void fillChar(char data, const size_t size);

	/**
	 *	@brief  생성되는 packet이 고정 길이 protocol인지 가변 길이 protocol인지를 반환.
	 *	@return  생성되는 packet이 고정 길이 protocol이라면 true 반환.
	 *
	 *	packet을 고정 길이 protocol이라면 true, 그렇지 않다면 false를 반환한다.
	 */
	bool isFixedSize() const  {  return packetSize_ != 0;  }

private:
	// multi-byte data is packed in the reverse order
	const bool isLittleEndian_;

	// size of protocol. if packetSize_ == 0, variable-size packet
	const size_t packetSize_;

	// buffer
	buffer_type dataBuf_;
	//buffer_type::iterator itEnd_;
	buffer_type::iterator itCurr_;
};

/*
//-----------------------------------------------------------------------------------
//  byte-based packet packer

//template<class T>
class SWL_UTIL_API PacketPacker
{
public:
	//typedef PacketPacker				base_type;
	//typedef T							value_type;
	typedef unsigned char				value_type;
	typedef std::list<value_type>		buffer_type;

protected:
	PacketPacker(const bool isLittleEndian, const size_t nData, const size_t nHeader, const size_t nFooter = 0);
public:
	virtual ~PacketPacker();

private:
	PacketPacker(const PacketPacker&);
	PacketPacker & operator=(const PacketPacker&);

public:
	//
	bool isLittleEndian() const  {  return isLittleEndian_;  }

	//
	value_type * getData() const;
	size_t getDataSize() const  {  return m_bufData.size();  }

	//
	bool initialize(bool DoesClearData = true);
	bool finalize();

	//
	virtual bool setHeader(const char *header, const size_t size);
	virtual bool setFooter(const char *footer, const size_t size);

	void putChar(char data);
	void putShort(short data);
	void putInt(int data);
	void putLong(long data);
	void putInt64(__int64 data);
	void putFloat(float data);
	void putDouble(double data);
	void putLDouble(long double data);
	void putText(const char *data, const size_t size);

	void fillChar(char data, const size_t size);

	//
	bool isFixedSize() const  {  return m_nData != 0;  }
	bool isDecorated() const  {  return m_nHeader || m_nFooter;  }

private:
	//
	const bool isLittleEndian_;

	//
	const size_t m_nData;  // size of "Header + Body + Footer": if m_nData == 0, variable size data
	const size_t m_nHeader;  // size of Header: if m_nHeader == 0, no header data
	const size_t m_nFooter;  // size of Footer: if m_nFooter == 0, no footer data

	// buffer
	buffer_type m_bufData;
	//buffer_type::iterator m_itEnd;
	buffer_type::iterator m_itCurr;
};


//-----------------------------------------------------------------------------------
//  byte-based packet unpacker

//template<class T>
class SWL_UTIL_API PacketUnpacker
{
public:
	//typedef PacketUnpacker			base_type;
	//typedef T							value_type;
	typedef unsigned char				value_type;
	typedef std::vector<value_type>		buffer_type;

protected:
	PacketUnpacker(const bool isLittleEndian, const size_t nHeader = 0, const size_t nFooter = 0);
public:
	virtual ~PacketUnpacker();

private:
	PacketUnpacker(const PacketUnpacker&);
	PacketUnpacker & operator=(const PacketUnpacker&);

public:
	//
	bool isLittleEndian() const  {  return isLittleEndian_;  }

	//
	void setData(buffer_type::iterator begin, buffer_type::iterator end)
	{  m_bufData.assign(begin, end);  }
	size_t getDataSize() const  {  return m_bufData.size();  }

	//
	bool initialize();
	bool finalize();

	//
	virtual bool getHeader(char *header, const size_t size);
	virtual bool getFooter(char *footer, const size_t size);

	bool getChar(char &data) const;
	bool getShort(short &data) const;
	bool getInt(int &data) const;
	bool getLong(long &data) const;
	bool getInt64(__int64 &data) const;
	bool getFloat(float &data) const;
	bool getDouble(double &data) const;
	bool getLDouble(long double &data) const;
	bool getText(char *data, const size_t size) const;

	//
	bool isDecorated() const  {  return m_nHeader || m_nFooter;  }

private:
	//
	const bool isLittleEndian_;

	//
	//const size_t m_nData;  // size of "Header + Body + Footer"
	const size_t m_nHeader;  // size of Header: if m_nHeader == 0, no header data
	const size_t m_nFooter;  // size of Footer: if m_nFooter == 0, no footer data

	// buffer
	buffer_type m_bufData;
	buffer_type::const_iterator m_itEnd;
	mutable buffer_type::const_iterator m_itCurr;
};


//-----------------------------------------------------------------------------------
//  fixed-size packet packer

class FixedSizePacketPacker: public PacketPacker
{
public:
	typedef PacketPacker base_type;

public:
	FixedSizePacketPacker(const bool isLittleEndian, const size_t nData)
	: PacketPacker(isLittleEndian, nData, 0, 0)
	{}
protected:  // for derived class
	FixedSizePacketPacker(const bool isLittleEndian, const size_t nData, const size_t nHeader, const size_t nFooter = 0)
	: PacketPacker(isLittleEndian, nData, nHeader, nFooter)
	{}
};


//-----------------------------------------------------------------------------------
//  decorated fixed-size packet packer

class DecoratedFixedSizePacketPacker: public FixedSizePacketPacker
{
public:
	typedef FixedSizePacketPacker base_type;

public:
	DecoratedFixedSizePacketPacker(const bool isLittleEndian, const size_t nData, const size_t nHeader, const size_t nFooter = 0)
	: FixedSizePacketPacker(isLittleEndian, nData, nHeader, nFooter)
	{}
};


//-----------------------------------------------------------------------------------
//  variable-size packet packer

class VariableSizePacketPacker: public PacketPacker
{
public:
	typedef PacketPacker base_type;

public:
	VariableSizePacketPacker(const bool isLittleEndian)
	: PacketPacker(isLittleEndian, 0, 0, 0)
	{}
protected:  // for derived class
	VariableSizePacketPacker(const bool isLittleEndian, const size_t nHeader, const size_t nFooter = 0)
	: PacketPacker(isLittleEndian, 0, nHeader, nFooter)
	{}
};


//-----------------------------------------------------------------------------------
//  decorated variable-size packet packer

class DecoratedVariableSizePacketPacker: public VariableSizePacketPacker
{
public:
	typedef VariableSizePacketPacker base_type;

public:
	DecoratedVariableSizePacketPacker(const bool isLittleEndian, const size_t nHeader, const size_t nFooter = 0)
	: VariableSizePacketPacker(isLittleEndian, nHeader, nFooter)
	{}
};
*/

}  // namespace swl


#include "swl/EnableCompilerWarning.h"


#endif  // __SWL_UTIL__PACKET_PACKER__H_ 
