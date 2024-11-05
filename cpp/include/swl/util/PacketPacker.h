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
 *	@brief  ��� ���� application ���߽� �۽ŵǴ� packet�� �����ϱ� ���� �����Ǵ� utility class.
 *
 *	��� application���� ���Ǵ� �۽� packet�� ���� ���� �� �ִ� ����� ������ �ִ� class�̴�.
 *
 *	��� ��) <br>
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
	 *	@brief  ����� ���� ����ϴ� �ڷ������� unsigned char�� ����.
	 *
	 *	Type definition�� �̿��� value_type�� �����ϹǷ�, ���� class ���ο��� ���������� ����ϴ� data ���� �ٲ����
	 *	�ܺ� ���α׷��� ������ �� �ְ� �ȴ�.
	 */
	//typedef T							value_type;
	typedef unsigned char				value_type;
	typedef std::list<value_type>		buffer_type;

public:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  packetSize  ��� �Ծ��� ����. packetSize�� 0�̶�� ���� ���� protocol�� ��.
	 *	@param[in]  isLittleEndian  multi-byte data�� little-endian (reverse order)���� packing�Ǵ��� ����.
	 *
	 *	packet ������ ���� �ʿ��� ������ �Է� �޴´�.
	 */
	PacketPacker(const std::size_t packetSize, const bool isLittleEndian);
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	�ش� class�κ��� �ڽ� class �Ļ��� �����ϵ��� virtual�� ����Ǿ� �ִ�.
	 */
	virtual ~PacketPacker();

private:
	PacketPacker(const PacketPacker &);
	PacketPacker & operator=(const PacketPacker &);

public:
	/**
	 *	@brief  packet�� little-endian���� �����ϴ��� big-endian���� �����ϴ��� ��ȯ.
	 *	@return  packet�� little-endian���� �����ǵ��� �����Ǿ��ٸ� true ��ȯ.
	 *
	 *	packet�� little-endian���� �����ϵ��� �����Ǿ��ٸ� true, �׷��� �ʴٸ� false�� ��ȯ�Ѵ�.
	 */
	bool isLittleEndian() const  {  return isLittleEndian_;  }

	/**
	 *	@brief  packet packer�� ���� ���� packet�� ��ȯ.
	 *	@param[in]  isComplete  ��ȯ�ؾ� �� packet�� ������ ������ ����. true��� �ش� packet�� �ϼ��Ǿ����� �ǹ�.
	 *	@return  packet packer�� ���� ������ packet pointer�� ��ȯ.
	 *
	 *	packet�� �ϼ��� ����(isComplete�� true���)�̰� ���� ���� protocol(isFixedSize()�� true���)�� ���
	 *	������� packet packer�� ����Ǿ� �ִ� packet�� ���̴� ctor���� ������ packet size�� �����Ͽ��� �Ѵ�.
	 *	���� �̸� �������� �ʴ´ٸ�, ��ȯ�Ǵ� packet pointer�� NULL�� �ȴ�.
	 *
	 *	packet ������ ���� data buffer�� ��� �ְų� �޸𸮰� �����ϴٸ� NULL�� ��ȯ�Ѵ�.
	 *
	 *	��ȯ�� packet pointer�� �� �Լ��� ȣ���� ������ array delete (e.g. delete [] packet_pointer;)�� �̿��Ͽ� �����Ͽ��� �Ѵ�.
	 */
	value_type * getPacket(bool isComplete = true) const;
	/**
	 *	@brief  �Լ��� ȣ���ϴ� �������� packet packer�� pack�� data size.
	 *	@return  packet packer�� ����Ǿ� �ִ� data size�� ��ȯ.
	 *
	 *	packet�� ���� ���̴��� ���� ���̴����� �����ϰ� ȣ�� ������ ����Ǿ� �ִ� packet data�� ���̸� ��ȯ�Ѵ�.
	 */
	std::size_t getPacketSize() const  {  return dataBuf_.size();  }

	/**
	 *	@brief  packer packer�� �ʱ�ȭ.
	 *	@param[in]  doesClearData  �ش� ���� true��� �ʱ�ȭ �������� ���� packet packer�� ����Ǿ� �ִ� data�� ����.
	 *	@return  �ʱ�ȭ�� ���������� �̷�����ٸ� true�� ��ȯ.
	 *
	 *	packer packer class�� ���� ������ �ʱ�ȭ ��Ų��.
	 */
	bool initialize(bool doesClearData = true);
	/**
	 *	@brief  packer packer�� packing ������ ������ ����.
	 *	@return  ������ ������ ���������� ����Ǿ��ٸ� true�� ��ȯ.
	 *
	 *	packet packer�� ����� data�� ���ٸ� false�� ��ȯ�Ѵ�.
	 *
	 *	���� ���� protocol�� ���, �Լ� ȣ�� ������ packet packer�� ����Ǿ� �ִ� data�� ���̿� ctor���� ������ packet ���̿� ���ƾ� �Ѵ�.
	 *	�̸� �������� �ʴ´ٸ�, false�� �����ϸ� �Լ��� �����Ѵ�.
	 *
	 *	�� �Լ��� ȣ��� ���Ŀ��� packet packer�� ���̻� data�� pack�� �� ����.
	 *	���� �߰������� data�� �����Ѵٸ� �� ����� ������ �� ���� �ȴ�.
	 */
	bool finalize();

	/**
	 *	@brief  packet packer�� char ���� data�� pack��.
	 *	@param[in]  data  packet packer�� ����� data.
	 */
	void putChar(char data);
	/**
	 *	@brief  packet packer�� short ���� data�� pack��.
	 *	@param[in]  data  packet packer�� ����� data.
	 */
	void putShort(short data);
	/**
	 *	@brief  packet packer�� int ���� data�� pack��.
	 *	@param[in]  data  packet packer�� ����� data.
	 */
	void putInt(int data);
	/**
	 *	@brief  packet packer�� long ���� data�� pack��.
	 *	@param[in]  data  packet packer�� ����� data.
	 */
	void putLong(long data);
	/**
	 *	@brief  packet packer�� 64-bit int ���� data�� pack��.
	 *	@param[in]  data  packet packer�� ����� data.
	 */
#if defined(__GNUC__)
	void putInt64(long long data);
#elif defined(_MSC_VER)
	void putInt64(__int64 data);
#endif
	/**
	 *	@brief  packet packer�� float ���� data�� pack��.
	 *	@param[in]  data  packet packer�� ����� data.
	 */
	void putFloat(float data);
	/**
	 *	@brief  packet packer�� double ���� data�� pack��.
	 *	@param[in]  data  packet packer�� ����� data.
	 */
	void putDouble(double data);
	/**
	 *	@brief  packet packer�� long double ���� data�� pack��.
	 *	@param[in]  data  packet packer�� ����� data.
	 */
	void putLDouble(long double data);
	/**
	 *	@brief  packet packer�� c-style string(text) ���� data�� pack��.
	 *	@param[in]  data  packet packer�� ����� data�� pointer.
	 *	@param[in]  size  packet packer�� ����� data�� ����.
	 */
	void putText(const char* data, const std::size_t size);

	/**
	 *	@brief  packet packer�� ���ڷ� �־��� char ���� data�� size ���� ��ŭ pack��.
	 *	@param[in]  data  packet packer�� �ݺ� ����� data.
	 *	@param[in]  size  packet packer�� ����� data�� ����.
	 */
	void fillChar(char data, const std::size_t size);

	/**
	 *	@brief  �����Ǵ� packet�� ���� ���� protocol���� ���� ���� protocol������ ��ȯ.
	 *	@return  �����Ǵ� packet�� ���� ���� protocol�̶�� true ��ȯ.
	 *
	 *	packet�� ���� ���� protocol�̶�� true, �׷��� �ʴٸ� false�� ��ȯ�Ѵ�.
	 */
	bool isFixedSize() const  {  return packetSize_ != 0;  }

private:
	// multi-byte data is packed in the reverse order
	const bool isLittleEndian_;

	// size of protocol. if packetSize_ == 0, variable-size packet
	const std::size_t packetSize_;

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
