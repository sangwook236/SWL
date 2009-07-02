#if !defined(__SWL_DATA_PACKET__PACKET_PACKER__H_ )
#define __SWL_DATA_PACKET__PACKET_PACKER__H_ 1


#include "swl/datapacket/ExportDataPacket.h"
#include "swl/CompilerWarning.h"
#include <list>
#include <vector>


namespace swl {

//-----------------------------------------------------------------------------------
//  byte data converter

SWL_DATA_PACKET_API unsigned char convDec2Ascii(const unsigned char dec);
SWL_DATA_PACKET_API unsigned char convAscii2Dec(const unsigned char ascii);
SWL_DATA_PACKET_API unsigned char convHex2Ascii(const unsigned char hex, const bool doesConvToUpperCase = true);
SWL_DATA_PACKET_API unsigned char convAscii2Hex(const unsigned char ascii);


//-----------------------------------------------------------------------------------
//  byte-based packet packer

//template<class T>
class SWL_DATA_PACKET_API PacketPacker
{
public:
	//typedef PacketPacker				base_type;
	//typedef T							value_type;
	typedef unsigned char				value_type;
	typedef std::list<value_type>		buffer_type;

public:
	PacketPacker(const size_t packetSize, const bool isLittleEndian);
	virtual ~PacketPacker();

private:
	PacketPacker(const PacketPacker&);
	PacketPacker & operator=(const PacketPacker&);

public:
	//
	bool isLittleEndian() const  {  return isLittleEndian_;  }

	//
	value_type * getPacket(bool isComplete = true) const;
	size_t getPacketSize() const  {  return dataBuf_.size();  }

	//
	bool initialize(bool doesClearData = true);
	bool finalize();

	//
	void putChar(char data);
	void putShort(short data);
	void putInt(int data);
	void putLong(long data);
	void putInt64(__int64 data);
	void putFloat(float data);
	void putDouble(double data);
	void putLDouble(long double data);
	void putText(const char* data, const size_t size);

	void fillChar(char data, const size_t size);

	//
	bool isFixedSize() const  {  return packetSize_ != 0;  }

private:
	// multi-byte data is packed in the reverse order
	const bool isLittleEndian_;

	//
	const size_t packetSize_;  // size of protocol. if packetSize_ == 0, variable-size packet

	// buffer
	buffer_type dataBuf_;
	//buffer_type::iterator itEnd_;
	buffer_type::iterator itCurr_;
};

/*
//-----------------------------------------------------------------------------------
//  byte-based packet packer

//template<class T>
class SWL_DATA_PACKET_API PacketPacker
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
class SWL_DATA_PACKET_API PacketUnpacker
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


#endif  // __SWL_DATA_PACKET__PACKET_PACKER__H_ 
