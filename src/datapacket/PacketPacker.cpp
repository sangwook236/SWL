#include "swl/datapacket/PacketPacker.h"
#include <iterator>
#include <algorithm>


#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//  byte data converter

unsigned char convDec2Ascii(const unsigned char dec)
{
	//AFX_MANAGE_STATE(AfxGetStaticModuleState());

	if (0x0 <= dec && dec <= 0x9)
		return dec + '0';
	else return -1;
}

unsigned char convAscii2Dec(const unsigned char ascii)
{
	//AFX_MANAGE_STATE(AfxGetStaticModuleState());

	if ('0' <= ascii && ascii <= '9')
		return ascii - '0';
	else return -1;
}

unsigned char convHex2Ascii(const unsigned char hex, const bool doesConvToUpperCase /*= true*/)
{
	//AFX_MANAGE_STATE(AfxGetStaticModuleState());

	if (0x0 <= hex && hex <= 0x9)
		return hex + '0';
	//else if (0xa <= hex && hex <= 0xf)
	else if (0xA <= hex && hex <= 0xF)
		return hex - 0xA + (doesConvToUpperCase ? 'A' : 'a');
	else return -1;
}

unsigned char convAscii2Hex(const unsigned char ascii)
{
	//AFX_MANAGE_STATE(AfxGetStaticModuleState());

	if ('0' <= ascii && ascii <= '9')
		return ascii - '0';
	else if ('a' <= ascii && ascii <= 'f')
		return ascii - 'a' + 10;
	else if ('A' <= ascii && ascii <= 'F')
		return ascii - 'A' + 10;
	else return -1;
}


//-----------------------------------------------------------------------------------
//  byte-based packet packer

/* Usage:
	const size_t PACKET_SIZE = 20;
	PacketPacker packer(PACKET_SIZE, false);
	
	bool bStatus = packer.initialize();

	// header: prompt (1 byte)
	packer.putChar(isLongResponseMsg_ ? DIO_SERIAL_PROTOCOL__PROMPT_LONG_RESPONSE_MSG : DIO_SERIAL_PROTOCOL__PROMPT_SHORT_RESPONSE_MSG);

	// header: address (1 byte)
	packer.putChar((char &)moduleAddress_);

	// header: command (2 or 3 bytes)
	packer.putText(commandStr_.c_str(), commandStr_.length());

	// data
	doMakeData(packer);

	// footer: checksum (2 bytes)
	{
		const boost::scoped_array<unsigned char> packetData(packer.getPacket(false));
		const size_t packetDataSize = packer.getDataSize();

		int sum = 0;
		const unsigned char * packet = packetData.get();
		for (size_t i = 1; i < packetDataSize; ++i)
			sum += packet[i];
		packer.putChar((char)convHex2Ascii((unsigned char)((sum >> 4) & 0x0000000F)));
		packer.putChar((char)convHex2Ascii((unsigned char)(sum & 0x0000000F)));
	}

	// footer: delimiter (1 byte)
	packer.putChar(DIO_SERIAL_PROTOCOL__DELIMITER);

	bStatus &= packer.finalize();
	if (!bStatus) return false;

	//
	const boost::shared_array<unsigned char> packetData(packer.getPacket());
	protocolCnt_ = packer.getPacketSize();

	const unsigned char * packet = packetData.get();

	return protocolCnt_ == PACKET_SIZE;
*/

PacketPacker::PacketPacker(const size_t packetSize, const bool isLittleEndian)
: packetSize_(packetSize), isLittleEndian_(isLittleEndian),
  dataBuf_()
{
}

PacketPacker::~PacketPacker()
{
}

bool PacketPacker::initialize(bool doesClearData /*= true*/)
{
	if (doesClearData) dataBuf_.clear();
	itCurr_ = dataBuf_.begin();
	return true;
}

bool PacketPacker::finalize()
{
	if (dataBuf_.empty()) return false;

	if (isFixedSize())
	{
		if (dataBuf_.size() != packetSize_) return false;
	}
	
	itCurr_ = dataBuf_.begin();
	return true;
}

PacketPacker::value_type * PacketPacker::getPacket(bool isComplete /*= true*/) const
{
	if (dataBuf_.empty()) return 0L;

	size_t size = dataBuf_.size();
	if (isComplete && isFixedSize() && size != packetSize_) return 0L;

	value_type* data = new value_type [size];
	if (!data) return 0L;

	//
	std::copy(dataBuf_.begin(), dataBuf_.end(), data);

	return data;
}

void PacketPacker::putChar(char data)
{
	//--S [] 2004/06/30 : Sang-Wook Lee
	//dataBuf_.push_back(data);
	std::inserter(dataBuf_, itCurr_) = data;
	//++itCurr_;  // don't need because of a mechanism of inserter
	//--E [] 2004/06/30
}

void PacketPacker::putShort(short data)
{
	if (isLittleEndian_)
		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(short), std::inserter(dataBuf_, itCurr_));
	else
	{
		std::copy((value_type *)&data, (value_type *)&data + sizeof(short), std::inserter(dataBuf_, itCurr_));
		//std::advance(itCurr_, sizeof(short));  // don't need because of a mechanism of inserter
	}
}

void PacketPacker::putInt(int data)
{
	if (isLittleEndian_)
		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(int), std::inserter(dataBuf_, itCurr_));
	else
	{
		std::copy((value_type *)&data, (value_type *)&data + sizeof(int), std::inserter(dataBuf_, itCurr_));
		//std::advance(itCurr_, sizeof(int));  // don't need because of a mechanism of inserter
	}
}

void PacketPacker::putLong(long data)
{
	if (isLittleEndian_)
		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(long), std::inserter(dataBuf_, itCurr_));
	else
	{
		std::copy((value_type *)&data, (value_type *)&data + sizeof(long), std::inserter(dataBuf_, itCurr_));
		//std::advance(itCurr_, sizeof(long));  // don't need because of a mechanism of inserter
	}
}

void PacketPacker::putInt64(__int64 data)
{
	if (isLittleEndian_)
		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(__int64), std::inserter(dataBuf_, itCurr_));
	else
	{
		std::copy((value_type *)&data, (value_type *)&data + sizeof(__int64), std::inserter(dataBuf_, itCurr_));
		//std::advance(itCurr_, sizeof(__int64));  // don't need because of a mechanism of inserter
	}
}

void PacketPacker::putFloat(float data)
{
	if (isLittleEndian_)
		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(float), std::inserter(dataBuf_, itCurr_));
	else
	{
		std::copy((value_type *)&data, (value_type *)&data + sizeof(float), std::inserter(dataBuf_, itCurr_));
		//std::advance(itCurr_, sizeof(float));  // don't need because of a mechanism of inserter
	}
}

void PacketPacker::putDouble(double data)
{
	if (isLittleEndian_)
		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(double), std::inserter(dataBuf_, itCurr_));
	else
	{
		std::copy((value_type *)&data, (value_type *)&data + sizeof(double), std::inserter(dataBuf_, itCurr_));
		//std::advance(itCurr_, sizeof(double));  // don't need because of a mechanism of inserter
	}
}

void PacketPacker::putLDouble(long double data)
{
	if (isLittleEndian_)
		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(long double), std::inserter(dataBuf_, itCurr_));
	else
	{
		std::copy((value_type *)&data, (value_type *)&data + sizeof(long double), std::inserter(dataBuf_, itCurr_));
		//std::advance(itCurr_, sizeof(long double));  // don't need because of a mechanism of inserter
	}
}

void PacketPacker::putText(const char* data, const size_t size)
{
	// FIXME [check] >>
/*
	if (isLittleEndian_)
		std::reverse_copy(data, data + size, std::inserter(dataBuf_, itCurr_));
	else
	{
//		//dataBuf_.insert(dataBuf_.end(), data, data + size);
//		std::copy(data, data + size, std::back_inserter(dataBuf_));
		std::copy((value_type *)data, (value_type *)data + size, std::inserter(dataBuf_, itCurr_));
		//std::advance(itCurr_, size);  // don't need because of a mechanism of inserter
	}
*/
	std::copy((value_type *)data, (value_type *)data + size, std::inserter(dataBuf_, itCurr_));
}

void PacketPacker::fillChar(char data, const size_t size)
{
	dataBuf_.insert(itCurr_, size, data);
}


////-----------------------------------------------------------------------------------
////  byte-based packet packer
//
//PacketPacker::PacketPacker(const bool isLittleEndian, const size_t nData, const size_t nHeader, const size_t nFooter /*= 0*/)
//: isLittleEndian_(isLittleEndian),
//  m_nData(nData), m_nHeader(nHeader), m_nFooter(nFooter),
//  //m_bufData(nData ? nData : m_nHeader)
//  m_bufData()
//{
//	if (isFixedSize() && m_nData <= m_nHeader + m_nFooter)
//	{
//		// TODO: need to add
//		// error
//	}
//}
//
//PacketPacker::~PacketPacker()
//{
//}
//
//bool PacketPacker::initialize(bool doesClearData /*= true*/)
//{
///*	
//	// DELETEME: don't need
//	size_t size = m_bufData.size();
//	if (size > m_nFooter)
//	{
//		//--S [] 2004/06/30 : Sang-Wook Lee
//		// data body    getFooter() getText()  m_itCurr == m_itEnd  error 
//		m_itEnd = m_bufData.end();
//		//m_itEnd = m_bufData.begin();
//		//std::advance(m_itEnd, size - m_nFooter);
//		//--E [] 2004/06/30
//		m_itCurr = m_bufData.begin();
//		return true;
//	}
//	else return false;
//*/
//	if (doesClearData) m_bufData.clear();
//	m_itCurr = m_bufData.begin();
//	return true;
//}
//
//bool PacketPacker::finalize()
//{
//	if (m_bufData.empty()) return false;
//
//	if (isFixedSize())
//	{
//		if (m_bufData.size() != m_nData) return false;
//	}
//	
//	m_itCurr = m_bufData.begin();
//	if (m_nHeader) std::advance(m_itCurr, m_nHeader);
//	return true;
//}
//
//PacketPacker::value_type * PacketPacker::getData() const
//{
//	if (m_bufData.empty()) return 0L;
//
//	size_t size = m_bufData.size();
//	if (isFixedSize() && size != m_nData) return 0L;
//
//	value_type *data = new value_type [size];
//	if (!data) return 0L;
//
//	//
//	std::copy(m_bufData.begin(), m_bufData.end(), data);
//
//	return data;
//}
//
//bool PacketPacker::setHeader(const char *header, const size_t size)
//{
//	if (!m_nHeader) return true;
//	if (m_nHeader != size) return false;
//
//	if (isLittleEndian_)
//		std::reverse_copy(header, header + size, std::inserter(m_bufData, m_bufData.begin()));
//	else
//	{
//		//m_bufData.insert(m_bufData.begin(), header, header + size);
//		std::copy(header, header + size, std::inserter(m_bufData, m_bufData.begin()));
//	}
//	return true;
//}
//
//bool PacketPacker::setFooter(const char *footer, const size_t size)
//{
//	if (!m_nFooter) return true;
//	if (m_nFooter != size) return false;
//
//	if (isLittleEndian_)
//		std::reverse_copy(footer, footer + size, std::back_inserter(m_bufData));
//	else
//	{
//		//m_bufData.insert(m_bufData.end(), footer, footer + size);
//		std::copy(footer, footer + size, std::back_inserter(m_bufData));
//	}
//
//	//--S [] 2004/06/30 : Sang-Wook Lee
///*
//	m_itCurr = m_bufData.end();
//	std::advance(m_itCurr, -size);
//*/
//	m_itCurr = m_bufData.begin();
//	std::advance(m_itCurr, m_bufData.size() - size);
//	//--E [] 2004/06/30
//	return true;
//}
//
//void PacketPacker::putChar(char data)
//{
//	//--S [] 2004/06/30 : Sang-Wook Lee
//	//m_bufData.push_back(data);
//	std::inserter(m_bufData, m_itCurr) = data;
//	//++m_itCurr;  // don't need because of a mechanism of inserter
//	//--E [] 2004/06/30
//}
//
//void PacketPacker::putShort(short data)
//{
//	if (isLittleEndian_)
//		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(short), std::inserter(m_bufData, m_itCurr));
//	else
//	{
///*
//		//m_bufData.insert(m_bufData.end(), (value_type *)&data, (value_type *)&data + sizeof(short));
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(short), std::back_inserter(m_bufData));
//*/
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(short), std::inserter(m_bufData, m_itCurr));
//		//std::advance(m_itCurr, sizeof(short));  // don't need because of a mechanism of inserter
//	}
//}
//
//void PacketPacker::putInt(int data)
//{
//	if (isLittleEndian_)
//		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(int), std::inserter(m_bufData, m_itCurr));
//	else
//	{
///*
//		//m_bufData.insert(m_bufData.end(), (value_type *)&data, (value_type *)&data + sizeof(int));
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(int), std::back_inserter(m_bufData));
//*/
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(int), std::inserter(m_bufData, m_itCurr));
//		//std::advance(m_itCurr, sizeof(int));  // don't need because of a mechanism of inserter
//	}
//}
//
//void PacketPacker::putLong(long data)
//{
//	if (isLittleEndian_)
//		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(long), std::inserter(m_bufData, m_itCurr));
//	else
//	{
///*
//		//m_bufData.insert(m_bufData.end(), (value_type *)&data, (value_type *)&data + sizeof(long));
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(long), std::back_inserter(m_bufData));
//*/
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(long), std::inserter(m_bufData, m_itCurr));
//		//std::advance(m_itCurr, sizeof(long));  // don't need because of a mechanism of inserter
//	}
//}
//
//void PacketPacker::putInt64(__int64 data)
//{
//	if (isLittleEndian_)
//		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(__int64), std::inserter(m_bufData, m_itCurr));
//	else
//	{
///*
//		//m_bufData.insert(m_bufData.end(), (value_type *)&data, (value_type *)&data + sizeof(__int64));
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(__int64), std::back_inserter(m_bufData));
//*/
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(__int64), std::inserter(m_bufData, m_itCurr));
//		//std::advance(m_itCurr, sizeof(__int64));  // don't need because of a mechanism of inserter
//	}
//}
//
//void PacketPacker::putFloat(float data)
//{
//	if (isLittleEndian_)
//		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(float), std::inserter(m_bufData, m_itCurr));
//	else
//	{
///*
//		//m_bufData.insert(m_bufData.end(), (value_type *)&data, (value_type *)&data + sizeof(float));
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(float), std::back_inserter(m_bufData));
//*/
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(float), std::inserter(m_bufData, m_itCurr));
//		//std::advance(m_itCurr, sizeof(float));  // don't need because of a mechanism of inserter
//	}
//}
//
//void PacketPacker::putDouble(double data)
//{
//	if (isLittleEndian_)
//		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(double), std::inserter(m_bufData, m_itCurr));
//	else
//	{
///*
//		//m_bufData.insert(m_bufData.end(), (value_type *)&data, (value_type *)&data + sizeof(double));
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(double), std::back_inserter(m_bufData));
//*/
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(double), std::inserter(m_bufData, m_itCurr));
//		//std::advance(m_itCurr, sizeof(double));  // don't need because of a mechanism of inserter
//	}
//}
//
//void PacketPacker::putLDouble(long double data)
//{
//	if (isLittleEndian_)
//		std::reverse_copy((value_type *)&data, (value_type *)&data + sizeof(long double), std::inserter(m_bufData, m_itCurr));
//	else
//	{
///*
//		//m_bufData.insert(m_bufData.end(), (value_type *)&data, (value_type *)&data + sizeof(long double));
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(long double), std::back_inserter(m_bufData));
//*/
//		std::copy((value_type *)&data, (value_type *)&data + sizeof(long double), std::inserter(m_bufData, m_itCurr));
//		//std::advance(m_itCurr, sizeof(long double));  // don't need because of a mechanism of inserter
//	}
//}
//
//void PacketPacker::putText(const char *data, const size_t size)
//{
//	// FIXME [check] >>
///*
//	if (isLittleEndian_)
//		std::reverse_copy(data, data + size, std::inserter(m_bufData, m_itCurr));
//	else
//	{
////		//m_bufData.insert(m_bufData.end(), data, data + size);
////		std::copy(data, data + size, std::back_inserter(m_bufData));
//		std::copy((value_type *)data, (value_type *)data + size, std::inserter(m_bufData, m_itCurr));
//		//std::advance(m_itCurr, size);  // don't need because of a mechanism of inserter
//	}
//*/
//	std::copy((value_type *)data, (value_type *)data + size, std::inserter(m_bufData, m_itCurr));
//}
//
//void PacketPacker::fillChar(char data, const size_t size)
//{
//	m_bufData.insert(m_itCurr, size, data);
//}
//
//
////-----------------------------------------------------------------------------------
////  byte-type packet unpacker
//
//PacketUnpacker::PacketUnpacker(const bool isLittleEndian, const size_t nHeader /*= 0*/, const size_t nFooter /*= 0*/)
//: isLittleEndian_(isLittleEndian),
//  m_nHeader(nHeader), m_nFooter(nFooter),
//  m_bufData()//, m_nData(m_bufData.size())
//{
//}
//
//PacketUnpacker::~PacketUnpacker()
//{
//}
//
//bool PacketUnpacker::initialize()
//{
//	if (m_bufData.empty()) return false;
//
//	size_t size = m_bufData.size();
//	//if (isFixedSize() && size != m_nData) return false;
//	if (size <= m_nHeader + m_nFooter) return false;
//
//	//--S [] 2004/06/30 : Sang-Wook Lee
//	// data body    getFooter() getText()  m_itCurr == m_itEnd  error 
//	m_itEnd = m_bufData.end();
///*
//	m_itEnd = m_bufData.begin();
//	std::advance(m_itEnd, size - m_nFooter);
//*/
//	//--E [] 2004/06/30
//
//	m_itCurr = m_bufData.begin();
//	std::advance(m_itCurr, m_nHeader);  //-- [] 2004/07/01 : Sang-Wook Lee
//	return true;
//}
//
//bool PacketUnpacker::finalize()
//{
//	m_itCurr = m_bufData.begin();
//	if (m_nHeader) std::advance(m_itCurr, m_nHeader);
//	return true;
//}
//
//bool PacketUnpacker::getHeader(char *header, const size_t size)
//{
//	if (!m_nHeader) return true;
//	if (m_nHeader != size) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	m_itCurr = m_bufData.begin();
//
//	bool bRet = getText(header, size);
//	m_itCurr = it;
//	return bRet;
//}
//
//bool PacketUnpacker::getFooter(char *footer, const size_t size)
//{
//	if (!m_nFooter) return true;
//	if (m_nFooter != size) return false;
//
//	size_t nData = m_bufData.size();
//	if (nData < m_nFooter) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	m_itCurr = m_bufData.begin();
//	std::advance(m_itCurr, nData - m_nFooter);
//
//	bool bRet = getText(footer, size);
//	m_itCurr = it;
//	return bRet;
//}
//
//bool PacketUnpacker::getChar(char &data) const
//{
//	PacketUnpacker::buffer_type::difference_type diff = std::distance(m_itCurr, m_itEnd);
//	if (diff < sizeof(char)) return false;
//
//	data = *m_itCurr;
//	++m_itCurr;
//	return true;
//}
//
//bool PacketUnpacker::getShort(short &data) const
//{
//	PacketUnpacker::buffer_type::difference_type diff = std::distance(m_itCurr, m_itEnd);
//	if (diff < sizeof(short)) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	std::advance(m_itCurr, sizeof(short));
//	if (isLittleEndian_)
//		std::reverse_copy(it, m_itCurr, (value_type *)&data);
//	else
//		std::copy(it, m_itCurr, (value_type *)&data);
//	return true;
//}
//
//bool PacketUnpacker::getInt(int &data) const
//{
//	PacketUnpacker::buffer_type::difference_type diff = std::distance(m_itCurr, m_itEnd);
//	if (diff < sizeof(int)) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	std::advance(m_itCurr, sizeof(int));
//	if (isLittleEndian_)
//		std::reverse_copy(it, m_itCurr, (value_type *)&data);
//	else
//		std::copy(it, m_itCurr, (value_type *)&data);
//	return true;
//}
//
//bool PacketUnpacker::getLong(long &data) const
//{
//	PacketUnpacker::buffer_type::difference_type diff = std::distance(m_itCurr, m_itEnd);
//	if (diff < sizeof(long)) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	std::advance(m_itCurr, sizeof(long));
//	if (isLittleEndian_)
//		std::reverse_copy(it, m_itCurr, (value_type *)&data);
//	else
//		std::copy(it, m_itCurr, (value_type *)&data);
//	return true;
//}
//
//bool PacketUnpacker::getInt64(__int64 &data) const
//{
//	PacketUnpacker::buffer_type::difference_type diff = std::distance(m_itCurr, m_itEnd);
//	if (diff < sizeof(__int64)) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	std::advance(m_itCurr, sizeof(__int64));
//	if (isLittleEndian_)
//		std::reverse_copy(it, m_itCurr, (value_type *)&data);
//	else
//		std::copy(it, m_itCurr, (value_type *)&data);
//	return true;
//}
//
//bool PacketUnpacker::getFloat(float &data) const
//{
//	PacketUnpacker::buffer_type::difference_type diff = std::distance(m_itCurr, m_itEnd);
//	if (diff < sizeof(float)) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	std::advance(m_itCurr, sizeof(float));
//	if (isLittleEndian_)
//		std::reverse_copy(it, m_itCurr, (value_type *)&data);
//	else
//		std::copy(it, m_itCurr, (value_type *)&data);
//	return true;
//}
//
//bool PacketUnpacker::getDouble(double &data) const
//{
//	PacketUnpacker::buffer_type::difference_type diff = std::distance(m_itCurr, m_itEnd);
//	if (diff < sizeof(double)) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	std::advance(m_itCurr, sizeof(double));
//	if (isLittleEndian_)
//		std::reverse_copy(it, m_itCurr, (value_type *)&data);
//	else
//		std::copy(it, m_itCurr, (value_type *)&data);
//	return true;
//}
//
//bool PacketUnpacker::getLDouble(long double &data) const
//{
//	PacketUnpacker::buffer_type::difference_type diff = std::distance(m_itCurr, m_itEnd);
//	if (diff < sizeof(long double)) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	std::advance(m_itCurr, sizeof(long double));
//	if (isLittleEndian_)
//		std::reverse_copy(it, m_itCurr, (value_type *)&data);
//	else
//		std::copy(it, m_itCurr, (value_type *)&data);
//	return true;
//}
//
//bool PacketUnpacker::getText(char *data, const size_t size) const
//{
//	PacketUnpacker::buffer_type::difference_type diff = std::distance(m_itCurr, m_itEnd);
//	if ((size_t)diff < size) return false;
//
//	PacketUnpacker::buffer_type::const_iterator it = m_itCurr;
//	std::advance(m_itCurr, size);
//	// FIXME [check] >>
///*
//	if (isLittleEndian_)
//		std::reverse_copy(it, m_itCurr, (value_type *)data);
//	else
//		std::copy(it, m_itCurr, (value_type *)data);
//*/
//	std::copy(it, m_itCurr, (value_type *)data);
//	return true;
//}

}  // namespace swl
