#include "swl/common/ByteBuffer.h"


#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//	byte buffer

ByteBuffer::ByteBuffer()
: bufData_()
{
}

ByteBuffer::~ByteBuffer()
{
	boost::mutex::scoped_lock lock(mutex_);
	bufData_.clear();
}

bool ByteBuffer::push(const ByteBuffer::value_type &data)
{
	if (bufData_.size() > bufData_.max_size())
		return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		bufData_.push_back(data);
		return true;
	}
}

bool ByteBuffer::push(const ByteBuffer::value_type* data, const size_t dataSize)
{
	if (bufData_.size() + dataSize >= bufData_.max_size())
		return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		//bufData_.insert(bufData_.end(), data, data + dataSize);
		std::copy(data, data + dataSize, std::back_inserter(bufData_));
		return true;
	}
}

bool ByteBuffer::pop()
{
	if (bufData_.empty()) return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		bufData_.pop_front();
		return true;
	}
}

bool ByteBuffer::pop(const size_t dataSize)
{
	if (bufData_.size() < dataSize) return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		buffer_type::iterator itBegin = bufData_.begin();
		buffer_type::iterator it = itBegin;
		std::advance(it, dataSize);
		bufData_.erase(itBegin, it);
		return true;
	}
}

bool ByteBuffer::top(ByteBuffer::value_type& data) const
{
	if (bufData_.empty()) return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		data = bufData_.front();
		return true;
	}
}

bool ByteBuffer::top(ByteBuffer::value_type* data, const size_t dataSize) const
{
	if (bufData_.size() < dataSize) return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		buffer_type::const_iterator itBegin = bufData_.begin();
		buffer_type::const_iterator it = itBegin;
		std::advance(it, dataSize);
		std::copy(itBegin, it, data);
		return true;
	}
}

void ByteBuffer::clear()
{
	if (!bufData_.empty())
	{
        boost::mutex::scoped_lock lock(mutex_);
		bufData_.clear();
	}
}

}  // namespace swl
