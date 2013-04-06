#include "swl/Config.h"
#include "swl/util/GuardedBuffer.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

#if 0

//-----------------------------------------------------------------------------------
//	byte buffer

GuardedBuffer::GuardedBuffer()
: buf_()
{
}

GuardedBuffer::~GuardedBuffer()
{
	boost::mutex::scoped_lock lock(mutex_);
	buf_.clear();
}

bool GuardedBuffer::push(const GuardedBuffer::value_type &data)
{
	if (buf_.size() > buf_.max_size())
		return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		buf_.push_back(data);
		return true;
	}
}

bool GuardedBuffer::push(const GuardedBuffer::value_type *data, const size_t len)
{
	if (buf_.size() + len >= buf_.max_size())
		return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		//buf_.insert(buf_.end(), data, data + len);
		std::copy(data, data + len, std::back_inserter(buf_));
		return true;
	}
}

bool GuardedBuffer::pop()
{
	if (buf_.empty()) return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		buf_.pop_front();
		return true;
	}
}

bool GuardedBuffer::pop(const size_t len)
{
	if (buf_.size() < len) return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		buffer_type::iterator itBegin = buf_.begin();
		buffer_type::iterator it = itBegin;
		std::advance(it, len);
		buf_.erase(itBegin, it);
		return true;
	}
}

bool GuardedBuffer::top(GuardedBuffer::value_type &data) const
{
	if (buf_.empty()) return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		data = buf_.front();
		return true;
	}
}

bool GuardedBuffer::top(GuardedBuffer::value_type *data, const size_t len) const
{
	if (buf_.size() < len) return false;
	else
	{
        boost::mutex::scoped_lock lock(mutex_);
		buffer_type::const_iterator itBegin = buf_.begin();
		buffer_type::const_iterator it = itBegin;
		std::advance(it, len);
		std::copy(itBegin, it, data);
		return true;
	}
}

void GuardedBuffer::clear()
{
	if (!buf_.empty())
	{
        boost::mutex::scoped_lock lock(mutex_);
		buf_.clear();
	}
}

#endif  // 0

}  // namespace swl
