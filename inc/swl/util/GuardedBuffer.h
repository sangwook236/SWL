#if !defined(__SWL_UTIL__GUARDED_BUFFER__H_)
#define __SWL_UTIL__GUARDED_BUFFER__H_ 1


#include "swl/DisableCompilerWarning.h"
#include <boost/thread/mutex.hpp>
#include <boost/config.hpp>
#include <deque>


namespace swl {

//-----------------------------------------------------------------------------------
//	mutex-guarded buffer

template<class T>
class GuardedBuffer
{
public:
	//typedef GuardedBuffer				base_type;
	typedef T							value_type;
	typedef std::deque<value_type>		buffer_type;

public:
	GuardedBuffer()
	: buf_()
	{}
	~GuardedBuffer()
	{
		boost::mutex::scoped_lock lock(mutex_);
		buf_.clear();
	}

public:
	bool push(const value_type &data)
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
	bool push(const value_type *data, const size_t len)
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
	template <class InputIterator>
	bool push(InputIterator first, InputIterator last)
	{
		const size_t len = std::distance(first, last);
		if (buf_.size() + len >= buf_.max_size())
			return false;
		else
		{
			boost::mutex::scoped_lock lock(mutex_);
			//buf_.insert(buf_.end(), first, last);
			std::copy(first, last, std::back_inserter(buf_));
			return true;
		}
	}
	bool pop()
	{
		if (buf_.empty()) return false;
		else
		{
			boost::mutex::scoped_lock lock(mutex_);
			buf_.pop_front();
			return true;
		}
	}
	bool pop(const size_t len)
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
	bool top(value_type &data) const
	{
		if (buf_.empty()) return false;
		else
		{
			boost::mutex::scoped_lock lock(mutex_);
			data = buf_.front();
			return true;
		}
	}
	bool top(value_type *data, const size_t len) const
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

	void clear()
	{
		if (!buf_.empty())
		{
			boost::mutex::scoped_lock lock(mutex_);
			buf_.clear();
		}
	}

	size_t getSize() const  {  return buf_.size();  }
	bool isEmpty() const  {  return buf_.empty();  }

private:
	buffer_type buf_;

	mutable boost::mutex mutex_;
};

//-----------------------------------------------------------------------------------
//	mutex-guarded byte(unsigned char) buffer

typedef GuardedBuffer<unsigned char> GuardedByteBuffer;

}  // namespace swl


#include "swl/EnableCompilerWarning.h"


#endif  // __SWL_UTIL__GUARDED_BUFFER__H_
