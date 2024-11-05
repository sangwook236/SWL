#if !defined(__SWL_UTIL__GUARDED_BUFFER__H_)
#define __SWL_UTIL__GUARDED_BUFFER__H_ 1


#include "swl/DisableCompilerWarning.h"
#include <boost/thread/mutex.hpp>
#include <boost/config.hpp>
#include <deque>


namespace swl {

//-----------------------------------------------------------------------------------
//	mutex-guarded buffer

/**
 *	@brief  ����� ���� data buffer class.
 *
 *	Serial ����̳� Ethernet ��ſ��� data�� �ۼ����ϴ� ��� �ۼ����ϴ� data�� �����ϴ� ������ �����Ѵ�.
 *
 *	push, pop, clear ���� operation�� multi-thread ȯ�濡�� ������ �� �ֵ��� mutex�� �̿��� synchronization�� �����Ѵ�.
 */
template<class T>
class GuardedBuffer
{
public:
	//typedef GuardedBuffer				base_type;
	/**
	 *	@brief  ����� ���� ����ϴ� �ڷ���. template parameter�� �־���.
	 *
	 *	Type definition�� �̿��� value_type�� �����ϹǷ�, ���� class ���ο��� ���������� ����ϴ� data ���� �ٲ���� �ܺ� ���α׷��� ������ �� �ְ� �ȴ�.
	 */
	typedef T							value_type;
private:
	typedef std::deque<value_type>		buffer_type;

public:
	/**
	 *	@brief  [ctor] default constructor.
	 *
	 *	���� data buffer�� �ʱ�ȭ�Ѵ�.
	 */
	GuardedBuffer()
	: buf_()
	{}
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	���� data buffer�� ����.
	 */
	~GuardedBuffer()
	{
		boost::mutex::scoped_lock lock(mutex_);
		buf_.clear();
	}

public:
	/**
	 *	@brief  ���ڷ� �Ѱ��� �ϳ��� ���� data buffer�� push.
	 *	@param[in]  data  data buffer�� ����� data.
	 *	@return  data�� ���������� buffer�� ����Ǿ��ٸ� true ��ȯ.
	 *
	 *	���� data buffer�� ������ �� �ִ� �ִ� ũ�⺸�� ���� ���� data�� ����Ǵ� ��� false�� ��ȯ�Ѵ�.
	 */
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
	/**
	 *	@brief  ���ڷ� �Ѱ��� ���� data�� data buffer�� push.
	 *	@param[in]  data  data buffer�� ����� data�� pointer.
	 *	@param[in]  len  data ���ڿ� ����Ǿ� �ִ� data ����.
	 *	@return  data�� ���������� buffer�� ����Ǿ��ٸ� true ��ȯ.
	 *
	 *	���� data buffer�� ������ �� �ִ� �ִ� ũ�⺸�� ���� ���� data�� ����Ǵ� ��� false�� ��ȯ�Ѵ�.
	 */
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
	/**
	 *	@brief  data buffer�κ��� �ϳ��� ���� pop.
	 *	@return  data buffer�� ��� �־ data�� pop�� �� ���ٸ� false ��ȯ.
	 *
	 *	���� data buffer�� ��� �ִٸ� false�� ��ȯ�Ѵ�.
	 */
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
	/**
	 *	@brief  data buffer�κ��� ������ ������ �ڷḦ pop.
	 *	@param[in]  len  data buffer�κ��� pop�ؾ��� �ڷ� ����.
	 *	@return  data buffer�� ����ִ� �ڷ��� ������ ���ڷ� �Ѱ��� dataSize���� �۴ٸ� false ��ȯ.
	 *
	 *	���� data buffer�� ����Ǿ� �ִ� �ڷ��� ������ �䱸�Ǵ� ����, dataSize���� �۴ٸ� false�� ��ȯ�Ѵ�.
	 */
	bool pop(const size_t len)
	{
		if (buf_.size() < len) return false;
		else
		{
			boost::mutex::scoped_lock lock(mutex_);
			typename buffer_type::iterator itBegin = buf_.begin();
			typename buffer_type::iterator it = itBegin;
			std::advance(it, len);
			buf_.erase(itBegin, it);
			return true;
		}
	}
	/**
	 *	@brief  data buffer�� ����Ǿ� �ִ� �ڷ��� ���� ���� ����� �ڷḦ ���ڸ� ���� ��ȯ.
	 *	@param[out]  data  data buffer�� ���� ���� ����� �ڷᰡ ��ȯ�Ǵ� ����.
	 *	@return  data buffer�� ��� �ִٸ� false�� ��ȯ.
	 *
	 *	data buffer�κ��� �ϳ��� �ڷḦ ��ȯ������ �ش� data�� data buffer���� ���������� �ʴ´�.
	 *	�ش� �ڷḦ data buffer���� �����Ϸ��� pop() �Լ��� ȣ���Ͽ��� �Ѵ�.
	 *	���� data buffer�� ��� �־� ��ȯ�� ���� ���ٸ� false�� ��ȯ�Ѵ�.
	 */
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
	/**
	 *	@brief  data buffer�� ����Ǿ� �ִ� �ڷ��� ���� ���� ����� �ڷḦ ���ڸ� ���� ��ȯ.
	 *	@param[out]  data  data buffer�κ��� �������� ��ȯ�� �ڷᰡ ����Ǵ� ����.
	 *	@param[in]  len  data buffer�κ��� ���� �ڷ��� ����.
	 *	@return  data buffer�� ����ִ� �ڷ��� ������ ���ڷ� �Ѱ��� dataSize���� �۴ٸ� false ��ȯ.
	 *
	 *	data buffer�κ��� �ϳ��� �ڷḦ ��ȯ������ �ش� data�� data buffer���� ���������� �ʴ´�.
	 *	�ش� �ڷḦ data buffer���� �����Ϸ��� pop() �Լ��� ȣ���Ͽ��� �Ѵ�.
	 *	���� data buffer�� ����Ǿ� �ִ� �ڷ��� ������ �䱸�Ǵ� ����, dataSize���� �۴ٸ� false�� ��ȯ�Ѵ�.
	 */
	bool top(value_type *data, const size_t len) const
	{
		if (buf_.size() < len) return false;
		else
		{
			boost::mutex::scoped_lock lock(mutex_);
			typename buffer_type::const_iterator itBegin = buf_.begin();
			typename buffer_type::const_iterator it = itBegin;
			std::advance(it, len);
			std::copy(itBegin, it, data);
			return true;
		}
	}

	/**
	 *	@brief  data buffer�� ����.
	 *
	 *	data buffer�� �ִ� ��� data�� �����.
	 */
	void clear()
	{
		if (!buf_.empty())
		{
			boost::mutex::scoped_lock lock(mutex_);
			buf_.clear();
		}
	}

	/**
	 *	@brief  data buffer�� ��� �ִ� data�� ������ �˷���.
	 *	@return  data buffer�� ��� �ִ� data�� ����.
	 *
	 *	data buffer�� ��� �ִ� data�� ������ �˷��ش�.
	 */
	size_t getSize() const  {  return buf_.size();  }
	/**
	 *	@brief  data buffer�� ��� �ִ��� Ȯ��.
	 *	@return  data buffer�� ��� �ִٸ� true�� ��ȯ.
	 *
	 *	data buffer�� ��� �ִٸ� true��, ��� ���� �ʴٸ� false�� ��ȯ�Ѵ�.
	 */
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
