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
 *	@brief  통신을 위한 data buffer class.
 *
 *	Serial 통신이나 Ethernet 통신에서 data를 송수신하는 경우 송수신하는 data를 관리하는 역할을 수행한다.
 *
 *	push, pop, clear 같은 operation은 multi-thread 환경에서 동작할 수 있도록 mutex를 이용해 synchronization을 수행한다.
 */
template<class T>
class GuardedBuffer
{
public:
	//typedef GuardedBuffer				base_type;
	/**
	 *	@brief  통신을 위해 사용하는 자료형. template parameter로 주어짐.
	 *
	 *	Type definition을 이용해 value_type을 지정하므로, 향후 class 내부에서 실제적으로 사용하는 data 형이 바뀌더라도 외부 프로그램에 영향을 덜 주게 된다.
	 */
	typedef T							value_type;
private:
	typedef std::deque<value_type>		buffer_type;

public:
	/**
	 *	@brief  [ctor] default contructor.
	 *
	 *	내부 data buffer를 초기화한다.
	 */
	GuardedBuffer()
	: buf_()
	{}
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	내부 data buffer를 정리.
	 */
	~GuardedBuffer()
	{
		boost::mutex::scoped_lock lock(mutex_);
		buf_.clear();
	}

public:
	/**
	 *	@brief  인자로 넘겨진 하나의 값을 data buffer에 push.
	 *	@param[in]  data  data buffer에 저장될 data.
	 *	@return  data가 정상적으로 buffer에 저장되었다면 true 반환.
	 *
	 *	내부 data buffer가 저장할 수 있는 최대 크기보다 많은 수의 data가 저장되는 경우 false를 반환한다.
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
	 *	@brief  인자로 넘겨진 여러 data를 data buffer에 push.
	 *	@param[in]  data  data buffer에 저장될 data의 pointer.
	 *	@param[in]  len  data 인자에 저장되어 있는 data 개수.
	 *	@return  data가 정상적으로 buffer에 저장되었다면 true 반환.
	 *
	 *	내부 data buffer가 저장할 수 있는 최대 크기보다 많은 수의 data가 저장되는 경우 false를 반환한다.
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
	 *	@brief  data buffer로부터 하나의 값을 pop.
	 *	@return  data buffer가 비어 있어서 data를 pop할 수 없다면 false 반환.
	 *
	 *	내부 data buffer가 비어 있다면 false를 반환한다.
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
	 *	@brief  data buffer로부터 지정된 개수의 자료를 pop.
	 *	@param[in]  len  data buffer로부터 pop해야할 자료 개수.
	 *	@return  data buffer에 들어있는 자료의 개수가 인자로 넘겨진 dataSize보다 작다면 false 반환.
	 *
	 *	내부 data buffer에 저장되어 있는 자료의 개수가 요구되는 개수, dataSize보다 작다면 false를 반환한다.
	 */
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
	/**
	 *	@brief  data buffer에 저장되어 있는 자료중 가장 먼저 저장된 자료를 인자를 통해 반환.
	 *	@param[out]  data  data buffer에 가장 먼저 저장된 자료가 반환되는 변수.
	 *	@return  data buffer가 비어 있다면 false를 반환.
	 *
	 *	data buffer로부터 하나의 자료를 반환하지만 해당 data가 data buffer에서 없어지지는 않는다.
	 *	해당 자료를 data buffer에서 제거하려면 pop() 함수를 호출하여야 한다.
	 *	내부 data buffer가 비어 있어 반환할 값이 없다면 false를 반환한다.
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
	 *	@brief  data buffer에 저장되어 있는 자료중 가장 먼저 저장된 자료를 인자를 통해 반환.
	 *	@param[out]  data  data buffer로부터 꺼내어져 반환될 자료가 저장되는 변수.
	 *	@param[in]  len  data buffer로부터 꺼낼 자료의 개수.
	 *	@return  data buffer에 들어있는 자료의 개수가 인자로 넘겨진 dataSize보다 작다면 false 반환.
	 *
	 *	data buffer로부터 하나의 자료를 반환하지만 해당 data가 data buffer에서 없어지지는 않는다.
	 *	해당 자료를 data buffer에서 제거하려면 pop() 함수를 호출하여야 한다.
	 *	내부 data buffer에 저장되어 있는 자료의 개수가 요구되는 개수, dataSize보다 작다면 false를 반환한다.
	 */
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

	/**
	 *	@brief  data buffer를 지움.
	 *
	 *	data buffer에 있는 모든 data를 지운다.
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
	 *	@brief  data buffer에 들어 있는 data의 개수를 알려줌.
	 *	@return  data buffer에 들어 있는 data의 개수.
	 *
	 *	data buffer에 들어 있는 data의 개수를 알려준다.
	 */
	size_t getSize() const  {  return buf_.size();  }
	/**
	 *	@brief  data buffer가 비어 있는지 확인.
	 *	@return  data buffer가 비어 있다면 true를 반환.
	 *
	 *	data buffer가 비어 있다면 true를, 비어 있지 않다면 false를 반환한다.
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
