#if !defined(__SWL_COMMON__BYTE_BUFFER__H_)
#define __SWL_COMMON__BYTE_BUFFER__H_ 1


#include "swl/common/ExportCommon.h"
#include "swl/common/VcWarningDisable.h"
#include <boost/thread/mutex.hpp>
#include <deque>


namespace swl {

//-----------------------------------------------------------------------------------
//	byte buffer

//template<class T>
class SWL_COMMON_API ByteBuffer
{
public:
	//typedef ByteBuffer				base_type;
	//typedef T							value_type;
	typedef unsigned char				value_type;
	typedef std::deque<value_type>		buffer_type;

public:
	ByteBuffer();
	~ByteBuffer();

public:
	bool push(const value_type& data);
	bool push(const value_type* data, const size_t dataSize);
	bool pop();
	bool pop(const size_t dataSize);
	bool top(value_type& data) const;
	bool top(value_type* data, const size_t dataSize) const;

	void clear();

	size_t getSize() const  {  return bufData_.size();  }
	bool isEmpty() const  {  return bufData_.empty();  }

private:
	buffer_type bufData_;

	mutable boost::mutex mutex_;
};

}  // namespace swl


#endif  // __SWL_COMMON__BYTE_BUFFER__H_
