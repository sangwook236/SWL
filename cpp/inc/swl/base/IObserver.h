#if !defined(__SWL_BASE__OBSERVER_INTERFACE__H_)
#define __SWL_BASE__OBSERVER_INTERFACE__H_ 1


#include "swl/base/INotifier.h"


namespace swl {

//--------------------------------------------------------------------------
//  class IObserver

class SWL_BASE_API IObserver
{
public:
	//typedef IObserver		base_type;
	typedef INotifier		notifier_type;

protected:
	explicit IObserver()  {}
	explicit IObserver(const IObserver &)  {}

public:
	virtual ~IObserver();

public:
	///
	virtual void updateObserver(const notifier_type &notifier, const boost::any &msg = boost::any()) = 0;
};

}  // namespace swl


#endif  // __SWL_BASE__OBSERVER_INTERFACE__H_
