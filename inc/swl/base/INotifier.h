#if !defined(__SWL_BASE__NOTIFIER_INTERFACE__H_)
#define __SWL_BASE__NOTIFIER_INTERFACE__H_ 1


#include "swl/base/ExportBase.h"
#include "swl/DisableCompilerWarning.h"
#include <boost/any.hpp>
#include <set>

namespace swl {

class IObserver;

//--------------------------------------------------------------------------
//  class INotifier

class SWL_BASE_API INotifier
{
public:
	//typedef INotifier base_type;

protected:
	explicit INotifier()  {}
	explicit INotifier(const INotifier&)  {}

public:
	virtual ~INotifier();

public:
	///
	virtual void notifyObservers(const boost::any& msg = boost::any()) = 0;
};

//--------------------------------------------------------------------------
//  class Notifier

class SWL_BASE_API Notifier: public INotifier
{
public:
	typedef INotifier					base_type;
	typedef IObserver					observer_type;
	typedef std::set<observer_type*>	observers_type;

protected:
    explicit Notifier();
    explicit Notifier(const Notifier& rhs);

public:
	virtual ~Notifier();

	Notifier& operator=(const Notifier& rhs);

public:
	///
	/*virtual*/ void notifyObservers(const boost::any& msg = boost::any());

	///
	bool isChanged() const  {  return isChanged_;  }
	void setChanged()  {  isChanged_ = true;  }
	void resetChanged()  {  isChanged_ = false;  }

	///
	bool findObserver(observer_type& observer) const;
	bool addObserver(observer_type& observer);
	bool removeObserver(observer_type& observer);
	void clearAllObservers();
	observers_type::size_type getObserverSize() const;
	bool isEmptyObserver() const;

private:
	bool isChanged_;

	observers_type observers_;
};

}  // namespace swl


#include "swl/EnableCompilerWarning.h"


#endif  // __SWL_BASE__NOTIFIER_INTERFACE__H_
