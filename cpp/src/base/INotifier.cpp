#include "swl/Config.h"
#include "swl/base/INotifier.h"
#include "swl/base/IObserver.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class INotifier

INotifier::~INotifier()
{}


//--------------------------------------------------------------------------
//  class Notifier

Notifier::Notifier()
: base_type(),
  observers_()
{
}

Notifier::Notifier(const Notifier &rhs)
: base_type(rhs),
  observers_(rhs.observers_)
{
}

Notifier::~Notifier()
{
	//clearAllObservers();
}

Notifier & Notifier::operator=(const Notifier &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	clearAllObservers();
	//std::copy(rhs.observers_.begin(), rhs.observers_.end(), std::inserter(observers_, observers_.end()));
	observers_.insert(rhs.observers_.begin(), rhs.observers_.end());
	return *this;
}

void Notifier::notifyObservers(const boost::any &msg /*= boost::any()*/)
{
	if (!isChanged()) return;

	for (observers_type::iterator it = observers_.begin(); it != observers_.end(); ++it)
		if (*it) (*it)->updateObserver(*this, msg);

	resetChanged();
}

bool Notifier::findObserver(observer_type &observer) const
{  return observers_.find(&observer) != observers_.end();  }

bool Notifier::addObserver(observer_type &observer)
{
	if (!findObserver(observer))
		return observers_.insert(&observer).second;
	else return false;
}

bool Notifier::removeObserver(observer_type &observer)
{
	Notifier::observers_type::iterator itObserver = observers_.find(&observer);
	if (itObserver != observers_.end())
	{
		observers_.erase(itObserver);
		return true;
	}
	else return false;
}

void Notifier::clearAllObservers()
{  observers_.clear();  }

Notifier::observers_type::size_type Notifier::countObserver() const
{  return observers_.size();  }

bool Notifier::containObserver() const
{  return !observers_.empty();  }

}  // namespace swl
