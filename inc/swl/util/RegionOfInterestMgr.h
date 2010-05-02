#if !defined(__SWL_UTIL__REGION_OF_INTEREST_MANAGER__H_)
#define __SWL_UTIL__REGION_OF_INTEREST_MANAGER__H_ 1


#include "swl/util/RegionOfInterest.h"
#include <boost/smart_ptr.hpp>
#include <list>


namespace swl {

//-----------------------------------------------------------------------------------------
//

class SWL_UTIL_API RegionOfInterestMgr
{
public:
	//typedef RegionOfInterestMgr base_type;
	typedef unsigned int roi_id_type;
	typedef RegionOfInterest roi_type;
	typedef roi_type::real_type real_type;
	typedef roi_type::point_type point_type;
	typedef roi_type::region_type region_type;
	typedef std::list<boost::shared_ptr<roi_type> > rois_type;

private:
	RegionOfInterestMgr()
	: validRegion_()
	{}
public:
	~RegionOfInterestMgr()  {}

private:
	RegionOfInterestMgr(const RegionOfInterestMgr &rhs);
	RegionOfInterestMgr & operator=(const RegionOfInterestMgr &rhs);

public:
	static RegionOfInterestMgr & getInstance();
	static void clearInstance();

public:
	//
	void addROI(const roi_type &roi);
	void removeROI(const roi_id_type &id);
	void clearAllROIs()  {  ROIs_.clear();  }
	size_t countROI() const  {  return ROIs_.size();  }
	bool containROI() const  {  return !ROIs_.empty();  }

	roi_type & getROI(const roi_id_type &id);
	const roi_type & getROI(const roi_id_type &id) const;

	//
	void setValidRegion(const region_type &validRegion);
	void resetValidRegion();
	bool isInValidRegion(const point_type &pt) const;

private:
	static boost::scoped_ptr<RegionOfInterestMgr> singleton_;

	rois_type ROIs_;
	boost::scoped_ptr<region_type> validRegion_;
};

}  // namespace swl


#endif  // __SWL_UTIL__REGION_OF_INTEREST_MANAGER__H_
