#include "swl/Config.h"
#include "swl/util/RegionOfInterestMgr.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
//

/*static*/ boost::scoped_ptr<RegionOfInterestMgr> RegionOfInterestMgr::singleton_;

/*static*/ RegionOfInterestMgr & RegionOfInterestMgr::getInstance()
{
	if (!singleton_)
		singleton_.reset(new RegionOfInterestMgr());

	return *singleton_;
}

/*static*/ void RegionOfInterestMgr::clearInstance()
{
	singleton_.reset();
}

void RegionOfInterestMgr::addROI(roi_type &roi)
{
	// FIXME [implement] >>
	throw std::runtime_error("not yet implemented");
}

void RegionOfInterestMgr::removeROI(const roi_id_type &id)
{
	rois_type::iterator it = ROIs_.begin();
	std::advance(it, id);
	ROIs_.erase(it);
}

RegionOfInterestMgr::roi_type & RegionOfInterestMgr::getROI(const roi_id_type &id)
{
	// FIXME [implement] >>
	throw std::runtime_error("not yet implemented");
}

const RegionOfInterestMgr::roi_type & RegionOfInterestMgr::getROI(const roi_id_type &id) const
{
	// FIXME [implement] >>
	throw std::runtime_error("not yet implemented");
}

bool RegionOfInterestMgr::isInValidRegion(const point_type &pt) const
{
	const point_type::value_type &tol = point_type::value_type(1.0e-5);
	return validRegion_.isIncluded(pt, tol);
}

}  // namespace swl
