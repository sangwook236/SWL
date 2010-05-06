#include "swl/Config.h"
#include "swl/util/RegionOfInterestMgr.h"
#include "swl/base/LogException.h"


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

void RegionOfInterestMgr::addROI(const roi_type &roi)
{
	ROIs_.push_back(boost::shared_ptr<roi_type>(roi.clone()));
}

void RegionOfInterestMgr::removeROI(const roi_id_type &id)
{
	const size_t count = ROIs_.size();
	if (id >= count) return;

	rois_type::iterator it = ROIs_.begin();
	std::advance(it, id);
	if (ROIs_.end() != it) ROIs_.erase(it);
}

RegionOfInterestMgr::roi_type & RegionOfInterestMgr::getROI(const roi_id_type &id)
{
	const size_t count = ROIs_.size();
	if (id >= count)
		throw LogException(LogException::L_ERROR, "invalid ID", __FILE__, __LINE__, __FUNCTION__);

	rois_type::iterator it = ROIs_.begin();
	std::advance(it, id);

	if (ROIs_.end() != it && *it) return *(it->get());
	else
		throw LogException(LogException::L_ERROR, "invalid ID", __FILE__, __LINE__, __FUNCTION__);
}

const RegionOfInterestMgr::roi_type & RegionOfInterestMgr::getROI(const roi_id_type &id) const
{
	const size_t count = ROIs_.size();
	if (id >= count)
		throw LogException(LogException::L_ERROR, "invalid ID", __FILE__, __LINE__, __FUNCTION__);

	rois_type::const_iterator it = ROIs_.begin();
	std::advance(it, id);

	if (ROIs_.end() != it && *it) return *(it->get());
	else
		throw LogException(LogException::L_ERROR, "invalid ID", __FILE__, __LINE__, __FUNCTION__);
}

bool RegionOfInterestMgr::swap(const roi_id_type &lhs, const roi_id_type &rhs)
{
	const size_t count = ROIs_.size();
	if (lhs >= count || rhs >= count) return false;

	rois_type::iterator itLhs = ROIs_.begin(), itRhs = itLhs;
	std::advance(itLhs, lhs);
	std::advance(itRhs, rhs);

	if (ROIs_.end() == itLhs || ROIs_.end() == itRhs) return false;

	std::iter_swap(itLhs, itRhs);
	return true;
}

void RegionOfInterestMgr::setValidRegion(const region_type &validRegion)
{
	validRegion_.reset(new region_type(validRegion));
}

void RegionOfInterestMgr::resetValidRegion()
{
	validRegion_.reset();
}

bool RegionOfInterestMgr::isInValidRegion(const point_type &pt) const
{
	const point_type::value_type &tol = point_type::value_type(1.0e-5);
	return !validRegion_ ? true : validRegion_->isIncluded(pt, tol);
}

bool RegionOfInterestMgr::pickROI(const point_type &pt)
{
	pickedVertex_ = roi_type::point_type();

	for (rois_type::reverse_iterator rit = ROIs_.rbegin(); rit != ROIs_.rend(); ++rit)
	{
		if (*rit && (*rit)->isVertex(pt, vertexTol_))
		{
			pickedROI_ = rit->get();
			isVertexPicked_ = true;
			pickedVertex_ = pt;
			return true;
		}
	}

	for (rois_type::reverse_iterator rit = ROIs_.rbegin(); rit != ROIs_.rend(); ++rit)
	{
		if (*rit && (*rit)->include(pt, vertexTol_))
		{
			pickedROI_ = rit->get();
			isVertexPicked_ = false;
			return true;
		}
	}

	pickedROI_ = NULL;
	return false;
}

bool RegionOfInterestMgr::isPickedVertex(const point_type &pt) const
{
	return pickedROI_ && isVertexPicked_ && roi_type::isNearPoint(pt, pickedVertex_, vertexTol_);
}

bool RegionOfInterestMgr::moveVertexInPickedROI(const point_type &pt, const point_type &delta)
{
	return validRegion_ ? pickedROI_->moveVertex(pt, delta, *validRegion_, vertexTol_) : pickedROI_->moveVertex(pt, delta, vertexTol_);
}

void RegionOfInterestMgr::movePickedROI(const point_type &delta)
{
	validRegion_ ? pickedROI_->moveRegion(delta, *validRegion_) : pickedROI_->moveRegion(delta);
}

}  // namespace swl
