#include "swl/Config.h"
#include "swl/graphics/GeometryPool.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class GeometryPool

/*static*/ GeometryPool *GeometryPool::singleton_ = NULL;

/*static*/ GeometryPool & GeometryPool::getInstance()
{
	if (NULL == singleton_)
		singleton_ = new GeometryPool();

	return *singleton_;
}

GeometryPool::geometry_id_type & GeometryPool::createGeometryId()
{
	const size_t count = geometryPool_.size();
	geometry_id_type geomId = count;
	for (geometry_id_type id = 0; id < count; ++id)
		if (geometryPool_.end() == geometryPool_.find(id))
		{
			geomId = id;
			break;
		}

	geometryPool_.insert(std::make_pair(geomId, boost::shared_ptr<geometry_type>()));
	return geomId;
}

void GeometryPool::deleteGeometryId(const GeometryPool::geometry_id_type &geomId)
{
	if (geometryPool_.end() != geometryPool_.find(geomId))
		geometryPool_.erase(geomId);
}

bool GeometryPool::setGeometry(const geometry_id_type &geomId, geometry_type &geom)
{
	if (geometryPool_.empty()) return false;
	geometry_pool_type::iterator it = geometryPool_.find(geomId);
	if (geometryPool_.end() == it) return false;
	else
	{
		// TODO [check] >>
		//it->second = boost::shared_ptr<geometry_type>(&geom);
		it->second.reset(&geom);
		return true;
	}
}

boost::shared_ptr<GeometryPool::geometry_type> GeometryPool::getGeometry(const GeometryPool::geometry_id_type &geomId)
{
	if (geometryPool_.empty()) return boost::shared_ptr<geometry_type>();
	geometry_pool_type::iterator it = geometryPool_.find(geomId);
	return geometryPool_.end() == it ? boost::shared_ptr<geometry_type>() : it->second;
}

const boost::shared_ptr<GeometryPool::geometry_type> GeometryPool::getGeometry(const GeometryPool::geometry_id_type &geomId) const
{
	if (geometryPool_.empty()) return boost::shared_ptr<geometry_type>();
	geometry_pool_type::const_iterator it = geometryPool_.find(geomId);
	return geometryPool_.end() == it ? boost::shared_ptr<geometry_type>() : it->second;
}

}  // namespace swl
