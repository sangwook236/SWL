#if !defined(__SWL_GRAPHICS__GEOMETRY_POOL__H_)
#define __SWL_GRAPHICS__GEOMETRY_POOL__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include "swl/graphics/Geometry.h"
#include <boost/smart_ptr.hpp>
#include <map>


namespace swl {

//-----------------------------------------------------------------------------------------
// class GeometryPool

class SWL_GRAPHICS_API GeometryPool
{
public:
	//typedef GeometryPool base_type;
	typedef boost::shared_ptr<Geometry> geometry_type;
	typedef size_t geometry_id_type;
	typedef std::map<geometry_id_type, geometry_type> geometry_pool_type;

private:
	GeometryPool()  {}
public:
	~GeometryPool()  {}

public:
	static GeometryPool & getInstance();

	geometry_id_type & createGeometryId();
	void deleteGeometryId(const geometry_id_type &geomId);

	bool setGeometry(const geometry_id_type &geomId, geometry_type &geom);
	geometry_type getGeometry(const geometry_id_type &geomId);
	const geometry_type getGeometry(const geometry_id_type &geomId) const;

private:
	static GeometryPool *singleton_;
	geometry_pool_type geometryPool_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__GEOMETRY_POOL__H_
