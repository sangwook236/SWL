#if !defined(__SWL_GRAPHICS__GEOMETRY_POOL_MANAGER__H_)
#define __SWL_GRAPHICS__GEOMETRY_POOL_MANAGER__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include "swl/graphics/Geometry.h"
#include <boost/smart_ptr.hpp>
#include <map>


namespace swl {

//-----------------------------------------------------------------------------------------
// class GeometryPoolMgr

class SWL_GRAPHICS_API GeometryPoolMgr
{
public:
	//typedef GeometryPoolMgr							base_type;
	typedef size_t										geometry_id_type;
	typedef boost::shared_ptr<Geometry>					geometry_type;
	typedef std::map<geometry_id_type, geometry_type>	geometry_pool_type;

public:
	static const geometry_id_type UNDEFINED_GEOMETRY_ID = -1;

private:
	GeometryPoolMgr()  {}
public:
	~GeometryPoolMgr()  {}

private:
	GeometryPoolMgr(const GeometryPoolMgr &rhs);
	GeometryPoolMgr & operator=(const GeometryPoolMgr &rhs);

public:
	static GeometryPoolMgr & getInstance();
	static void clearInstance();

public:
	geometry_id_type createGeometryId();
	void deleteGeometryId(const geometry_id_type &geomId);

	bool setGeometry(const geometry_id_type &geomId, geometry_type &geom);
	geometry_type getGeometry(const geometry_id_type &geomId);
	const geometry_type getGeometry(const geometry_id_type &geomId) const;

private:
	static GeometryPoolMgr *singleton_;

	geometry_pool_type geometryPool_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__GEOMETRY_POOL_MANAGER__H_
