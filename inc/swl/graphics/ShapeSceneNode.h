#if !defined(__SWL_GRAPHICS__SHAPE_SCENE_NODE__H_)
#define __SWL_GRAPHICS__SHAPE_SCENE_NODE__H_ 1


#include "swl/graphics/SceneNode.h"
#include "swl/graphics/Shape.h"


namespace swl {

//--------------------------------------------------------------------------
// class ShapeSceneNode

class SWL_GRAPHICS_API ShapeSceneNode: public LeafSceneNode
{
public:
	typedef LeafSceneNode				base_type;
	typedef boost::shared_ptr<Shape>	shape_type;
	typedef Shape::geometry_type		geometry_type;
	typedef Shape::appearance_type		appearance_type;

public:
	ShapeSceneNode(shape_type &shape);
	ShapeSceneNode(const ShapeSceneNode &rhs);
	virtual ~ShapeSceneNode();

	ShapeSceneNode & operator=(const ShapeSceneNode &rhs);
 
public:
	/*final*/ /*virtual*/ void accept(const ISceneVisitor &visitor) const;

	shape_type & getShape()  {  return shape_;  }
	const shape_type & getShape() const  {  return shape_;  }

private:
	shape_type shape_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__SHAPE_SCENE_NODE__H_
