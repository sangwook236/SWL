#if !defined(__SWL_GRAPHICS__SHAPE_SCENE_NODE__H_)
#define __SWL_GRAPHICS__SHAPE_SCENE_NODE__H_ 1


#include "swl/graphics/SceneNode.h"
#include "swl/graphics/Shape.h"


namespace swl {

//--------------------------------------------------------------------------
// class ShapeSceneNode

template<typename SceneVisitor>
class ShapeSceneNode: public LeafSceneNode<SceneVisitor>
{
public:
	typedef LeafSceneNode				base_type;
	typedef boost::shared_ptr<Shape>	shape_type;
	typedef Shape::geometry_type		geometry_type;
	typedef Shape::appearance_type		appearance_type;

public:
#if defined(UNICODE) || defined(_UNICODE)
	ShapeSceneNode(shape_type &shape, const std::wstring &name = std::wstring())
#else
	ShapeSceneNode(shape_type &shape, const std::string &name = std::string())
#endif
	: base_type(name),
	  shape_(shape)
	{}
	ShapeSceneNode(const ShapeSceneNode &rhs)
	: base_type(rhs),
	  shape_(rhs.shape_)
	{}
	virtual ~ShapeSceneNode()
	{}

	ShapeSceneNode & operator=(const ShapeSceneNode &rhs)
	{
		if (this == &rhs) return *this;
		static_cast<base_type &>(*this) = rhs;
		shape_ = rhs.shape_;
		return *this;
	}

public:
	/*final*/ /*virtual*/ void accept(const visitor_type &visitor) const
	{
		visitor.visit(*this);
	}

	shape_type & getShape()  {  return shape_;  }
	const shape_type & getShape() const  {  return shape_;  }

private:
	shape_type shape_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__SHAPE_SCENE_NODE__H_
