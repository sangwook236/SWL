#if !defined(__SWL_GL_UTIL__GL_SHAPE_SCENE_NODE__H_)
#define __SWL_GL_UTIL__GL_SHAPE_SCENE_NODE__H_ 1


#include "swl/graphics/SceneNode.h"
#include "swl/glutil/GLShape.h"


namespace swl {

//--------------------------------------------------------------------------
// class GLShapeSceneNode

template<typename SceneVisitor>
class GLShapeSceneNode: public LeafSceneNode<SceneVisitor>
{
public:
	typedef LeafSceneNode<SceneVisitor> base_type;
	typedef boost::shared_ptr<GLShape> shape_type;
	typedef GLShape::geometry_type geometry_type;
	typedef GLShape::appearance_type appearance_type;
	typedef typename base_type::visitor_type visitor_type;

public:
#if defined(_UNICODE) || defined(UNICODE)
	GLShapeSceneNode(shape_type &shape, const std::wstring &name = std::wstring())
#else
	GLShapeSceneNode(shape_type &shape, const std::string &name = std::string())
#endif
	: base_type(name),
	  shape_(shape)
	{}
	GLShapeSceneNode(const GLShapeSceneNode &rhs)
	: base_type(rhs),
	  shape_(rhs.shape_)
	{}
	virtual ~GLShapeSceneNode()
	{}

	GLShapeSceneNode & operator=(const GLShapeSceneNode &rhs)
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


#endif  // __SWL_GL_UTIL__GL_SHAPE_SCENE_NODE__H_
