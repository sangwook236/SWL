#if !defined(__SWL_GL_UTIL__GL_SCENE_VISITOR_INTERFACE__H_)
#define __SWL_GL_UTIL__GL_SCENE_VISITOR_INTERFACE__H_ 1


namespace swl {

template<typename SceneVistor> class AppearanceSceneNode;
template<typename SceneVistor> class GeometrySceneNode;
template<typename SceneVistor> class GLShapeSceneNode;
template<typename SceneVistor> class TransformSceneNode;

//--------------------------------------------------------------------------
// struct IGLSceneVisitor

struct IGLSceneVisitor
{
public:
	//typedef IGLSceneVisitor base_type;
	typedef AppearanceSceneNode<IGLSceneVisitor> appearance_node_type;
	typedef GeometrySceneNode<IGLSceneVisitor> geometry_node_type;
	typedef GLShapeSceneNode<IGLSceneVisitor> shape_node_type;
	typedef TransformSceneNode<IGLSceneVisitor> transform_node_type;

public:
	virtual ~IGLSceneVisitor()  {}

public:
	virtual void visit(const appearance_node_type &node) const = 0;
	virtual void visit(const geometry_node_type &node) const = 0;
	virtual void visit(const shape_node_type &node) const = 0;

	virtual void visit(const transform_node_type &node) const = 0;
};

}  // namespace swl


#endif  // __SWL_GL_UTIL__GL_SCENE_VISITOR_INTERFACE__H_
