#if !defined(__SWL_GRAPHICS__SCENE_VISITOR_INTERFACE__H_)
#define __SWL_GRAPHICS__SCENE_VISITOR_INTERFACE__H_ 1


namespace swl {

template<typename SceneVistor> class AppearanceSceneNode;
template<typename SceneVistor> class GeometrySceneNode;
template<typename SceneVistor> class ShapeSceneNode;
template<typename SceneVistor> class TransformSceneNode;

//--------------------------------------------------------------------------
// struct ISceneVisitor

struct ISceneVisitor
{
public:
	//typedef ISceneVisitor base_type;
	typedef AppearanceSceneNode<ISceneVisitor> appearance_node_type;
	typedef GeometrySceneNode<ISceneVisitor> geometry_node_type;
	typedef ShapeSceneNode<ISceneVisitor> shape_node_type;
	typedef TransformSceneNode<ISceneVisitor> transform_node_type;

public:
	virtual ~ISceneVisitor()  {}

public:
	virtual void visit(const appearance_node_type &node) const = 0;
	virtual void visit(const geometry_node_type &node) const = 0;
	virtual void visit(const shape_node_type &node) const = 0;

	virtual void visit(const transform_node_type &node) const = 0;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__SCENE_VISITOR_INTERFACE__H_
