#if !defined(__SWL_GRAPHICS__SCENE_VISITOR_INTERFACE__H_)
#define __SWL_GRAPHICS__SCENE_VISITOR_INTERFACE__H_ 1


namespace swl {

class AppearanceSceneNode;
class GeometrySceneNode;
class TransformSceneNode;

//--------------------------------------------------------------------------
// struct ISceneVisitor

struct ISceneVisitor
{
public:
	//typedef ISceneVisitor base_type;

public:
	virtual ~ISceneVisitor()  {}

public:
	virtual void visit(const AppearanceSceneNode &node) const = 0;
	virtual void visit(const GeometrySceneNode &node) const = 0;
	virtual void visit(const TransformSceneNode &node) const = 0;

};

}  // namespace swl


#endif  // __SWL_GRAPHICS__SCENE_VISITOR_INTERFACE__H_
