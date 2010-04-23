#if !defined(__SWL_GRAPHICS__OBJECT_PICKER_MANAGER__H_)
#define __SWL_GRAPHICS__OBJECT_PICKER_MANAGER__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include "swl/graphics/Color.h"
#include <set>


namespace swl {

//-----------------------------------------------------------------------------------------
//

class SWL_GRAPHICS_API ObjectPickerMgr
{
public:
	//typedef ObjectPickerMgr base_type;
	typedef unsigned int object_id_type;
	typedef std::set<object_id_type> picked_objects_type;
	typedef Color4<float> color_type;

private:
	ObjectPickerMgr()
	: pickedObjectIds_(), temporarilyPickedObjectIds_(), isPicking_(false), pickedColor_(1.0f, 1.0f, 0.0f, 1.0f), temporarilyPickedColor_(1.0f, 0.2f, 0.6f, 1.0f)
	{}
public:
	~ObjectPickerMgr()  {}

private:
	ObjectPickerMgr(const ObjectPickerMgr &rhs);
	ObjectPickerMgr & operator=(const ObjectPickerMgr &rhs);

public:
	static ObjectPickerMgr & getInstance();
	static void clearInstance();

public:
	bool addPickedObject(const object_id_type &id)
	{  return pickedObjectIds_.insert(id).second;  }
	void removePickedObject(const object_id_type &id)
	{  pickedObjectIds_.erase(id);  }
	void clearAllPickedObjects()  {  pickedObjectIds_.clear();  }
	size_t countPickedObject() const  {  return pickedObjectIds_.size();  }
	bool containPickedObject() const  {  return !pickedObjectIds_.empty();  }

	bool isPickedObject(const object_id_type &id) const
	{  return pickedObjectIds_.end() != pickedObjectIds_.find(id);  }

	const picked_objects_type & getPickedObjects() const  {  return pickedObjectIds_;  }

	//
	bool addTemporarilyPickedObject(const object_id_type &id)
	{  return temporarilyPickedObjectIds_.insert(id).second;  }
	void clearAllTemporarilyPickedObjects()  {  temporarilyPickedObjectIds_.clear();  }
	bool containTemporarilyPickedObject() const  {  return !temporarilyPickedObjectIds_.empty();  }

	bool isTemporarilyPickedObject(const object_id_type &id) const
	{  return temporarilyPickedObjectIds_.end() != temporarilyPickedObjectIds_.find(id);  }

	//
	void startPicking()
	{
		temporarilyPickedObjectIds_.clear();
		isPicking_ = true;
	}
	void stopPicking()
	{
		isPicking_ = false;
		temporarilyPickedObjectIds_.clear();
	}
	bool isPicking() const  {  return isPicking_;  }

	//
	void setPickedColor(const color_type &color)  {  pickedColor_ = color;  }
	const color_type & getPickedColor() const  {  return pickedColor_;  }

	void setTemporarilyPickedColor(const color_type &color)  {  temporarilyPickedColor_ = color;  }
	const color_type & getTemporarilyPickedColor() const  {  return temporarilyPickedColor_;  }

private:
	static ObjectPickerMgr *singleton_;

	picked_objects_type pickedObjectIds_;
	picked_objects_type temporarilyPickedObjectIds_;
	bool isPicking_;
	color_type pickedColor_;
	color_type temporarilyPickedColor_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__OBJECT_PICKER_MANAGER__H_
