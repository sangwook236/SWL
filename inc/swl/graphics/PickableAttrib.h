#if !defined(__SWL_GRAPHICS__PICKABLE_ATTRIBUTE__H_)
#define __SWL_GRAPHICS__PICKABLE_ATTRIBUTE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct PickableAttrib

struct PickableAttrib
{
public:
	PickableAttrib(const bool isPickable = true)
	: isPickable_(isPickable)
	{}
	PickableAttrib(const PickableAttrib &rhs)
	: isPickable_(rhs.isPickable_)
	{}
	~PickableAttrib()
	{}

	PickableAttrib & operator=(const PickableAttrib &rhs)
	{
		if (this == &rhs) return *this;
		//static_cast<base_type &>(*this) = rhs;
		isPickable_ = rhs.isPickable_;
		return *this;
	}

public:
	/// mutator
	void setPickable(bool isPickable)  {  isPickable_ = isPickable;  }
	/// accessor
	bool isPickable() const {  return isPickable_;  }

private:
	bool isPickable_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__PICKABLE_ATTRIBUTE__H_
