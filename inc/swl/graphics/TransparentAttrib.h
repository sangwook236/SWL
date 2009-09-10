#if !defined(__SWL_GRAPHICS__TRANSPARENT_ATTRIBUTE__H_)
#define __SWL_GRAPHICS__TRANSPARENT_ATTRIBUTE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct TransparentAttrib

struct TransparentAttrib
{
public:
	TransparentAttrib(const bool isTransparent = false)
	: isTransparent_(isTransparent)
	{}
	TransparentAttrib(const TransparentAttrib &rhs)
	: isTransparent_(rhs.isTransparent_)
	{}
	~TransparentAttrib()
	{}

	TransparentAttrib & operator=(const TransparentAttrib &rhs)
	{
		if (this == &rhs) return *this;
		//static_cast<base_type &>(*this) = rhs;
		isTransparent_ = rhs.isTransparent_;
		return *this;
	}

public:
	/// mutator
	void setTransparent(bool isTransparent)  {  isTransparent_ = isTransparent;  }
	/// accessor
	bool isTransparent() const  {  return isTransparent_;  }

private:
	bool isTransparent_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__TRANSPARENT_ATTRIBUTE__H_
