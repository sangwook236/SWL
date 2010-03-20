#if !defined(__SWL_BASE__PRINTABLE_INTERFACE__H_)
#define __SWL_BASE__PRINTABLE_INTERFACE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct IPrintable: mix-in style class

struct IPrintable
{
public:
	//typedef IPrintable base_type;

protected:
	virtual ~IPrintable()  {}

public:
	//
	virtual bool print() const = 0;
};

}  // namespace swl


#endif  // __SWL_BASE__PRINTABLE_INTERFACE__H_
