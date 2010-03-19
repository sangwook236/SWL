#if !defined(__SWL_BASE__VISITABLE_INTERFACE__H_)
#define __SWL_BASE__VISITABLE_INTERFACE__H_ 1


namespace swl {

//--------------------------------------------------------------------------
// struct IVisitable

template<typename Visitor>
struct IVisitable
{
public:
	//typedef IVisitable base_type;
	typedef Visitor visitor_type;

public:
	virtual ~IVisitable()  {}

public:
	virtual void accept(const visitor_type &visitor) const = 0;
};

}  // namespace swl


#endif  // __SWL_BASE__VISITABLE_INTERFACE__H_
