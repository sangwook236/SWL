#if !defined(__SWL_GRAPHICS__COLOR__H_)
#define __SWL_GRAPHICS__COLOR__H_ 1


namespace swl {

//--------------------------------------------------------------------------
// struct Color3

template<typename T>
struct Color3
{
	Color3(const T& tRed = T(0), const T& tGreen = T(0), const T& tBlue = T(0))
	: red(tRed), green(tGreen), blue(tBlue)
	{}

	union
	{
        struct { T r, g, b; };
        struct { T red, green, blue; };
        T color[3];
    };
};


//--------------------------------------------------------------------------
// struct Color4

template<typename T>
struct Color4
{
	Color4(const T& tRed = T(0), const T& tGreen = T(0), const T& tBlue = T(0), const T& tAlpha = T(0))
	: red(tRed), green(tGreen), blue(tBlue), alpha(tAlpha)
	{}

    union
	{
        struct { T r, g, b, a; };
        struct { T red, green, blue, alpha; };
        T color[4];
    };
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__COLOR__H_
