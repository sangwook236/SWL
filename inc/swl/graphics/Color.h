#if !defined(__SWL_GRAPHICS__COLOR__H_)
#define __SWL_GRAPHICS__COLOR__H_ 1


namespace swl {

//--------------------------------------------------------------------------
// struct RGBColor

template<typename T>
struct RGBColor
{
	RGBColor(const T &tRed = T(0), const T &tGreen = T(0), const T &tBlue = T(0))
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
// struct RGBAColor

template<typename T>
struct RGBAColor
{
	RGBAColor(const T &tRed = T(0), const T &tGreen = T(0), const T &tBlue = T(0), const T &tAlpha = T(0))
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
