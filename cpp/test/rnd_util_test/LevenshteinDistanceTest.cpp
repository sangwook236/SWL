//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/LevenshteinDistance.h"
#include <iostream>
#include <string>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void levenshtein_distance()
{
	const std::string s1("rosettacode");
	const std::string s2("raisethysword");
	
	const std::size_t distance = swl::computeLevenshteinDistance(s1, s2);

	std::cout << "Levnshtein distance between '" << s1 << "' and '" << s2 << "': " << distance << std::endl;
}
