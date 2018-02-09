// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <tchar.h>

#include <string>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <iterator>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>  
#include <regex>
#include <unordered_map>
#include <thread>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/range/numeric.hpp>

namespace fs = boost::filesystem;

using string_vec = std::vector<std::string>;