/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/

#pragma once
#include "stdafx.h"
#include <kaldi-win/stdafx.h>
#include <ctime>
#include "base/kaldi-error.h"
#include "kaldi-win/utility/TwinLoggerMT.h"

//global logging
extern VOICEBRIDGE_API twinLogger::TwinLoggerMT oTwinLog;
#define LOGTW_DEBUG oTwinLog(twinLogger::Debug, __FILE__, __LINE__)
#define LOGTW_INFO oTwinLog(twinLogger::Info)
#define LOGTW_WARNING oTwinLog(twinLogger::Warning)
#define LOGTW_ERROR oTwinLog(twinLogger::Error)
#define LOGTW_FATALERROR oTwinLog(twinLogger::FatalError)

//---Logging override for Windows version of Kaldi -

VOICEBRIDGE_API void ReplaceKaldiLogHandlerEx(bool reset = false);

//--------------------------------------------------

//write to the output window of VS makro
#define DBOUT( s )            \
{                             \
   std::wostringstream os_;    \
   os_ << s;                   \
   OutputDebugStringW( os_.str().c_str() );  \
}

VOICEBRIDGE_API void BindStdHandlesToConsole();

std::string bool_as_text(bool b);
VOICEBRIDGE_API int MergeFiles(std::vector<fs::path> _in, fs::path out);

int CheckFilesExist(std::vector<fs::path> _files, bool report=true);

using StringTable = std::vector<std::vector<std::string>>;
VOICEBRIDGE_API StringTable readData(std::string const path, std::string delimiter = " \t");
VOICEBRIDGE_API int ReadStringTable(std::string const path, StringTable & table, std::string delimiter = " \t");
VOICEBRIDGE_API int SaveStringTable(std::string const& path, StringTable & table);
VOICEBRIDGE_API int SortStringTable(StringTable & table, int col1, int col2, std::string type1 = "string", std::string type2 = "string", bool bMakeUnique=false, bool ascend1=true, bool ascend2=true);
VOICEBRIDGE_API std::string GetFirstLineFromFile(std::string path);
VOICEBRIDGE_API bool IsTheSame(StringTable table1, StringTable table2);

VOICEBRIDGE_API void ReplaceStringInPlace(std::string& subject, const std::string& search, const std::string& replace);
VOICEBRIDGE_API bool ContainsString(std::string subject, const std::string search);

VOICEBRIDGE_API std::string ConvertToUTF8(std::string codepage_str);
bool validate_utf8_whitespaces(std::string str);
VOICEBRIDGE_API std::vector<std::string> instersection(std::vector<std::string> &v1, std::vector<std::string> &v2);
VOICEBRIDGE_API bool IsDisjoint(std::vector<std::string> v1, std::vector<std::string> v2);
VOICEBRIDGE_API bool is_positive_int(const std::string& s);

VOICEBRIDGE_API int is_sortedonfield_and_uniqe(fs::path file, int fieldnr=0, bool bCheckunique=true, bool bShowError = true);
VOICEBRIDGE_API bool ContainsString(StringTable table, std::string s, bool bExactword = false);

VOICEBRIDGE_API bool CopyDir(boost::filesystem::path const & source, boost::filesystem::path const & destination);
VOICEBRIDGE_API std::vector<fs::path> GetFilesInDir(const fs::path &_path, const std::string &extension = "", const bool &recursive = false);
VOICEBRIDGE_API std::vector<fs::path> GetFilesInDir(const fs::path &_path, const std::vector<std::string> &extension, const bool &recursive = false);
VOICEBRIDGE_API std::vector<fs::path> GetDirectories(const fs::path &_path, const std::string &rex);
VOICEBRIDGE_API int CreateDir(fs::path dir, bool deletecontents=true);

template <typename T>
T StringToNumber(const std::string &Text, T defValue = T())
{
	std::stringstream ss;
	for (std::string::const_iterator i = Text.begin(); i != Text.end(); ++i)
		if (isdigit(*i) || *i == 'e' || *i == '-' || *i == '+' || *i == '.')
			ss << *i;
	T result;
	return ss >> result ? result : defValue;
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 1)
{
	std::ostringstream out;
	out << std::fixed << std::setprecision(n) << a_value;
	return out.str();
}

template <typename T>
std::string join_vector(const T& v, const std::string& delim) {
	std::ostringstream s;
	for (const auto& i : v) {
		if (&i != &v[0]) {
			s << delim;
		}
		s << i;
	}
	return s.str();
}

// --------------------------------------------------------------------------------------------------
// With the templates below it is possible to sort an unordered_map on the second field (VALUE) first
// and then on the first field (KEY). ( e.g. std::unordered_map<std::string, int> )
// usage:
// print a sorted (sorted by values and then by the keys) snapshot of the map
//		for (const auto& pair : sorted_snapshot_of(unordered_map))
//			std::cout << pair.first.get() << " " << pair.second.get() << '\n';

template < typename A, typename B >
using wrapped_pair = std::pair< std::reference_wrapper<A>, std::reference_wrapper<B> >;

struct cmp_wrapped_pair
{
	template < typename A, typename B > bool operator() (const A& a, const B& b) const
	{
		return a.second.get() < b.second.get() ||
			(!(b.second.get() < a.second.get()) && a.first.get() < b.first.get());
	}
}; 

template < typename MAP_TYPE >
auto sorted_snapshot_of(const MAP_TYPE& map) // auto: function return type deduction is C++14
{
	using KEY = typename MAP_TYPE::key_type;
	using VALUE = typename MAP_TYPE::mapped_type;
	std::multiset< wrapped_pair< const KEY, const VALUE >, cmp_wrapped_pair > snapshot;

	for (auto& pair : map) snapshot.emplace(pair.first, pair.second);
	return snapshot;
}

// --------------------------------------------------------------------------------------------------

// redirect outputs to another output stream.
class VOICEBRIDGE_API redirect_outputs
{
	std::ostream& myStream;
	std::streambuf *const myBuffer;
public:
	redirect_outputs(std::ostream& lhs, std::ostream& rhs = std::cout)
		: myStream(rhs), myBuffer(myStream.rdbuf())
	{
		myStream.rdbuf(lhs.rdbuf());
	}

	~redirect_outputs() {
		myStream.rdbuf(myBuffer);
	}
};

// redirect output stream to a string.
class VOICEBRIDGE_API capture_outputs
{
	std::ostringstream myContents;
	const redirect_outputs myRedirect;
public:
	capture_outputs(std::ostream& stream = std::cout)
		: myContents(), myRedirect(myContents, stream)
	{}
	std::string contents() const
	{
		return (myContents.str());
	}
};

//----------------------------------------------------------------------------------
/* Check if a bit is set in any type of integer
E.g.: IsBitSet( foo, BIT(3) | BIT(6) );  // Checks if Bit 3 OR 6 is set.
Amongst other things, this approach:

- Accommodates 8/16/32/64 bit integers.
- Detects IsBitSet(int32,int64) calls without my knowledge & consent.
- Inlined Template, so no function calling overhead.
- const& references, so nothing needs to be duplicated/copied. And we are guaranteed that the compiler will pick up any typo's that attempt to change the arguments.
- 0!= makes the code more clear & obvious. The primary point to writing code is always to communicate clearly and efficiently with other programmers, including those of lesser skill.
- While not applicable to this particular case... In general, templated functions avoid the issue of evaluating arguments multiple times. A known problem with some #define macros.
  E.g.: #define ABS(X) (((X)<0) ? - (X) : (X)) ABS(i++);
*/

/* Return type (8/16/32/64 int size) is specified by argument size. */
template<class TYPE> inline TYPE BIT(const TYPE & x)
{
	return TYPE(1) << x;
}

template<class TYPE> inline bool IsBitSet(const TYPE & x, const TYPE & y)
{
	return 0 != (x & y);
}


//----------------------------------------------------------------------------------

//whether the vector only contains unique elements
template <typename T>
bool is_unique(std::vector<T> vec)
{
	std::sort(vec.begin(), vec.end());
	return std::unique(vec.begin(), vec.end()) == vec.end();
}

//----------------------------------------------------------------------------------

inline void SleepWait(clock_t sec) // clock_t is a like typedef unsigned int clock_t. Use clock_t instead of integer in this context
{
	clock_t start_time = clock();
	clock_t end_time = sec * 1000 + start_time;
	while (clock() != end_time);
}

//----------------------------------------------------------------------------------

//This code will make a map sorted by value instead of by key.
/*	std::map<int, double> src;
	...
	std::multimap<double, int> dst = flip_map(src);
	//dst is now sorted by what used to be the value in src! */

template <typename A, typename B>
std::multimap<B, A> flip_map(std::map<A, B> & src) {

	std::multimap<B, A> dst;

	for (std::map<A, B>::const_iterator it = src.begin(); it != src.end(); ++it)
		dst.insert(std::pair<B, A>(it->second, it->first));

	return dst;
}
//----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------------------
//	GZIP/GUNZIP Library API
//-----------------------------------------------------------------------------------------------------------------
VOICEBRIDGE_API int CompressFileGz(fs::path file);
VOICEBRIDGE_API int DecompressFileGz(fs::path file);

//-----------------------------------------------------------------------------------------------------------------
//ZIP/UNZIP library API
//-----------------------------------------------------------------------------------------------------------------
VOICEBRIDGE_API int Zip(fs::path p);
VOICEBRIDGE_API int UnZip(fs::path p);


//-----------------------------------------------------------------------------------------------------------------
//BOOST Wildcard file actions library API
//-----------------------------------------------------------------------------------------------------------------
VOICEBRIDGE_API void GetAllMatchingFiles(std::vector< fs::path > & all_matching_files, fs::path dir, boost::regex regex_filename);
VOICEBRIDGE_API int CopyAllMatching(fs::path dir_source, fs::path dir_destination, boost::regex regex_filename);
int DeleteAllMatching(fs::path dir_source, boost::regex regex_filename);

//-----------------------------------------------------------------------------------------------------------------
//General purpose helpers
//-----------------------------------------------------------------------------------------------------------------
VOICEBRIDGE_API int FilterFile(fs::path in, fs::path out, std::unordered_map<std::string, std::string> filter);
std::string ConvertToCaseUtf8(std::string & s, bool toUpper=false);

