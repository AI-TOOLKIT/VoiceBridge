/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/

#include "Utility.h"

//global logging
VOICEBRIDGE_API twinLogger::TwinLoggerMT oTwinLog; 

//--- Logging override for Windows version of Kaldi --------------->

//NOTE: this code replaces the Kaldi main logging meachanism with the new Kaldi Windows global twin logging mechanism
void HandleKaldiLogMessage(const kaldi::LogMessageEnvelope &envelope, const char *message)
{
	switch (envelope.severity) {
	case kaldi::LogMessageEnvelope::kInfo:
		oTwinLog(twinLogger::Info) << message;
		break;
	case kaldi::LogMessageEnvelope::kWarning:
		oTwinLog(twinLogger::Warning) << message;
		break;
	case kaldi::LogMessageEnvelope::kError:
		oTwinLog(twinLogger::Error) << message;
		break;
	case kaldi::LogMessageEnvelope::kAssertFailed:
		oTwinLog(twinLogger::Debug) << message;
		break;
	default:
		oTwinLog(twinLogger::Info) << message;  // coding error (unknown 'severity'),
	}
}

VOICEBRIDGE_API void ReplaceKaldiLogHandlerEx(bool reset)
{
	if (!reset) {
		kaldi::SetLogHandler(NULL);
	}
	else {
		kaldi::SetLogHandler(HandleKaldiLogMessage);
	}
}

//<------------------- Logging override for Windows version of Kaldi


//replaces all occurances of the 'search' string with 'replace' string in the 'subject' string and in place
VOICEBRIDGE_API void ReplaceStringInPlace(std::string& subject, const std::string& search, const std::string& replace)
{
	size_t pos = 0;
	while ((pos = subject.find(search, pos)) != std::string::npos) {
		subject.replace(pos, search.length(), replace);
		pos += replace.length();
	}
}

VOICEBRIDGE_API bool ContainsString(std::string subject, const std::string search) {
	return (subject.find(search, 0) != std::string::npos);
}

VOICEBRIDGE_API std::string GetFirstLineFromFile(std::string path)
{
	std::ifstream ifs(path);
	if (!ifs) {
		throw std::runtime_error("Error opening file " + path);
	}
	std::string line;
	std::getline(ifs, line);
	return line;
}

//-------------------------------------------------------------------------------------------------------------------

VOICEBRIDGE_API StringTable readData(std::string const path, std::string delimiter)
{
	std::ifstream ifs(path);
	if (!ifs) {
		throw std::runtime_error("Error opening file.");
	}

	StringTable table;
	std::string line;
	while (std::getline(ifs, line))	{
		std::vector<std::string> _w;
		boost::algorithm::trim(line);
		strtk::parse(line, delimiter, _w, strtk::split_options::compress_delimiters);
		table.emplace_back(_w);
	}
	return table;
}


VOICEBRIDGE_API int ReadStringTable(std::string const path, StringTable & table, std::string delimiter)
{
	try {
		table = readData(path, delimiter);
		return 0;
	}
	catch (std::exception const& e) {
		LOGTW_FATALERROR << " " << e.what() << ". (Reading file " << path << ")";
		return -1;
	}
	catch (...) {
		LOGTW_FATALERROR << " Unknown Error. (Reading file " << path << ")";
		return -1;
	}
}


VOICEBRIDGE_API int SaveStringTable(std::string const& path, StringTable & table)
{
try {
	//NOTE: ofstream write adds an '\r' in front of the '\n' automatically and some code in Kaldi crashes. To prevent this
	//		add std::ios::binary option to each ofstream to make sure that '\r' is not added!		
	fs::ofstream file_(path, std::ios::binary | std::ios::out);
	if (!file_) {
		LOGTW_ERROR << " can't open output file: " << path << ".";
		return -1;
	}

	for (StringTable::const_iterator it(table.begin()), it_end(table.end()); it != it_end; ++it)
	{
		int c = 0;
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
		{
			if (c > 0) file_ << " "; //Separator!
			file_ << *itc1;
			c++;
		}
		file_ << '\n';
	}
	file_.flush(); file_.close();
	return 0;
}
catch (std::exception const& e) {
	LOGTW_FATALERROR << " " << e.what() << ".";
	return -1;
}
catch (...) {
	LOGTW_FATALERROR << " Unknown Error.";
	return -1;
}
}

//comparer function for StringTable short : max 2 columns sort
class StringTableSortCompare2
{
public:
	//fCol and sCol are "string" or "number"
	explicit StringTableSortCompare2(int column1, int column2, std::string fCol, std::string sCol, bool ascend1, bool ascend2)
		: m_column1(column1), m_column2(column2), fColType(fCol), sColType(sCol), m_ascend1(ascend1), m_ascend2(ascend2){}
	bool operator()(const std::vector<std::string>& lhs, const std::vector<std::string>& rhs)
	{
		//sort by column2 after sorting by column1; if column1 == column2 then makes no difference
		if (lhs[m_column1] == rhs[m_column1]) {
			if (m_ascend2) {
				if (sColType.compare("string") == 0)
					return lhs[m_column2] < rhs[m_column2];
				else if (sColType.compare("number") == 0)
					return StringToNumber<double>(lhs[m_column2].c_str()) < StringToNumber<double>(rhs[m_column2].c_str());
				else return lhs[m_column2] < rhs[m_column2];
			}
			else {
				if (sColType.compare("string") == 0)
					return lhs[m_column2] > rhs[m_column2];
				else if (sColType.compare("number") == 0)
					return StringToNumber<double>(lhs[m_column2].c_str()) > StringToNumber<double>(rhs[m_column2].c_str());
				else return lhs[m_column2] > rhs[m_column2];
			}
		}
		else {
			if (m_ascend1) {
				if (fColType.compare("string") == 0)
					return lhs[m_column1] < rhs[m_column1];
				else  if (fColType.compare("number") == 0)
					return StringToNumber<double>(lhs[m_column1].c_str()) < StringToNumber<double>(rhs[m_column1].c_str());
				else return lhs[m_column1] < rhs[m_column1];
			}
			else {
				if (fColType.compare("string") == 0)
					return lhs[m_column1] > rhs[m_column1];
				else  if (fColType.compare("number") == 0)
					return StringToNumber<double>(lhs[m_column1].c_str()) > StringToNumber<double>(rhs[m_column1].c_str());
				else return lhs[m_column1] > rhs[m_column1];
			}
		}
	}
private:
	int m_column1;
	int m_column2;
	std::string fColType;
	std::string sColType;
	bool m_ascend1;
	bool m_ascend2;
};

/*
	In place short of a StringTable by col1 and then by col2; col1 can be = col2.
	When requested bMakeUnique makes the table unique (erashes duplicate) by the first column (unique first column!)
*/
VOICEBRIDGE_API int SortStringTable(StringTable & table, 
	int col1, int col2, 
	std::string type1, std::string type2, bool bMakeUnique,
	bool ascend1, bool ascend2)
{
	if (table[0].size() < col1 + 1 || table[0].size() < col2 + 1)
	{
		LOGTW_ERROR << " The string table size does not much the requested sort column index.";
		return -1;
	}
	StringTableSortCompare2 sort_compare(col1, col2, type1, type2, ascend1, ascend2);
	std::sort(table.begin(), table.end(), sort_compare);

	if (bMakeUnique) 
	{
		//IMPORTANT NOTE: when unique is requested we always mean unique by the first supplied column (columns[0]) 
		//				  and not the whole line! The whole line will be removed from the StringTable.
		int line = 0;
		StringTable::const_iterator it = table.begin();
		while ( it != table.end() ) {
			if (line == 0) { line++; ++it; continue; }
			if ((*it)[col1] == (*(it - 1))[col1]) {
				//cout << "deleting line: " << (*it)[0] << "\n";
				it = table.erase(it); 
			}
			else {
				++it;
			}
		}
		//NOTE: erase() invalidates the current iterator but returns a new one to the next element 
		//		which can be used further! However, if erase() is not used the iterator must be incremented!
	}

	return 0;
}

//fieldnr: zero based index of the columns!
VOICEBRIDGE_API int is_sortedonfield_and_uniqe(fs::path file, int fieldnr, bool bCheckunique, bool bShowError)
{
	StringTable table_;
	std::vector<std::string> _v;
	if (ReadStringTable(file.string(), table_) < 0) {
		LOGTW_ERROR << " fail to open: " << file.string() << ".";
		return -3;
	}
	for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it) {
		if ((*it).size() < (fieldnr + 1)) return -1;
		_v.push_back((*it)[fieldnr]);
	}

	if (!std::is_sorted(_v.begin(), _v.end())) {
		if(bShowError)
			LOGTW_ERROR << " the file is not sorted on field nr " << (fieldnr + 1) << " in file " << file.string() << ".";
		return -2;
	}
	if (bCheckunique && !(std::unique(_v.begin(), _v.end()) == _v.end())) {
		if (bShowError)
			LOGTW_ERROR << " the file contains duplicate elements: " << file.string() << ".";
		return -1;
	}

	return 0;
}

//checks the whole table if the string s is present; 
//in case bExactword is true it checks for the whole field instead of contents
VOICEBRIDGE_API bool ContainsString(StringTable table, std::string s, bool bExactword)
{
	for (StringTable::const_iterator it(table.begin()), it_end(table.end()); it != it_end; ++it) {
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) {
			if (bExactword) {
				if (*itc1 == s) return true;
			}
			else {
				if ((*itc1).find(s) != std::string::npos) {
					return true;
				}
			}
		}
	}
	return false;
}

//element wise comparision of 2 StringTable's
VOICEBRIDGE_API bool IsTheSame(StringTable table1, StringTable table2)
{
	if (table1.size() != table2.size()) return false;
	for (int i = 0; i < table1.size(); i++) {
		if (table1[i].size() != table2[i].size()) return false;
		if (!std::equal(std::begin(table1[i]), std::end(table1[i]), std::begin(table2[i])))
			return false;
	}
	return true;
}

//-------------------------------------------------------------------------------------------------------------------

///https://stackoverflow.com/questions/311955/redirecting-cout-to-a-console-in-windows
VOICEBRIDGE_API void BindStdHandlesToConsole()
{
	// Redirect the CRT standard input, output, and error handles to the console
	freopen("CONIN$", "r", stdin);
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);

	//Clear the error state for each of the C++ standard stream objects. We need to do this, as
	//attempts to access the standard streams before they refer to a valid target will cause the
	//iostream objects to enter an error state. In versions of Visual Studio after 2005, this seems
	//to always occur during startup regardless of whether anything has been read from or written to
	//the console or not.
	std::wcout.clear();
	std::cout.clear();
	std::wcerr.clear();
	std::cerr.clear();
	std::wcin.clear();
	std::cin.clear();
}

//converts the string to UTF-8
//Returns an empty string if fails
VOICEBRIDGE_API std::string ConvertToUTF8(std::string codepage_str)
{
	int size = MultiByteToWideChar(CP_ACP, MB_COMPOSITE, codepage_str.c_str(), (int)codepage_str.length(), nullptr, 0);
	std::wstring utf16_str(size, '\0');
	int ret = MultiByteToWideChar(CP_ACP, MB_COMPOSITE, codepage_str.c_str(), (int)codepage_str.length(), &utf16_str[0], size);
	if (ret == 0) return "";
	int utf8_size = WideCharToMultiByte(CP_UTF8, 0, utf16_str.c_str(), (int)utf16_str.length(), nullptr, 0, nullptr, nullptr);
	if (utf8_size == 0) return "";
	std::string utf8_str(utf8_size, '\0');
	int ret1 = WideCharToMultiByte(CP_UTF8, 0, utf16_str.c_str(), (int)utf16_str.length(), &utf8_str[0], utf8_size, nullptr, nullptr);
	if (ret1 == 0) return "";

	return utf8_str;
}

//returns true if the str contains characters besides space, tab, newline
bool validate_utf8_whitespaces(std::string str) 
{
	std::string sallow(" \t\r\n");
	if (str.find_first_not_of(sallow) != std::string::npos)
	{
		// There's a non-space. OK!
		return true;
	}
	return false;
}

//returns a vector with the elements which are in both input vectors; it returns an empty vector if there are no the same elements in both
VOICEBRIDGE_API std::vector<std::string> instersection(std::vector<std::string> &v1, std::vector<std::string> &v2)
{
	std::vector<std::string> v3;
	sort(v1.begin(), v1.end());
	sort(v2.begin(), v2.end());
	set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v3));
	return v3;
}

//checks if two string vectors has overlap
// returns true if there is no overlap. false if there is overlap.
VOICEBRIDGE_API bool IsDisjoint(std::vector<std::string> v1, std::vector<std::string> v2)
{
	std::vector<std::string> _v = instersection(v1, v2);
	if (_v.size() > 0) {
		return false;
	}
	return true;
}


VOICEBRIDGE_API bool is_positive_int(const std::string& s)
{
	return !s.empty() && std::find_if(s.begin(),
		s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}


//Copies the complete source directory to the destination directory (files and folders). The source directory will
//not be a subdirectory but the contents is copied!
VOICEBRIDGE_API bool CopyDir(boost::filesystem::path const & source, boost::filesystem::path const & destination)
{
	try
	{
		// Check whether the function call is valid
		if (!boost::filesystem::exists(source) ||
			!boost::filesystem::is_directory(source))
		{
			LOGTW_ERROR << "Source directory " << source.string() << " does not exist or is not a directory.";
			return false;
		}

		if (boost::filesystem::exists(destination))
		{
			LOGTW_ERROR << "Destination directory " << destination.string() << " already exists.";
			return false;
		}

		// Create the destination directory
		if (!boost::filesystem::create_directory(destination))
		{
			LOGTW_ERROR << "Unable to create destination directory" << destination.string();
			return false;
		}
	}

	catch (boost::filesystem::filesystem_error const & e)
	{
		LOGTW_ERROR << e.what();
		return false;
	}

	// Iterate through the source directory
	for (boost::filesystem::directory_iterator file(source);
		file != boost::filesystem::directory_iterator();
		++file)
	{
		try
		{
			boost::filesystem::path current(file->path());
			if (boost::filesystem::is_directory(current))
			{
				// Found directory: Recursion
				if (!CopyDir(current, destination / current.filename()))
				{
					return false;
				}
			}
			else
			{
				// Found file: Copy
				boost::filesystem::copy_file(current, destination / current.filename());
			}
		}
		catch (boost::filesystem::filesystem_error const & e)
		{
			LOGTW_ERROR << e.what();
		}
	}
	return true;
}

VOICEBRIDGE_API int CreateDir(fs::path dir, bool deletecontents) {
	try {
		if (fs::exists(dir)) {
			if (deletecontents) {
				//in order to remove only the contents but no the directory:
				for (fs::directory_iterator end_dir_it, it(dir); it != end_dir_it; ++it) {
					fs::remove_all(it->path());
				}
			}
		}
		else {
			if (!fs::create_directories(dir)) {
				LOGTW_ERROR << " could not create directory " << dir.string() << ".";
				return -1;
			}
		}
	}
	catch (const std::exception&) {
		LOGTW_ERROR << " could not create directory " << dir.string() << ".";
		return -1;
	}
	return 0;
}


//[start] get directory contents ---------------------------------------------------------------------------------------------
// usage
//std::vector<std::string> exts{ ".jpg", ".png" };
//std::vector<std::string> files = getFilesInDir("/path/to/root/directory/", exts, true);

/**
* @note If extension is empty string ("") then all files are returned
* @param Path root directory
* @param extension String or vector of strings to filter - empty string returns all files - default: ""
* @param recursive Set true to scan directories recursively - default false
* @return Vector of strings with full paths
*/
VOICEBRIDGE_API std::vector<fs::path> GetFilesInDir(const fs::path &_path, const std::string &extension, const bool &recursive)
{
	std::vector<fs::path> files;
	//boost::filesystem::path dir(_path);
	if (boost::filesystem::exists(_path) && boost::filesystem::is_directory(_path)) {

		if (recursive) {
			boost::filesystem::recursive_directory_iterator it(_path);
			boost::filesystem::recursive_directory_iterator endit;
			while (it != endit) {
				if (boost::filesystem::is_regular_file(*it) && (extension == "") ? true : it->path().extension() == extension) {
					files.push_back(it->path());
				}
				++it;
			}
		}
		else {
			boost::filesystem::directory_iterator it(_path);
			boost::filesystem::directory_iterator endit;
			while (it != endit) {
				if (boost::filesystem::is_regular_file(*it) && (extension == "") ? true : it->path().extension() == extension) {
					files.push_back(it->path());
				}
				++it;
			}
		}
	}
	return files;
}

//the same as above but with several file extensions
VOICEBRIDGE_API std::vector<fs::path> GetFilesInDir(const fs::path &_path, const std::vector<std::string> &extension, const bool &recursive) {
	if (extension.size() <= 0) return GetFilesInDir(_path, "", recursive);
	std::vector<fs::path> outArray;
	for (const std::string &ext : extension) {
		std::vector<fs::path> files = GetFilesInDir(_path, ext, recursive);
		outArray.insert(outArray.end(), files.begin(), files.end());
	}
	return outArray;
}

//this version gets the directories in a given dir which correspond to the given regex
VOICEBRIDGE_API std::vector<fs::path> GetDirectories(const fs::path &_path, const std::string &rex)
{
	std::vector<fs::path> files;
	if (boost::filesystem::exists(_path) && boost::filesystem::is_directory(_path)) 
	{
		const boost::regex re(rex);
		boost::match_results<std::string::const_iterator> re_res;
		boost::filesystem::directory_iterator it(_path);
		boost::filesystem::directory_iterator endit;
		while (it != endit) {
			std::string dirname = (*it).path().filename().string();
			if (boost::filesystem::is_directory(*it) && (rex == "") ? true : boost::regex_match(dirname, re_res, re))
			{
				files.push_back(it->path());
			}
			++it;
		}
	}
	return files;
}

//[end] get directory contents ---------------------------------------------------------------------------------------------

std::string bool_as_text(bool b)
{
	std::stringstream converter;
	converter << b;
	return converter.str();
}

VOICEBRIDGE_API int MergeFiles(std::vector<fs::path> _in, fs::path out)
{
	try
	{
		fs::ofstream of_out(out, std::ios_base::binary | std::ios_base::out);
		if (!of_out) {
			LOGTW_ERROR << " Could not merge files because output file is not accessible.";
			return -1;
		}
		for each(fs::path p in _in)
		{
			fs::ifstream if_(p, std::ios_base::binary);
			of_out << if_.rdbuf();
		}
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << " Could not merge files. Reason: " << ex.what() << ".";
		return -1;
	}
	return 0;
}

//-----------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------------------
//	GZIP/GUNZIP Library API
//-----------------------------------------------------------------------------------------------------------------
//NOTE: boost+gzip only supports archivig 1 file! It can not zip/unzip a directory tree.
//include section for gzip:

//TODO:... not tested

#include <boost/iostreams/device/file.hpp> 
#include <boost/iostreams/filtering_stream.hpp> 
#include <boost/iostreams/stream.hpp> 
#include <boost/iostreams/copy.hpp> 
#include <boost/iostreams/filter/gzip.hpp> 
namespace io = boost::iostreams;

VOICEBRIDGE_API int CompressFileGz(fs::path file)
{
	//NOTE: gzip does not work without specifying 'fs::ofstream::binary | fs::ofstream::in' for both in/out files!
	try {
		io::file_source infile(file.string(), fs::ofstream::binary | fs::ofstream::in);
		io::filtering_istream fis;
		io::gzip_compressor gzip;
		fis.push(gzip);
		fis.push(infile);
		io::file_sink outfile(file.string() + ".gz", fs::ofstream::binary | fs::ofstream::out);
		io::stream<io::file_sink> os(outfile);
		io::copy(fis, os);

	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << ex.what();
		return -1;
	}
	return 0;
}

VOICEBRIDGE_API int DecompressFileGz(fs::path file)
{
	//NOTE: gzip does not work without specifying 'fs::ofstream::binary | fs::ofstream::in' for both in/out files!
	try {
		io::file_source infile(file.string(), fs::ofstream::binary | fs::ofstream::in);
		io::filtering_istream fis;
		io::gzip_decompressor ungzip;
		fis.push(ungzip);
		fis.push(infile);
		std::string s(file.string());
		ReplaceStringInPlace(s, ".gz", "");
		io::file_sink outfile(s, fs::ofstream::binary | fs::ofstream::out);
		io::stream<io::file_sink> os(outfile);
		boost::iostreams::copy(fis, os);
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << ex.what();
		return -1;
	}
	return 0;
}

//-----------------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------------------
//ZIP/UNZIP library API
//-----------------------------------------------------------------------------------------------------------------
#include <zipper.h>
#include <unzipper.h>

//recursive function for adding several files to a ZIP file
int AddFileToZipArchive(ziputils::zipper & zipFile, fs::path p, fs::path ziproot)
{
	if (fs::is_directory(p))
	{//directory
	 // Iterate through the source directory
		for (boost::filesystem::directory_iterator file(p); file != boost::filesystem::directory_iterator(); ++file) {
			try {
				boost::filesystem::path current(file->path());
				if (boost::filesystem::is_directory(current))
				{
					// Found directory: Recursion
					//current
					if (AddFileToZipArchive(zipFile, current, ziproot) < 0) return -1;
				}
				else
				{
					// Found file: Add to ZIP
					//current
					std::ifstream file(current.string(), std::ios::in | std::ios::binary);
					if (file.is_open())
					{
						//LOGTW_INFO << "Adding file " << current.string();
						//must add the relative path to the archive's root folder!
						std::string relpath((fs::relative(current, ziproot)).string());

						zipFile.addEntry(relpath.c_str());
						zipFile << file;
					}
				}
			}
			catch (boost::filesystem::filesystem_error const & e)
			{
				LOGTW_ERROR << e.what() << '\n';
				return -1;
			}
		}
	}
	else
	{//1 file
		std::ifstream file(p.string(), std::ios::in | std::ios::binary);
		if (file.is_open())
		{
			//LOGTW_INFO << "Adding file " << p.string();
			//must add the relative path to the archive's root folder!
			std::string relpath((fs::relative(p, ziproot)).string());

			zipFile.addEntry(relpath.c_str());
			zipFile << file;
		}
	}

	return 0;
}

//If the input is a folder it will ZIP recursively all directories and files.
//If the input is a file it will ZIP the file.
VOICEBRIDGE_API int Zip(fs::path p)
{
	if (!fs::exists(p)) {
		LOGTW_ERROR << "The file " << p.string() << " does not exist.";
		return -1;
	}

	try	{
		ziputils::zipper zipFile;
		fs::path outpath(p.parent_path() / (p.filename().stem().string() + ".zip"));

		LOGTW_INFO << "Creating archive " << outpath;

		if (fs::exists(outpath))
			zipFile.open(outpath.string().c_str(), true);
		else zipFile.open(outpath.string().c_str(), false);

		if (zipFile.isOpen()) {
			//call recursive function
			if (AddFileToZipArchive(zipFile, p, p.parent_path()) < 0) return -1;
			zipFile.close();
		}
		else {
			LOGTW_ERROR << "Failed to create archive.";
			return -1;
		}
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << ex.what();
		return -1;
	}

	return 0;
}

VOICEBRIDGE_API int UnZip(fs::path p)
{
	try	{
		ziputils::unzipper zipFile;
		zipFile.open(p.string().c_str());
		if (!fs::exists(p)) {
			LOGTW_ERROR << "The file " << p.string() << " does not exist.";
			return -1;
		}
		auto filenames = zipFile.getFilenames();

		for (auto it = filenames.begin(); it != filenames.end(); it++)
		{
			zipFile.openEntry((*it).c_str());
			//zipFile >> std::cout;

			std::string abspath((fs::absolute((*it), p.parent_path())).string());
			fs::path outpath(abspath);
			if (!fs::is_directory(outpath) && fs::exists(outpath))
			{// do not overwrite
				LOGTW_ERROR << "The file " << outpath.string() << " does exist. Please delete it first.";
				return -1;
			}
			//if directory does not exist then create it
			if (!fs::exists(outpath.parent_path())) {
				fs::create_directories(outpath.parent_path());
			}

			fs::ofstream oss(outpath); 
			zipFile >> oss;
		}
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << ex.what();
		return -1;
	}

	return 0;
}
//-----------------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------------
//BOOST Wildcard file actions library API
//-----------------------------------------------------------------------------------------------------------------
/*
dir: the directory in which we are searching
regex_filename: the file name including regex wildcards 
IMPORTANT: must model the whole file name in the regex because boost::regex_match is used! 
examples:
	//NOTE: the \ must be written as \\ because the compiler eats the first \ as escape!
	//NOTE: the first literal dot (.) must be escaped because the dot have a special meaning in regex!
	boost::regex("^(word_boundary\\.).*"))       =>      word_boundary.*	
*/
VOICEBRIDGE_API void GetAllMatchingFiles(std::vector< fs::path > & all_matching_files, fs::path dir, boost::regex regex_filename)
{
	boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
	for (boost::filesystem::directory_iterator i(dir); i != end_itr; ++i)
	{
		// Skip if not a file
		if (!boost::filesystem::is_regular_file(i->status())) continue;

		boost::match_results<std::string::const_iterator> results;
		if( !boost::regex_match( i->path().filename().string(), results, regex_filename ) ) continue;

		// File matches, store it
		all_matching_files.push_back(i->path());
	}
}

VOICEBRIDGE_API int CopyAllMatching(fs::path dir_source, fs::path dir_destination, boost::regex regex_filename)
{
	try	{
		std::vector< fs::path > all_matching_files;
		GetAllMatchingFiles(all_matching_files, dir_source, regex_filename);
		if (all_matching_files.size() > 0) {
			for (fs::path p : all_matching_files)
				fs::copy_file(p, dir_destination / p.filename());
		}
		return 0;
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << ex.what();
		return -1;
	}
}

//NOTE: this is not made API visible because of the danger of deleting several files
int DeleteAllMatching(fs::path dir_source, boost::regex regex_filename)
{
	try {
		std::vector< fs::path > all_matching_files;
		GetAllMatchingFiles(all_matching_files, dir_source, regex_filename);
		if (all_matching_files.size() > 0) {
			for (fs::path p : all_matching_files)
				fs::remove(p);
		}
		return 0;
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << ex.what();
		return -1;
	}
}
//-----------------------------------------------------------------------------------------------------------------

/* checks if the files exist and returns -1 if one of the files do not exist, otherwise returns 0.*/
int CheckFilesExist(std::vector<fs::path> _files, bool report)
{
	for (fs::path p : _files) {
		if (!fs::exists(p))
		{
			if (report)
				LOGTW_ERROR << "Could not find " << p.string();
			return -1;
		}
	}
	return 0;
}


//-----------------------------------------------------------------------------------------------------------------
//General purpose helpers
//-----------------------------------------------------------------------------------------------------------------

/*
	Filter each line of 'in' with the 'filter' and write it to 'out'
	'filter' is a map of regex expressions as keys and the replacement string as value. By the regex selected string
	will be replaced with the 'value' in the map, and this for each key-value pair for each line.
*/
VOICEBRIDGE_API int FilterFile(fs::path in, fs::path out, std::unordered_map<std::string, std::string> filter)
{
	try
	{
		fs::ifstream ifs(in);
		if (!ifs) {
			LOGTW_ERROR << "Could not open input file " << in.string();
			return -1;
		}
		fs::ofstream ofs(out);
		if (!ofs) {
			LOGTW_ERROR << "Could not open output file " << out.string();
			return -1;
		}
		std::string line;
		while (std::getline(ifs, line)) {
			for (auto &pair : filter) {
				boost::regex rexp(pair.first);
				line = boost::regex_replace(line, rexp, pair.second);
			}
			ofs << line << "\n";
		}
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << "Could not filter file. Reason: " << ex.what();
		return -1;
	}
	return 0;
}

/*
	In place convert utf8 string to lower/upper case
*/
std::string ConvertToCaseUtf8(std::string & s, bool toUpper)
{
	auto ss = ConvertToUTF8(s);
	if (toUpper) {
		for (auto& c : ss)
			c = std::toupper(c);
	}
	else {
		for (auto& c : ss)
			c = std::tolower(c);
	}
	s = ss;

	return ss;
}
