/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/

#include "Utility2.h"
#include "Utility.h"

/*
	read in the options from a one line options string
*/
int GetOptionsVector(std::string ino, std::vector<std::string> & _ov)
{
	try {
		if (ino != "") {
			_ov.clear();
			strtk::parse(ino, " ", _ov, strtk::split_options::compress_delimiters);
		}
	}
	catch (const std::exception&) {
		LOGTW_ERROR << " wrong parameter sequence: " << ino;
		return -1;
	}

	return 0;
}

int CheckIfFilesExist(std::vector<fs::path> _f)
{
	//check if necessary files exist
	for (fs::path p : _f) {
		if (!fs::exists(p)) {
			LOGTW_ERROR << "Expected file " << p.string() << " to exist.";
			return -1;
		}
	}
	return 0;
}

int SaveOptionsToFile(fs::path p, std::vector<std::string> _o)
{
	fs::ofstream ofs(p, fs::ofstream::binary | fs::ofstream::out);
	if (!ofs) {
		LOGTW_ERROR << "Failed to save options in " << p.string();
		return -1;
	}
	int i = 0;
	for (std::string s : _o) {
		//make sure that there are no tainling spaces
		boost::algorithm::trim(s);
		if(i>0) ofs << " ";
		ofs << s;
		i++;
	}
	ofs.flush(); ofs.close();
	return 0;
}

/* NOTE: this does not seem to work anymore in Windows 10 (does not provide short paths)
		 but this may also be caused by system settings...
std::string GetWindowsShortPath(fs::path p)
{
	long     length = 0;
	TCHAR*   buffer = NULL;

	//convert std::string to std::wstring
	std::string spath(p.string());
	std::wstring path;
	std::copy(spath.begin(), spath.end(), std::back_inserter(path));

	// First obtain the size needed by passing NULL and 0.
	length = GetShortPathName(path.c_str(), NULL, 0);
	if (length == 0) {
		LOGTW_ERROR << "Error getting the short path.";
		return "";
	}

	// Dynamically allocate the correct size 
	// (terminating null char was included in length)
	buffer = new TCHAR[length];

	// Now simply call again using same long path.
	length = GetShortPathName(path.c_str(), buffer, length);
	if (length == 0) {
		LOGTW_ERROR << "Error getting the short path.";
		return "";
	}
	//convert to std::string
	std::wstring ws(buffer);
	std::string str(ws.begin(), ws.end());
	delete[] buffer;

	return str;
}
*/
