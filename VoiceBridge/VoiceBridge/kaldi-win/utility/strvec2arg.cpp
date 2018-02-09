/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : http://www.dreamincode.net/forums/topic/251269-converting-vector-string-to-char-const/
*/

#include <kaldi-win/utility/strvec2arg.h>

#include "kaldi-win/utility/Utility.h"

StrVec2Arg::StrVec2Arg(const std::string progname) 
	: size(0), list(NULL), program_name(progname)
{
	if (program_name == "") program_name = "KaldiWin";
}

StrVec2Arg::StrVec2Arg(const std::vector< std::string > &v, const std::string progname)
						: size(v.size()+1), program_name(progname) //NOTE: +1 for program name
{
	list = new char *[size];
	if (program_name == "") program_name = "KaldiWin";
	list[0] = const_cast< char* >(program_name.c_str());
	for (int i = 1; i<size; ++i) {
		list[i] = const_cast< char* >(v[i-1].c_str());
	}
}

StrVec2Arg::~StrVec2Arg() {
	try	{
		if (list != NULL) {
			delete[] list;
			list = NULL;
		}
	}
	catch (const std::exception&)
	{
		LOGTW_ERROR << " internal error (delete StrVec2Arg).";
		std::getchar();
	}
}

void StrVec2Arg::Read(const std::vector< std::string > &v) {
	size = v.size() + 1; //NOTE: +1 for program name
	list = new char *[size];
	list[0] = const_cast< char* >(program_name.c_str());
	for (int i = 1; i<size; ++i) {
		list[i] = const_cast< char* >(v[i - 1].c_str());
	}
}

char *StrVec2Arg::operator[](int i) { return list[i]; }

std::ostream &operator<<(std::ostream &out, StrVec2Arg &list) {
	for (int i = 0; i<list.size; i++) {
		if (i != 0) { out << ','; }
		out << '"' << list[i] << '"';
	}
	return out;
}

std::ostream &operator<<(std::ostream &out, const std::vector< std::string > &v) {
	for (int i = 0; i<v.size(); i++) {
		if (i != 0) { out << ','; }
		out << '"' << v[i] << '"';
	}
	return out;
}

