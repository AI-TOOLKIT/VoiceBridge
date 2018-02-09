/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : http://www.dreamincode.net/forums/topic/251269-converting-vector-string-to-char-const/
*/

#pragma once

#include <kaldi-win/stdafx.h>

//This class makes the argument list for a function for in the form of: "int argc, char *argv[]"
//NOTE: 1. properties must be passed as "--property-name=property-value" in one 
//		   string (per element in the input vector) and without space between them!
//		2. bool properties may be without the "=" as e.g. "--do-this"
//		3. the first parameter should be the path to the program as e.g. "C://prog.exe"

class StrVec2Arg {
public:
	StrVec2Arg(const std::vector< std::string > &, const std::string progname="");
	StrVec2Arg(const std::string progname = "");
	~StrVec2Arg();

	void Read(const std::vector< std::string > &);
	char *operator[](int);
	char** argv() { return list; }
	int argc() { return size; }
	int size;
private:
	char **list;
	std::string program_name;
};

std::ostream &operator<<(std::ostream &, StrVec2Arg &);
std::ostream &operator<<(std::ostream &, const std::vector< std::string > &);

