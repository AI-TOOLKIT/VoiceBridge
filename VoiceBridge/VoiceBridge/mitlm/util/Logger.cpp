/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
Based on: see below
*/
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2008, Massachusetts Institute of Technology              //
// All rights reserved.                                                   //
//                                                                        //
// Redistribution and use in source and binary forms, with or without     //
// modification, are permitted provided that the following conditions are //
// met:                                                                   //
//                                                                        //
//     * Redistributions of source code must retain the above copyright   //
//       notice, this list of conditions and the following disclaimer.    //
//                                                                        //
//     * Redistributions in binary form must reproduce the above          //
//       copyright notice, this list of conditions and the following      //
//       disclaimer in the documentation and/or other materials provided  //
//       with the distribution.                                           //
//                                                                        //
//     * Neither the name of the Massachusetts Institute of Technology    //
//       nor the names of its contributors may be used to endorse or      //
//       promote products derived from this software without specific     //
//       prior written permission.                                        //
//                                                                        //
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS    //
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT      //
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  //
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT   //
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,  //
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT       //
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  //
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY  //
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT    //
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE  //
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   //
////////////////////////////////////////////////////////////////////////////

#include <cstdarg>
#include <cstdio>
#include "Logger.h"

#include "mitlm/mitlm.h"

namespace mitlm {
	//@+zso
	template<typename ... Args>
	std::string string_format(const std::string& format, Args ... args)
	{
		size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
		std::unique_ptr<char[]> buf(new char[size]);
		snprintf(buf.get(), size, format.c_str(), args ...);
		return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
	}
////////////////////////////////////////////////////////////////////////////////

#ifdef NDEBUG
	int  Logger::_verbosity = 0;
	bool Logger::_timestamp = false;
#else
	int  Logger::_verbosity = 1;
	bool Logger::_timestamp = true;
#endif
	clock_t Logger::_startTime = clock();
	FILE*   Logger::_err_file = stderr;

	void Logger::Log(int level, const char *fmt, ...) {
		if (_verbosity >= level) {
			/*@-zso
			if (_err_file != NULL) {
				va_list args;
				va_start(args, fmt);
				if (_timestamp)
					fprintf(_err_file, "%.3f\t", (double)(clock() - _startTime) / CLOCKS_PER_SEC);
				vfprintf(_err_file, fmt, args);
				va_end(args);
				fflush(_err_file);
			}
			*/
			//@+zso
			va_list args;
			va_start(args, fmt);
			if (_timestamp)
				LOGTW_INFO << string_format("%.3f\t", (double)(clock() - _startTime) / CLOCKS_PER_SEC);
			char buffer[1024];
			vsprintf(buffer, fmt, args);
			LOGTW_INFO << buffer;
			va_end(args);
		}
	}

	void Logger::Warn(int level, const char *fmt, ...) {
		if (_verbosity >= level) {
			va_list args;
			/*@-zso
			if (_err_file != NULL) {
				va_start(args, fmt);
				fprintf(_err_file, "\033[0;33m");
				if (_timestamp)
					fprintf(_err_file, "%.3f\t", (double)(clock() - _startTime) / CLOCKS_PER_SEC);
				vfprintf(_err_file, fmt, args);
				fprintf(_err_file, "\033[m");
				va_end(args);
				fflush(_err_file);
			}
			*/
			//@+zso
			va_start(args, fmt);
			if (_timestamp)
				LOGTW_WARNING << string_format("%.3f\t", (double)(clock() - _startTime) / CLOCKS_PER_SEC);
			char buffer[1024];
			vsprintf(buffer, fmt, args);
			LOGTW_WARNING << buffer;
			va_end(args);
		}
	}

	void Logger::Error(int level, const char *fmt, ...) {
		if (_verbosity >= level) {
			va_list args;
			/*@-zso
			if (_err_file != NULL) {
				va_start(args, fmt);
				fprintf(_err_file, "\033[1;31m");
				if (_timestamp)
					fprintf(_err_file, "%.3f\t", (double)(clock() - _startTime) / CLOCKS_PER_SEC);
				vfprintf(_err_file, fmt, args);
				fprintf(_err_file, "\033[m");
				va_end(args);
				fflush(_err_file);
			}
			*/
			//@+zso
			va_start(args, fmt);
			if (_timestamp)
				LOGTW_ERROR << string_format("%.3f\t", (double)(clock() - _startTime) / CLOCKS_PER_SEC);
			char buffer[1024];
			vsprintf(buffer, fmt, args);
			LOGTW_ERROR << buffer;
			va_end(args);
		}
	}

}
