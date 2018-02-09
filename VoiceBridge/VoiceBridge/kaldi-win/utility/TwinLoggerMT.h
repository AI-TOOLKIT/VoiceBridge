#pragma once
/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018: 
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.

	Based on: Vili Petek's theread safe simple logger (http://www.vilipetek.com)
*/

/*
	Description:
		Thread safe logger which writes to a log file and/or to std::cout. Can write to the same log file from multiple
		threads.
	External Dependencies:
		#include <boost/filesystem.hpp>		
		//- needs extra statically linked boost cpp filesystem files or boost filesystem lib!
		//- needs boost\system\src\error_code.cpp or boost system lib.

	Usage for application level global logging:
		CPP file:
			//define global variable:
			twinLogger::TwinLoggerMT oLog;

			//initialize at startup:
			oLog.init("test.log", false);

			//log messages
			LOG_WARNING << "Message"; //NOTE: no new line is needed, added automatically

		H file:
			//define global variable:
			extern twinLogger::TwinLoggerMT oLog;
			#define LOG_DEBUG oLog(twinLogger::Debug)
			#define LOG_INFO oLog(twinLogger::Info)
			#define LOG_WARNING oLog(twinLogger::Warning)
			#define LOG_ERROR oLog(twinLogger::Error)
			#define LOG_FATALERROR oLog(twinLogger::FatalError)
*/
#include "stdafx.h"

#include <string>
#include <sstream>
#include <mutex>
#include <memory>
#include <fstream>
#include <unordered_map>

namespace twinLogger {

	// log message levels
	enum Level { Debug, Info, Warning, Error, FatalError };
	class TwinLoggerMT;

#pragma warning( push )
#pragma warning( disable : 4251)
	class VOICEBRIDGE_API LogStream : public std::ostringstream
	{
	public:
		LogStream(TwinLoggerMT& oLogger, Level nLevel, std::string extraInfo);
		LogStream(const LogStream& ls);
		~LogStream();

	private:
		TwinLoggerMT& m_oLogger;
		Level m_nLevel;
		std::string m_extraInfo;
	};

	class VOICEBRIDGE_API TwinLoggerMT
	{
	public:
		TwinLoggerMT();
		TwinLoggerMT(std::string filename, bool bAppend = true, int maxsize=10);
		virtual ~TwinLoggerMT();
		//set the log file name, whether to append to existing log file or overwrite, and the max size of the log file in MB (will be split)
		void init(std::string filename, bool bAppend = true, int maxsize=10);
		void log(Level nLevel, std::string oMessage, std::string extraInfo);
		//operator to make logging easier
		LogStream operator()();
		LogStream operator()(Level nLevel);
		LogStream operator()(Level nLevel, std::string filename, int linenumber);

	private:
		const tm* getLocalTime();
		void openlog(std::string filename, bool bAppend);

	private:
		std::mutex		m_oMutex;
		std::ofstream	m_oFile;
		tm				m_oLocalTime;
		int				m_maxSizeMB;
	};
#pragma warning( pop ) 
}
