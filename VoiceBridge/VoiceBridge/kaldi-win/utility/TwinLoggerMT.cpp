/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on: Vili Petek's theread safe simple logger (http://www.vilipetek.com)
*/

/*See decription in the header file*/

#include "TwinLoggerMT.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>

#include "Utility.h"

// needed for MSVC
#ifdef WIN32
#define localtime_r(_Time, _Tm) localtime_s(_Tm, _Time)
#endif // localtime_r

namespace twinLogger {

	TwinLoggerMT::TwinLoggerMT() : m_maxSizeMB(10)
	{
	}

	TwinLoggerMT::TwinLoggerMT(std::string filename, bool bAppend, int maxsize) : m_maxSizeMB(maxsize)
	{
		if (m_maxSizeMB < 1) m_maxSizeMB = 1;
		openlog(filename, bAppend);
	}

	void TwinLoggerMT::init(std::string filename, bool bAppend, int maxsize)
	{
		m_maxSizeMB = maxsize;
		if (m_maxSizeMB < 1) m_maxSizeMB = 1;
		openlog(filename, bAppend);
	}

	std::string FindLastLogFile(std::string filename) 
	{
		fs::path f(filename);
		std::string fn = f.stem().string(); //without extension!
		std::string fe = f.extension().string(); //there is a dot!
		std::string fp = f.parent_path().string();

		std::vector<fs::path> logs = GetFilesInDir(fp, fe, false);
		if (logs.size() == 0) return "";
		int n = 0;
		for each(fs::path p in logs)
		{
			std::string nfn = p.stem().string();
			size_t pos = nfn.find_last_of("_");
			if (pos >= std::string::npos-1) continue; //no index
			else {
				if (is_positive_int(nfn.substr(pos + 1))) {
					int nn = std::stoi(nfn.substr(pos + 1));
					if (nn > n) n = nn;
				}
			}
		}

		if (n > 0) 
		{//found indexed log file; return name with max index (last file)
			if (fp != "") fp += "//";
			return (fp + fn + "_" + std::to_string(n) + fe);
		}

		return "";
	}

	void TwinLoggerMT::openlog(std::string filename, bool bAppend)
	{
		if (m_oFile.is_open()) {
			m_oFile.flush();
			m_oFile.close();
		}		

		//in case the log file is becomming too big then start a new one
		if (bAppend && fs::exists(filename)) 
		{
			//find the last written log file; can be different in case the log file is split
			std::string lastlog = FindLastLogFile(filename);
			if (lastlog != "") filename = lastlog;

			uintmax_t szMB = fs::file_size(filename) /1024 /1024;
			if (szMB >= m_maxSizeMB)
			{
				fs::path f(filename);
				std::string fn = f.stem().string(); //without extension!
				std::string fe = f.extension().string(); //there is a dot!
				std::string fp = f.parent_path().string();
				if (fp != "") fp += "//";
				size_t pos=fn.find_last_of("_");
				if (pos == std::string::npos)
				{
					filename = fp + fn + "_1" + fe;
				}
				else if (pos == std::string::npos - 1)
				{
					filename = fp + fn + "1" + fe;
				}
				else {
					if (is_positive_int(fn.substr(pos + 1))) {
						int n = std::stoi(fn.substr(pos + 1)) + 1;
						filename = fp  + fn.substr(0, pos) + "_" + std::to_string(n) + fe;
					}
					else {
						filename = fp  + fn + "_1" + fe;
					}
				}
				//iterate till we find a not existing log number
				pos = filename.find_last_of("_");
				int maxiter = 10000, i = 0;
				int n = 0;
				while (fs::exists(filename))
				{
					n = std::stoi(filename.substr(pos + 1)) + 1;
					filename = fp + fn.substr(0, pos) + "_" + std::to_string(n) + fe;
					if (i == maxiter) break;
				}
				//in case we are over the maxiter then append a new index; 
				//should never happen but we can write forever untill there is HD space
				if (i == maxiter) {
					filename = fp + fn.substr(0, pos) + "_" + std::to_string(n) + "_1" + fe;
				}
			}
		}

		try	{
			if (bAppend)
				m_oFile.open(filename, std::fstream::out | std::fstream::app | std::fstream::ate);
			else
				m_oFile.open(filename, std::fstream::out);
		}
		catch (const std::exception&)
		{
			std::cerr << "ERROR: could not open log file " << filename << "\n";
			//NOTE: the logger will only write to std::cout
		}
	}

	TwinLoggerMT::~TwinLoggerMT()
	{
		if (m_oFile.is_open()) {
			m_oFile.flush();
			m_oFile.close();
		}
	}

	LogStream TwinLoggerMT::operator()()
	{
		return LogStream(*this, Info, "");
	}

	LogStream TwinLoggerMT::operator()(Level nLevel)
	{
		return LogStream(*this, nLevel, "");
	}

	LogStream TwinLoggerMT::operator()(Level nLevel, std::string filename, int linenumber)
	{
		std::string extrainfo("File: " + filename + ", Line: " + std::to_string(linenumber));

		return LogStream(*this, nLevel, extrainfo);
	}

	const tm* TwinLoggerMT::getLocalTime()
	{
		auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		localtime_r(&in_time_t, &m_oLocalTime);
		return &m_oLocalTime;
	}

	// Convert date and time info from tm to a character string in format "YYYY-mm-DD HH:MM:SS" and send it to a stream
	std::ostream& operator<< (std::ostream& stream, const tm* tm)
	{
		return stream << std::put_time(tm, "%Y-%m-%d %H:%M:%S");

		//in case std::put_time would not work for your compiler:
		//return stream << 1900 + tm->tm_year << '-' <<
		//	std::setfill('0') << std::setw(2) << tm->tm_mon + 1 << '-'
		//	<< std::setfill('0') << std::setw(2) << tm->tm_mday << ' '
		//	<< std::setfill('0') << std::setw(2) << tm->tm_hour << ':'
		//	<< std::setfill('0') << std::setw(2) << tm->tm_min << ':'
		//	<< std::setfill('0') << std::setw(2) << tm->tm_sec;
	}

	//NOTE: logging is more detailed in the log file than on std::cout
	//		If the log file is not open it will write only to std::cout
	void TwinLoggerMT::log(Level nLevel, std::string oMessage, std::string extraInfo)
	{
		const static char* LevelStr[] = { "DEBUG", "INFO", "WARNING", "ERROR", "FATAL ERROR" };

		m_oMutex.lock();

		//check if the message only contains new line characters and if yes then only output the new lines and no headings
		std::string s(oMessage);
		ReplaceStringInPlace(s, "\n", "");
		boost::algorithm::trim(s);
		if (s == "") {
			//write log to file
			if (m_oFile.is_open())
				m_oFile << oMessage;
			//write log to std::cout
			std::cout << oMessage;
		}
		else {
			//write log to file
			if (m_oFile.is_open()) {
				m_oFile << '[' << getLocalTime() << ']' << '[' << LevelStr[nLevel] << "]\t" << oMessage << std::endl;
				if (extraInfo != "") m_oFile << extraInfo << std::endl;
			}
			//write log to std::cout
			std::cout << '[' << LevelStr[nLevel] << "]\t" << oMessage << std::endl;
			if (extraInfo != "") std::cout << extraInfo << std::endl;
		}
		m_oMutex.unlock();
	}

	LogStream::LogStream(TwinLoggerMT& oLogger, Level nLevel, std::string extraInfo) :
		m_oLogger(oLogger), m_nLevel(nLevel), m_extraInfo(extraInfo)
	{
	}
	LogStream::LogStream(const LogStream& ls) :
		m_oLogger(ls.m_oLogger), m_nLevel(ls.m_nLevel)
	{
	}
	LogStream::~LogStream()
	{
		m_oLogger.log(m_nLevel, str(), m_extraInfo);
	}
}
