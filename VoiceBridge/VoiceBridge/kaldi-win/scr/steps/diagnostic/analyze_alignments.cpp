/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright Johns Hopkins University (Author: Daniel Povey) 2016.  Apache 2.0.
*/

/*
 This function performs some analysis of alignments on disk, currently in terms of phone lengths, 
 including lenghts of leading and trailing silences.
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"
#include <kaldi-win/utility/strvec2arg.h>

//the return values from each thread/job
static std::vector<int> _ret;
static void LaunchJob(int argc1, char *argv1[], fs::path out, fs::path log);

int AnalyzeAlignments(
	fs::path lang,			//lang directory
	fs::path dir			//training directory
)
{
	fs::path model(dir / "final.mdl");
	std::vector<fs::path> _f = { lang / "words.txt", model, dir / "ali.1", dir / "num_jobs" };
	for each(fs::path f in _f) {
		if (!fs::exists(f)) {
			LOGTW_ERROR <<  "Expecting file " << f.string() << " to exist.";
			return -1;
		}
	}
	if (CreateDir(dir / "log", false) < 0) return -1;

	//NOTE: in case the training was done on a different computer with different available number of hardware threads
	//		the code will run but may not have optimal speed. If there are less threads available then the code will
	//		run slower than normal.
	//get the number of jobs (threads) used to make the files in the training directory.
	std::string snj("");
	try	{
		snj = GetFirstLineFromFile((dir / "num_jobs").string());
	} catch (const std::exception&)	{}
	int nj = StringToNumber<int>(snj, -1);
	if (nj < 0) {
		LOGTW_ERROR << "Could not read number of jobs from file " << (dir / "num_jobs").string() << ".";
		return -1;
	}

	{
		std::vector<std::thread> _threads;
		std::vector<StrVec2Arg *> _args1;
		std::vector<string_vec> _options;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			string_vec options;
			options.push_back("--print-args=false");
			options.push_back("--write-lengths=true");
			options.push_back(model.string());
			options.push_back("ark,t:" + (dir / "ali.JOBID").string());
			options.push_back("ark,t:" + (dir / "ali.JOBID.temp").string()); //output
			//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
			for (std::string &s : options)
			{//NOTE: accesing by ref for in place editing
				ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
			}
			_options.push_back(options);
			StrVec2Arg *args1 = new StrVec2Arg(_options[JOBID-1]);
			_args1.push_back(args1);

			//------->
			//logfile
			fs::path log(dir / "log" / ("get_phone_alignments." + std::to_string(JOBID) + ".log"));
			//
			_threads.emplace_back(
				LaunchJob,
				_args1[JOBID - 1]->argc(), _args1[JOBID - 1]->argv(),
				(dir / ("ali."+ std::to_string(JOBID)+".temp")),	//send also the output of AliToPhones that it does not need to be searched for 
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}
		//clean up
		try {
			for (int JOBID = 1; JOBID <= nj; JOBID++) {
				delete _args1[JOBID - 1];
				//delete temp files:
				if (fs::exists(dir / ("ali." + std::to_string(JOBID) + ".temp")))
					fs::remove(dir / ("ali." + std::to_string(JOBID) + ".temp"));
			}
			_args1.clear();
		}
		catch (const std::exception& ex)
		{
			LOGTW_WARNING << "Could not free up memory and/or delete temporary files. Reason: " << ex.what() << ".";
		}
		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret[JOBID - 1] < 0)
				return -1;
		}
	}

	//analyze alignments
	std::vector<fs::path> _phonestats;
	for (int JOBID = 1; JOBID <= nj; JOBID++) {
		if (fs::exists(dir / ("phone_stats." + std::to_string(JOBID))))
			_phonestats.push_back(dir / ("phone_stats." + std::to_string(JOBID)));
	}

	LOGTW_INFO << "Running some diagnostics...";

	if (AnalyzePhoneLengthStats(_phonestats, lang) < 0) {
		LOGTW_WARNING << "Phone length stats diagnostics failed.";
	}

	//LOGTW_INFO << "See diagnostics results in file " << (dir / "log" / "analyze_alignments.log").string() << ".";

	//cleanup
	try {
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (fs::exists(dir / ("phone_stats." + std::to_string(JOBID))))
				fs::remove(dir / ("phone_stats." + std::to_string(JOBID)));
		}
	}
	catch (const std::exception&) {}

	return 0;
}


//NOTE: this will be called from several threads
static void LaunchJob(int argc1, char *argv1[], fs::path out, fs::path log)
{
	//we redirect Kaldi logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	int ret1 = 0;
	
	try	{
		ret1 = AliToPhones(argc1, argv1, file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (AliToPhones). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	//
	StringTable table;
	if (ReadStringTable(out.string(), table, " ;") < 0) {
		_ret.push_back(-1);
		return;
	}
	//format: ID 1 88 ; 3 51 ; 3 61 ; 3 59 ; 3 72 ; 2 60 ; 2 66 ; 2 68 ; 2 108 
	std::vector<std::string> _v; 
	for (StringTable::const_iterator it(table.begin()), it_end(table.end()); it != it_end; ++it)
	{		
		int NF = (*it).size();
		if (NF < 5)
		{ //expect minimum 1 ID column and 2 value pairs!
			_ret.push_back(-1);
			return;
		}
		_v.push_back("begin " + (*it)[1] + " " + (*it)[2]);
		_v.push_back("end " + (*it)[NF - 2] + " " + (*it)[NF-1]);		
		for (int c = 0; c < NF-1; c++) {
			if ( (c+1) % 2 == 0) {
				_v.push_back("all " + (*it)[c] + " " + (*it)[c+1]);
			}
		}
	}

	//count the number of times each line appears before deleting repeated lines
	unordered_map<std::string, size_t> count;  // holds count of each line
	for (int i = 0; i<_v.size(); i++)
		count[_v[i]]++;

	std::sort(_v.begin(), _v.end());
	_v.erase(std::unique(_v.begin(), _v.end()), _v.end());

	for (int i = 0; i<_v.size(); i++) {
		_v[i].insert(0, (std::to_string(count[_v[i]]) + " "));
	}

	//save the phone stats in 'phone_stats.JOBID' files
	std::string filename(out.filename().string());
	ReplaceStringInPlace(filename, "ali", "phone_stats");
	ReplaceStringInPlace(filename, ".temp", "");
	fs::path phonestats(out.parent_path() / filename);

	fs::ofstream file_(phonestats, std::ios::binary | std::ios::out);
	if (!file_) {
		LOGTW_ERROR << " can't open output file: " << phonestats;
		_ret.push_back(-1);
		return;
	}
	for (int i = 0; i < _v.size(); i++)
		file_ << _v[i] << "\n";
	file_.flush(); file_.close();

	_ret.push_back(ret1);
}
