/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright Johns Hopkins University (Author: Daniel Povey) 2016.  Apache 2.0.
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"

//the return values from each thread/job
static std::vector<int> _ret;

int WaitForThreadsReady(int num_jobs, std::vector<std::thread> & _threads);
static void LaunchJob(int JOBID, fs::path dir, fs::path model, float acwt);
static void LaunchJobLatticeDepth(int JOBID, fs::path dir, fs::path model);

using MAPUNORDSI = std::unordered_map<std::string, int>;

/*
	This function writes some diagnostics to <decode-dir>/log/alignments.log.
	This function does the same type of diagnostics as AnalyzeAlignments(), except it starts from lattices 
	(so it has to convert the lattices to alignments first).
*/
int AnalyzeLats (
	fs::path lang,			//lang directory <lang-dir>|<graph-dir>
	fs::path dir,			//training directory <decode-dir>
	std::string iter,		//e.g. 'final' (default: final)
	float acwt				//Acoustic scale for getting best-path (default: 0.1)
)
{
	if (iter == "") iter = "final";

	fs::path model = dir.parent_path() / (iter + ".mdl");

	std::vector<fs::path> required = { lang / "words.txt", model, dir / "lat.1", dir / "num_jobs" };
	for (fs::path p : required) {
		if (!fs::exists(p)) {
			LOGTW_ERROR << "Failed to find " << p.string();
			return -1;
		}
	}

	std::string snj("");
	try {
		snj = GetFirstLineFromFile((dir / "num_jobs").string());
	}
	catch (const std::exception&) {}
	int num_jobs = StringToNumber<int>(snj, -1);
	if (num_jobs < 0) {
		LOGTW_ERROR << "Could not read number of jobs from file " << (dir / "num_jobs").string() << ".";
		return -1;
	}

	if (CreateDir(dir / "log", true) < 0) {
		LOGTW_ERROR << "Failed to create " << (dir / "log").string();
		return -1;
	}

	//NOTE "^(phone_stats\\.).*"   =>   any file starting with 'phone_stats.' ( = phone_stats.* )
	if (DeleteAllMatching(dir, boost::regex("^(phone_stats\\.).*")) < 0) return -1;

	//this writes two archives of depth_tmp and ali_tmp of (depth per frame, alignment per frame).
	//and then makes phone_stats
	std::vector<std::thread> _threadsLatticeBestPath;
	_ret.clear();
	for (int JOBID = 1; JOBID <= num_jobs; JOBID++) {
		_threadsLatticeBestPath.emplace_back(
			LaunchJob,
			JOBID, dir, model, acwt);
	}
	//wait for all threads till they are ready and check success
	if (WaitForThreadsReady(num_jobs, _threadsLatticeBestPath) < 0) return -1;

	//AnalyzePhoneLengthStats from all jobs
	std::vector<fs::path> _phonestats;
	for (int JOBID = 1; JOBID <= num_jobs; JOBID++) {
		if (fs::exists(dir / ("phone_stats." + std::to_string(JOBID))))
			_phonestats.push_back(dir / ("phone_stats." + std::to_string(JOBID)));
	}
	if (AnalyzePhoneLengthStats(_phonestats, lang) < 0) {
		LOGTW_WARNING << "Phone length stats diagnostics failed.";
	}

	//Analyze Lattice Depths
	std::vector<std::thread> _threadsLD;
	_ret.clear();
	for (int JOBID = 1; JOBID <= num_jobs; JOBID++) {
		_threadsLD.emplace_back(
			LaunchJobLatticeDepth,
			JOBID, dir, model);
	}
	//wait for all threads till they are ready and check success
	if (WaitForThreadsReady(num_jobs, _threadsLD) < 0) return -1;

	//Analyze all (dir/depth_stats_tmp.*) stat files
	std::vector<fs::path> _latticestats;
	for (int JOBID = 1; JOBID <= num_jobs; JOBID++)
		_latticestats.push_back(dir / ("depth_stats_tmp." + std::to_string(JOBID)));
	if (AnalyzeLatticeDepthStats(_latticestats, lang) < 0) return -1;

	//remove temporary files
	//NOTE "^(ali_trn\\.).*"   =>   any file starting with 'ali_trn.' ( = ali_trn.* )
	if (DeleteAllMatching(dir, boost::regex("^(phone_stats\\.).*")) < 0) return -1;
	if (DeleteAllMatching(dir, boost::regex("^(depth_tmp\\.).*")) < 0) return -1; //depth_tmp.1, depth_tmp.1.temp, ...
	if (DeleteAllMatching(dir, boost::regex("^(depth_stats_tmp\\.).*")) < 0) return -1;
	if (DeleteAllMatching(dir, boost::regex("^(ali_tmp\\.).*")) < 0) return -1;
	if (DeleteAllMatching(dir, boost::regex("^(ali_trn\\.).*")) < 0) return -1;

	return 0;
}


static void LaunchJob(int JOBID, fs::path dir, fs::path model, float acwt)
{
	std::string SJOBID(std::to_string(JOBID));
	fs::ofstream file_log((dir / "log" / ("lattice_best_path." + SJOBID + ".log")), fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / ("lattice_best_path." + SJOBID + ".log")).string() << ".";

	//lattice-depth-per-frame  ------------------------
	string_vec options;
	options.push_back("--print-args=false");
	options.push_back("ark:" + (dir / ("lat." + SJOBID)).string());
	options.push_back("ark,t:" + (dir / ("depth_tmp." + SJOBID)).string());
	options.push_back("ark,t:" + (dir / ("depth_tmp." + SJOBID + ".temp")).string()); //output
	StrVec2Arg args1(options);
	int ret = LatticeDepthPerFrame(args1.argc(), args1.argv(), file_log);
	if (ret < 0) { //do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//lattice-best-path  -------------------------------
	options.clear();
	options.push_back("--print-args=false");
	options.push_back("--acoustic-scale=" + std::to_string(acwt));
	options.push_back("ark,t:" + (dir / ("depth_tmp." + SJOBID + ".temp")).string()); //input from lattice-depth-per-frame
	options.push_back("ark,t:" + (dir / ("ali_trn." + SJOBID)).string() + ".temp"); //this will not be used
	options.push_back("ark,t:" + (dir / ("ali_tmp." + SJOBID)).string()); //output
	StrVec2Arg args2(options);
	ret = LatticeBestPath(args2.argc(), args2.argv(), file_log);
	if (ret < 0) { //do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//ali-to-phones -------------------------------------
	file_log.flush(); file_log.close();
	file_log.open((dir / "log" / ("get_lattice_stats." + SJOBID + ".log")).string(), fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / ("get_lattice_stats." + SJOBID + ".log")).string() << ".";

	options.clear();
	options.push_back("--print-args=false");
	options.push_back("--write-lengths=true");
	options.push_back(model.string());
	options.push_back("ark,t:" + (dir / "ali_tmp.JOBID").string());
	options.push_back("ark,t:" + (dir / "ali_tmp.JOBID.temp").string()); //output
	//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
	for (std::string &s : options)
	{//NOTE: accesing by ref for in place editing
		ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	}
	StrVec2Arg args3(options);
	try {
		ret = AliToPhones(args3.argc(), args3.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (AliToPhones). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	//make dir/phone_stats.JOBID from (dir / "ali_tmp.JOBID.temp")
	StringTable table;
	if (ReadStringTable((dir / ("ali_tmp."+SJOBID+".temp")).string(), table, " ;") < 0) {
		_ret.push_back(-1);
		return;
	}
	//ali_tmp.JOBID.temp format: ID 1 88 ; 3 51 ; 3 61 ; 3 59 ; 3 72 ; 2 60 ; 2 66 ; 2 68 ; 2 108 
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
		_v.push_back("end " + (*it)[NF - 2] + " " + (*it)[NF - 1]);
		for (int c = 0; c < NF - 1; c++) {
			if ((c + 1) % 2 == 0) {
				_v.push_back("all " + (*it)[c] + " " + (*it)[c + 1]);
			}
		}
	}
	//count the number of times each line appears before deleting repeated lines
	unordered_map<std::string, size_t> count;  // holds count of each line
	for (int i = 0; i<_v.size(); i++)
		count[_v[i]]++;
	//sort
	std::sort(_v.begin(), _v.end());
	//make unique
	_v.erase(std::unique(_v.begin(), _v.end()), _v.end());
	for (int i = 0; i<_v.size(); i++) {
		_v[i].insert(0, (std::to_string(count[_v[i]]) + " "));
	}
	//save the phone stats
	std::string phonestats((dir / ("phone_stats." + SJOBID)).string());
	fs::ofstream file_(phonestats, std::ios::binary | std::ios::out);
	if (!file_) {
		LOGTW_ERROR << "Can't open output file: " << phonestats;
		_ret.push_back(-1);
		return;
	}
	for (int i = 0; i < _v.size(); i++)
		file_ << _v[i] << "\n";
	file_.flush(); file_.close();

	_ret.push_back(ret);
}

static void LaunchJobLatticeDepth(int JOBID, fs::path dir, fs::path model)
{
	std::string SJOBID(std::to_string(JOBID));
	fs::ofstream file_log(dir / "log" / ("lattice_best_path." + SJOBID + ".log"), fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / ("lattice_best_path." + SJOBID + ".log")).string() << ".";

	string_vec options;
	options.push_back("--print-args=false");
	options.push_back("--per-frame=true");
	options.push_back(model.string());
	options.push_back("ark,t:" + (dir / ("ali_tmp." + SJOBID)).string());
	options.push_back("ark,t:" + (dir / ("ali_tmp." + SJOBID + ".temp2")).string()); //output (ali_tmp.JOBID.temp2)
	StrVec2Arg args(options);
	int ret = -1;
	try {
		ret = AliToPhones(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (AliToPhones). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) { //do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//merge dir/ali_tmp.JOBID and dir/depth_tmp.JOBID horizontally
	//paste together the phone-indexes and the depths so that one line will be like 
	// utt-id1 phone1 phone2 phone3 ..utt-id1 depth1 depth2 depth3 ...
	//ali_tmp.JOBID.temp format: ID 1 88 ; 3 51 ; 3 61 ; 3 59 ; 3 72 ; 2 60 ; 2 66 ; 2 68 ; 2 108 
	StringTable t_ali, t_depth, t_comb;
	if (ReadStringTable((dir / ("ali_tmp." + SJOBID + ".temp2")).string(), t_ali, " ;") < 0) {
		_ret.push_back(-1);
		return;
	}
	if (ReadStringTable((dir / ("depth_tmp." + SJOBID)).string(), t_depth, " ;") < 0) {
		_ret.push_back(-1);
		return;
	}
	if (t_ali.size() != t_depth.size()) {
		LOGTW_ERROR << "The number of records in ali_tmp does not correspond to depth_tmp.";
		_ret.push_back(-1);
		return;
	}
	for (int r = 0; r < t_ali.size(); r++){
		string_vec _v;
		for (int c = 0; c < t_ali[r].size(); c++)
			_v.push_back(t_ali[r][c]);
		for (int c = 0; c < t_depth[r].size(); c++)
			_v.push_back(t_depth[r][c]);
		t_comb.emplace_back(_v);
	}
	//Compute counts of pairs (phone, lattice-depth) and output lines containing 3 integers representing :
	//phone lattice_depth, count[phone, lattice_depth]
	MAPUNORDSI count;
	for (int r = 0; r < t_comb.size(); r++) {
		int half = t_comb[r].size() / 2;
		for (int c = 1; c < half; c++) {
			int m = (c + 1) + half; //NOTE: c is zero based and therefore +1
			std::string s(std::to_string(c+1) + " " + std::to_string(m));
			MAPUNORDSI::const_iterator ic = count.find(s);
			if (ic == count.end())
			{//not found
				count.emplace(s, 0);
			}
			count[s]++;
		}
	}
	fs::ofstream fdepthstats(dir / ("depth_stats_tmp." + SJOBID), std::ios::binary | std::ios::out);
	if (!fdepthstats) {
		LOGTW_ERROR << "Can't open output file: " << (dir / ("depth_stats_tmp." + SJOBID)).string() << ".";
		_ret.push_back(-1);
		return;
	}
	for (const auto& pair : count) {
		fdepthstats 
			<< pair.first << " " //phone lattice_depth
			<< pair.second		 //count[phone, lattice_depth]
			<< "\n";
	}
	fdepthstats.flush(); fdepthstats.close();

	_ret.push_back(ret);
}


int WaitForThreadsReady(int num_jobs, std::vector<std::thread> & _threads)
{
	for (auto& t : _threads) {
		t.join();
	}
	//check return values from the threads/jobs
	for (int JOBID = 1; JOBID <= num_jobs; JOBID++) {
		if (_ret[JOBID - 1] < 0)
			return -1;
	}
	return 0;
}
