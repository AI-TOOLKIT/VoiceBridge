/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal), Apache 2.0.
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"
#include <kaldi-win/utility/strvec2arg.h>

static void LaunchJobComputeWer(
	int JOBID,
	bool decode_mbr,
	int minlmwt,
	int maxlmwt,
	std::string	wip,
	double beam,
	fs::path symtab,
	UMAPSS wer_hyp_filter,
	fs::path dir
);

//the return values from each thread/job
static std::vector<int> _ret;
static std::mutex m_oMutex; //NOTE: needed for locking threads accessing the merged lat.* file! An other option would be to make several copies of the file.

/*
	Computes the WER (Word Error Rate)
	NOTE: when run both WER and CER then call WER first and then CER with stage=2, in this way the
	the lattice decoding won't be run twice.
*/
int ScoreKaldiWER(
	fs::path data,											//data directory <data-dir>
	fs::path lang_or_graph,									//language or graph directory <lang-dir|graph-dir>
	fs::path dir,											//decode directory <decode-dir>
	UMAPSS & wer_ref_filter,								//for filtering the text; contains a 'key' which is a regex and the result will be replaced in the input text with the 'value' property in the map
	UMAPSS & wer_hyp_filter,								//for filtering the text; contains a 'key' which is a regex and the result will be replaced in the input text with the 'value' property in the map
	int nj,													//the number of hardware threads available
	int stage, //= 0										//start scoring script from part-way through.
	bool decode_mbr, //= false								//maximum bayes risk decoding (confusion network).
	bool stats, // = true									//output statistics
	double beam, // = 6										//Pruning beam [applied after acoustic scaling]
	std::string word_ins_penalty, // = 0.0,0.5,1.0			//word insertion penalty
	int min_lmwt, // = 7									//minumum LM-weight for lattice rescoring
	int max_lmwt, // = 17									//maximum LM-weight for lattice rescoring
	std::string iter // = final								//which model to use; default final.mdl
)
{
	fs::path symtab = lang_or_graph / "words.txt";

	if (CheckFilesExist(std::vector<fs::path> {symtab, dir / "lat.1", data / "text"}) < 0) return -1;

	if (decode_mbr)
		LOGTW_INFO << "Scoring with MBR, word insertion penalty = " << word_ins_penalty;
	else
		LOGTW_INFO << "Scoring with word insertion penalty = " << word_ins_penalty;
		
	if (CreateDir(dir / "scoring_kaldi") < 0) return -1;

	//filter the input text for testing
	fs::path text(data / "text"), text_filt(dir / "scoring_kaldi" / "test_filt.txt");

	//NOTE: the filter can be empty and then the file is copied as it is
	if (FilterFile(text, text_filt, wer_ref_filter) < 0) return -1;

	std::vector<std::string> _wip;
	boost::algorithm::trim(word_ins_penalty);
	strtk::parse(word_ins_penalty, ",", _wip, strtk::split_options::compress_delimiters);

	if (stage <= 0) 
	{
		//Merge all dir/lat.* files into 1 file for lattice-scale then simply go through all functions
		std::vector<fs::path> _f;
		fs::path lat_merged(dir / "lat.merged");
		GetAllMatchingFiles(_f, dir, boost::regex("^(lat\\.).*")); //lat.*
		if (_f.size() < 1) {
			LOGTW_ERROR << "Could not find lat.* in " << dir.string();
			return -1;
		}
		if (MergeFiles(_f, lat_merged) < 0) {
			LOGTW_ERROR << "Could not merge lat.* in " << dir.string();
			return -1;
		}

		for (std::string wip : _wip) 
		{
			if (CreateDir(dir / "scoring_kaldi" / ("penalty_" + wip) / "log") < 0) return -1;

			//Start parallel processing of LMWT's
			std::vector<std::thread> _threads;
			_ret.clear();
			int LMWT = min_lmwt;
			int nLMWT = max_lmwt - min_lmwt + 1;
			int dLMWT = nLMWT / nj;
			int minlmwt = min_lmwt, maxlmwt = min_lmwt;
			int nnj = 0; //NOTE: for in case there are less LMWT's than number of threads (nj)
			for (int JOBID = 1; JOBID <= nj; JOBID++)
			{
				nnj++;
				//distribute the LMWT's to each thread
				maxlmwt = minlmwt + dLMWT;
				//NOTE: the last thread will have less work in case nLMWT is not dividable by nj
				if (maxlmwt > max_lmwt) maxlmwt = max_lmwt; 

				//start the parallel jobs
				_threads.emplace_back(
					LaunchJobComputeWer,
					JOBID,
					decode_mbr,
					minlmwt,
					maxlmwt,
					wip,
					beam,
					symtab,
					wer_hyp_filter,
					dir);

				minlmwt = maxlmwt + 1;
				if (minlmwt > max_lmwt) break;
			}
			//wait for the threads till they are ready
			for (auto& t : _threads) {
				t.join();
			}
			//check return values from the threads/jobs
			for (int JOBID = 1; JOBID <= nnj; JOBID++) {
				if (_ret[JOBID - 1] < 0)
					return -1;
			}
		} ///_wip

		//clean up
		try {
			if (fs::exists(lat_merged)) fs::remove(lat_merged);
		}
		catch (const std::exception& ex) {
			LOGTW_WARNING << "Could not free up memory and/or delete temporary files. Reason: " << ex.what() << ".";
		}

		//clean up all tmp files
		if (DeleteAllMatching(dir, boost::regex("^(lat.merged\\.).*")) < 0) {
			LOGTW_WARNING << "Could not clean up temporary files lat.merged.*";
		}
		//NOTE: do not delete the 'penalty_' files because they are needed later

	} ///stage <= 0

	if (stage <= 1)
	{
		std::vector<fs::path> _wer;
		for (std::string wip : _wip) {
			for (int LMWT = min_lmwt; LMWT <= max_lmwt; LMWT++)	{
				fs::path p(dir / ("wer_"+ std::to_string(LMWT) + "_" + wip));
				_wer.push_back(p);
			}
		}
		std::string best_wip;
		int best_lmwt = -1;
		if (BestWer(_wer, best_wip, best_lmwt) < 0) return -1;

		if (stats)
		{
			if (CreateDir(dir / "scoring_kaldi" / "wer_details") < 0) return -1;
			//record best language model weight and best word insertion penalty
			fs::ofstream f_lmwt(dir / "scoring_kaldi" / "wer_details" / "lmwt", fs::ofstream::binary | fs::ofstream::out);
			fs::ofstream f_wip(dir / "scoring_kaldi" / "wer_details" / "wip", fs::ofstream::binary | fs::ofstream::out);
			if (!f_lmwt || !f_wip) {
				LOGTW_ERROR << "Could not save file (lmwt, wip).";
				return -1;
			}
			f_lmwt << best_lmwt << "\n";
			f_wip << best_wip << "\n";
			f_lmwt.flush(); f_lmwt.close();
			f_wip.flush(); f_wip.close();

			//1. align-text 
			string_vec options_align_text;
			//ComputeWer
			options_align_text.push_back("--print-args=false");
			options_align_text.push_back("--special-symbol='***'");
			options_align_text.push_back("ark:" + (dir / "scoring_kaldi" / "test_filt.txt").string());
			options_align_text.push_back("ark:" + (dir / "scoring_kaldi" / ("penalty_" + best_wip) / (std::to_string(best_lmwt) + ".txt")).string());
			options_align_text.push_back("ark,t:" + (dir / "scoring_kaldi" / ("penalty_" + best_wip) / (std::to_string(best_lmwt) + ".at")).string());
			try {
				StrVec2Arg args(options_align_text);
				if (AlignText(args.argc(), args.argv()) < 0) return -1;
			}
			catch (const std::exception& ex)
			{
				LOGTW_FATALERROR << "Error in (AlignText). Reason: " << ex.what();
				return -1;
			}
			//2. wer_per_utt_details
			if (WerPerUttDetails(dir / "scoring_kaldi" / ("penalty_" + best_wip) / (std::to_string(best_lmwt) + ".at"),
								 dir / "scoring_kaldi" / "wer_details" / "per_utt", "'***'") < 0) return -1;
			//3. 
			if (WerPerSpkDetails(dir / "scoring_kaldi" / "wer_details" / "per_utt", data / "utt2spk",
				dir / "scoring_kaldi" / "wer_details" / "per_spk") < 0) return -1;

			//1.
			if (WerOpsDetails(dir / "scoring_kaldi" / "wer_details" / "per_utt", 
				dir / "scoring_kaldi" / "wer_details" / "per_utt.temp", "'***'") < 0) return -1;
			//2. sort on field 1 and then 4 and then 2 and then 3 (1 based here)
			StringTable t_details;
			if (ReadStringTable((dir / "scoring_kaldi" / "wer_details" / "per_utt.temp").string(), t_details) < 0) return -1;
			//NOTE: only sorting on fields 1 and 4 and reversing (descending) fields 4
			if (SortStringTable(t_details, 0, 3, "string", "number", false, true, false) < 0) return -1;
			//3. save output to dir/scoring_kaldi/wer_details/ops
			if (SaveStringTable((dir / "scoring_kaldi" / "wer_details" / "ops").string(), t_details) < 0) return -1;

			//compute confidence interval
			CreateDir(dir / "scoring_kaldi" / "log");
			fs::ofstream file_log_bci(dir / "scoring_kaldi" / "log" / "wer_bootci.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log_bci) LOGTW_WARNING << "Log file is not accessible " << (dir / "scoring_kaldi" / "log" / "wer_bootci.log").string() << ".";
			string_vec options_compute_wer_bootci;
			//ComputeWer
			options_compute_wer_bootci.push_back("--print-args=false");
			options_compute_wer_bootci.push_back("--mode=present");
			options_compute_wer_bootci.push_back("ark:" + (dir / "scoring_kaldi" / "test_filt.txt").string());
			options_compute_wer_bootci.push_back("ark:" + (dir / "scoring_kaldi" / ("penalty_" + best_wip) / (std::to_string(best_lmwt) + ".txt")).string());
			options_compute_wer_bootci.push_back((dir / "scoring_kaldi" / "wer_details" / "wer_bootci").string());
			try {
				StrVec2Arg args(options_compute_wer_bootci);
				if (ComputeWerBootci(args.argc(), args.argv(), file_log_bci) < 0) return -1;
			}
			catch (const std::exception& ex)
			{
				LOGTW_FATALERROR << "Error in (ComputeWerBootci). Reason: " << ex.what();
				return -1;
			}

			//clean up files
			try	{
				if (fs::exists((dir / "scoring_kaldi" / ("penalty_" + best_wip) / (std::to_string(best_lmwt) + ".at"))))
					fs::remove(dir / "scoring_kaldi" / ("penalty_" + best_wip) / (std::to_string(best_lmwt) + ".at"));
				if (fs::exists(dir / "scoring_kaldi" / "wer_details" / "per_utt.temp"))
					fs::remove(dir / "scoring_kaldi" / "wer_details" / "per_utt.temp");

			}
			catch (const std::exception& ex)
			{
				LOGTW_WARNING << "Could not remove temporary files because " << ex.what();
			}
		} ///stats

	} ///stage <= 1

	return 0;
}


static void LaunchJobComputeWer(
	int JOBID,
	bool decode_mbr,
	int minlmwt,
	int maxlmwt,
	std::string	wip,
	double beam,
	fs::path symtab,
	UMAPSS wer_hyp_filter,
	fs::path dir
)
{
	//logfile NOTE: we make only 1 log file per thread instead of a lot of log files per LMWT!
	fs::path log(dir / "scoring_kaldi" / ("penalty_" + wip) / "log" / ("LMWT." + std::to_string(JOBID) + ".log"));
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log << ".";
	int ret;

	for (int LMWT = minlmwt; LMWT <= maxlmwt; LMWT++) 
	{
		//prepare a file postfix for this thread and LMWT
		std::string JOBIDLMWT(std::to_string(JOBID) + "." + std::to_string(LMWT));
		string_vec options_lattice_scale, options_lattice_add_penalty;
		//LatticeScale
		options_lattice_scale.push_back("--print-args=false");
		options_lattice_scale.push_back("--inv-acoustic-scale=" + std::to_string(LMWT));
		options_lattice_scale.push_back("ark:"+(dir / "lat.merged").string()); //input
		options_lattice_scale.push_back("ark:" + (dir / ("lat.merged." + JOBIDLMWT + ".tmp")).string()); //output
		try {
			StrVec2Arg args(options_lattice_scale);
			//NOTE: must take care of reading lat.merged in several threads; try mutex lock/unlock while reading
			//lat.merged must be locked read because all threads are needing it
			m_oMutex.lock();
			ret = LatticeScale(args.argc(), args.argv(), file_log);
			m_oMutex.unlock();
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (LatticeScale). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
		//LatticeAddPenalty
		options_lattice_add_penalty.push_back("--print-args=false");
		options_lattice_add_penalty.push_back("--word-ins-penalty=" + wip);
		options_lattice_add_penalty.push_back("ark:" + (dir / ("lat.merged." + JOBIDLMWT + ".tmp")).string()); //input
		options_lattice_add_penalty.push_back("ark:" + (dir / ("lat.merged." + JOBIDLMWT)).string()); //output
		try {
			StrVec2Arg args(options_lattice_add_penalty);
			ret = LatticeAddPenalty(args.argc(), args.argv(), file_log);
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (LatticeAddPenalty). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}

		if (decode_mbr) {
			string_vec options_lattice_prune, options_lattice_mbr_decode;
			//LatticePrune
			options_lattice_prune.push_back("--print-args=false");
			options_lattice_prune.push_back("--beam="+ std::to_string(beam));
			options_lattice_prune.push_back("ark:" + (dir / ("lat.merged." + JOBIDLMWT)).string()); //input
			options_lattice_prune.push_back("ark:" + (dir / ("lat.merged." + JOBIDLMWT + ".p")).string()); //output
			try {
				StrVec2Arg args(options_lattice_prune);
				ret = LatticePrune(args.argc(), args.argv(), file_log);
			}
			catch (const std::exception& ex)
			{
				LOGTW_FATALERROR << " in (LatticePrune). Reason: " << ex.what();
				_ret.push_back(-1);
				return;
			}
			//LatticeMbrDecode
			options_lattice_mbr_decode.push_back("--print-args=false");
			options_lattice_mbr_decode.push_back("--word-symbol-table=" + symtab.string());
			options_lattice_mbr_decode.push_back("ark:" + (dir / ("lat.merged." + JOBIDLMWT + ".p")).string()); //input
			options_lattice_mbr_decode.push_back("ark,t:" + (dir / ("lat.merged." + JOBIDLMWT + ".bp")).string()); //output
			try {
				StrVec2Arg args(options_lattice_mbr_decode);
				ret = LatticeMbrDecode(args.argc(), args.argv(), file_log);
			}
			catch (const std::exception& ex)
			{
				LOGTW_FATALERROR << "Error in (LatticeMbrDecode). Reason: " << ex.what();
				_ret.push_back(-1);
				return;
			}
		}
		else {
			string_vec options_lattice_best_path;
			//LatticeBestPath
			options_lattice_best_path.push_back("--print-args=false");
			options_lattice_best_path.push_back("--word-symbol-table=" + symtab.string());
			options_lattice_best_path.push_back("ark:" + (dir / ("lat.merged." + JOBIDLMWT)).string()); //input
			options_lattice_best_path.push_back("ark,t:" + (dir / ("lat.merged." + JOBIDLMWT + ".bp")).string()); //output
			try {
				StrVec2Arg args(options_lattice_best_path);
				ret = LatticeBestPath(args.argc(), args.argv(), file_log);
			}
			catch (const std::exception& ex)
			{
				LOGTW_FATALERROR << "Error in (LatticeBestPath). Reason: " << ex.what();
				_ret.push_back(-1);
				return;
			}
		} ///else decode_mbr
	
		//int2sym
		StringTable t_symtab, t_LMWT;
		if (ReadStringTable(symtab.string(), t_symtab) < 0) 
		{//symtab
			_ret.push_back(-1);
			return;
		}
		if (ReadStringTable((dir / ("lat.merged." + JOBIDLMWT + ".bp")).string(), t_LMWT) < 0) 
		{ //input
			_ret.push_back(-1);
			return;
		}
		fs::path sym_out(dir / ("lat.merged." + JOBIDLMWT + ".int2sym"));
		if (Int2Sym(t_symtab, t_LMWT, sym_out, 1, -1) < 0) { //NOTE: fields 2- (zero based index 1- till the end)
			_ret.push_back(-1);
			return;
		}
		//hyp_filtering
		if (FilterFile(sym_out, dir / "scoring_kaldi" / ("penalty_"+wip) / (std::to_string(LMWT) + ".txt"), wer_hyp_filter) < 0) {
			_ret.push_back(-1);
			return;
		}

		//compute word error rate
		string_vec options_compute_wer;
		//ComputeWer
		options_compute_wer.push_back("--print-args=false");
		options_compute_wer.push_back("--mode=present");
		options_compute_wer.push_back("ark:" + (dir / "scoring_kaldi" / "test_filt.txt").string());
		options_compute_wer.push_back("ark,p:" + (dir / "scoring_kaldi" / ("penalty_" + wip) / (std::to_string(LMWT) + ".txt")).string());
		options_compute_wer.push_back((dir / ("wer_" + std::to_string(LMWT) + "_" + wip)).string()); //output
		try {
			StrVec2Arg args(options_compute_wer);
			ret = ComputeWer(args.argc(), args.argv(), file_log);
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (ComputeWer). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
	} ///for (int LMWT

	_ret.push_back(0);
}

