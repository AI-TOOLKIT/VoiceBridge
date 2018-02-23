/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012-2016  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"
#include <kaldi-win/utility/strvec2arg.h>

/*
	IMPORTANT NOTES: 
	 - using 'JOBID' in file names which will be replaced by the numerical id of the job (parallel process). The word JOBID
	   should not be in any input file name.
*/

static void LaunchJob(int argc1, char *argv1[], int argc2, char *argv2[], fs::path log);
static void LaunchJob_segments(int argc0, char *argv0[], int argc1, char *argv1[], int argc2, char *argv2[], fs::path log);
//the return values from each thread/job
std::vector<int> _ret_comp, _ret_copy, _ret_extr;

VOICEBRIDGE_API int MakeMfcc(
	fs::path datadir,			//data directory
	fs::path mfcc_config,		//mfcc config file path
								//NOTE: when the config file is passed with the "--config" option specifier then
								//		it is read automatically when parsing the otions!
	int nj,						//default: 4, number of parallel jobs
	bool compress,				//default: true, compress mfcc features
	bool write_utt2num_frames	//default: false, if true writes utt2num_frames	
)
{
	fs::path logdir = datadir / "log";
	fs::path mfccdir = datadir / "mfcc";
	std::string name = datadir.stem().string();

	try {
		fs::create_directory(logdir);
		fs::create_directory(mfccdir);
		if (fs::exists(datadir / "feats.scp"))
		{
			fs::create_directory(datadir / "backup");
			LOGTW_INFO << "Moving data/feats.scp to data/backup in " << datadir.string() << ".";
			fs::copy_file(datadir / "feats.scp", datadir / "backup" / "feats.scp", fs::copy_option::overwrite_if_exists);
			fs::remove(datadir / "feats.scp");
		}
	} catch (const std::exception& ex)	{
		LOGTW_ERROR << " " << ex.what() << ".";
		return -1;
	}

	//required:
	if (CheckFileExistsAndNotEmpty(mfcc_config, true) < 0) return -1;
	fs::path scp = datadir / "wav.scp";
	if (CheckFileExistsAndNotEmpty(scp, true) < 0) return -1;

	//validate data directory
	if (ValidateData(datadir, true, false, true) < 0) return -1;

	//spk2warp, utt2warp
	std::vector<std::string> vtln_opts;
	if (fs::exists(datadir / "spk2warp"))
	{
		LOGTW_INFO << "Using VTLN warp factors from data/spk2warp.";	
		vtln_opts.push_back("--vtln-map=ark:" + (datadir / "spk2warp").string());
		vtln_opts.push_back("--utt2spk=ark:" + (datadir / "utt2spk").string());
	}
	else if (fs::exists(datadir / "utt2warp"))
	{
		LOGTW_INFO << "Using VTLN warp factors from data/utt2warp.";
		vtln_opts.push_back("--vtln-map=ark:" + (datadir / "utt2warp").string());
	}

	//setup parallel processing
	std::vector<fs::path> _fullpaths;
	for (int i = 1; i <= nj; i++) {
		fs::path dir(mfccdir / ("raw_mfcc_" + name + "." + std::to_string(i) + ".ark"));
		_fullpaths.push_back(dir);
	}

	std::vector<std::string> write_num_frames_opt;
	if (write_utt2num_frames) {
		write_num_frames_opt.push_back("--write-num-frames=ark,t:" + (logdir / "utt2num_frames.JOBID").string());
		//NOTE: "t:" means text mode
	}

	//this error file could have been made by a former run
	DeleteAllMatching(logdir, boost::regex("^(\\.error).*"));

	if (fs::exists(datadir / "segments"))
	{
		LOGTW_INFO << "Segments file exists: using that.";
		std::vector<fs::path> split_segments;
		for (int n = 1; n <= nj; n++) {
			split_segments.push_back((logdir / ("segments." + std::to_string(n))));
		}
		if (SplitScp(datadir / "segments", split_segments) < 0) return -1;

		std::vector<std::thread> _threads;
		std::vector<string_vec> _extract_options, _compute_mfc_options, _copy_feats_options;		
		std::vector<StrVec2Arg *> _args0, _args1, _args2;

		//parallel section
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			//add all options for extract-segments
			string_vec extract_options;
			extract_options.push_back("--print-args=false"); //NOTE: do not print arguments
			extract_options.push_back("scp,p:" + scp.string());
			extract_options.push_back((logdir / ("segments.JOBID")).string());
			//NOTE: the output must be written into a temporary file which will be passed to copy-feats
			extract_options.push_back("ark:" + ((logdir / ("segments.JOBID.temp")).string()));
			//replace 'JOBID' with the current job ID of the thread
			for (std::string &s : extract_options)
			{//NOTE: accesing by ref for in place editing
				ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
			}
			_extract_options.push_back(extract_options);
			StrVec2Arg *args0 = new StrVec2Arg(_extract_options[JOBID-1]);
			_args0.push_back(args0);

			//add all options for compute-mfcc-feats
			string_vec compute_mfc_options;
			compute_mfc_options.insert(compute_mfc_options.end(), vtln_opts.begin(), vtln_opts.end());
			compute_mfc_options.push_back("--print-args=false"); //NOTE: do not print arguments
			compute_mfc_options.push_back("--verbose=2"); //NOTE: verbose may be decreased from 2 to 1!
			compute_mfc_options.push_back("--config=" + mfcc_config.string());
			compute_mfc_options.push_back("ark:" + ((logdir / ("segments.JOBID.temp")).string()));
			//NOTE: the output must be written into a temporary file which will be passed to copy-feats
			compute_mfc_options.push_back("ark:" + ((logdir / ("wav_" + name + ".JOBID.mfctemp")).string()));
			//replace 'JOBID' with the current job ID of the thread
			for (std::string &s : compute_mfc_options)
			{//NOTE: accesing by ref for in place editing
				ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
			}
			_compute_mfc_options.push_back(compute_mfc_options);
			StrVec2Arg *args1 = new StrVec2Arg(_compute_mfc_options[JOBID-1]);
			_args1.push_back(args1);

			//add all options for copy-feats
			string_vec copy_feats_options;
			copy_feats_options.insert(copy_feats_options.end(), write_num_frames_opt.begin(), write_num_frames_opt.end());
			copy_feats_options.push_back("--print-args=false"); //NOTE: do not print arguments
			copy_feats_options.push_back("--compress=" + bool_as_text(compress));
			copy_feats_options.push_back("ark:" + ((logdir / ("wav_" + name + ".JOBID.mfctemp")).string()));
			copy_feats_options.push_back("ark,scp:" + (mfccdir / ("raw_mfcc_" + name + ".JOBID.ark")).string() + "," + (mfccdir / ("raw_mfcc_" + name + ".JOBID.scp")).string());
			//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
			for (std::string &s : copy_feats_options)
			{//NOTE: accesing by ref for in place editing
				ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
			}
			_copy_feats_options.push_back(copy_feats_options);
			StrVec2Arg *args2 = new StrVec2Arg(_copy_feats_options[JOBID-1]);
			_args2.push_back(args2);

			//logfile 
			fs::path log(logdir / ("make_mfcc_"+name+"."+ std::to_string(JOBID) +".log"));

			_threads.emplace_back(LaunchJob_segments,
				_args0[JOBID - 1]->argc(), _args0[JOBID - 1]->argv(),		//params for ExtractSegments
				_args1[JOBID - 1]->argc(), _args1[JOBID - 1]->argv(),		//params for ComputeMFCCFeats
				_args2[JOBID - 1]->argc(), _args2[JOBID - 1]->argv(),		//params for CopyFeats
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}
		//clean up
		try {
			for (int JOBID = 1; JOBID <= nj; JOBID++) {
				delete _args0[JOBID - 1];
				delete _args1[JOBID - 1];
				delete _args2[JOBID - 1];
			}
			_args0.clear();
			_args1.clear();
			_args2.clear();
		}
		catch (const std::exception&)
		{
			LOGTW_WARNING << "Could not free up memory (StrVec2Arg).";
		}
		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret_extr[JOBID - 1] < 0 || _ret_comp[JOBID - 1] < 0 || _ret_copy[JOBID - 1] < 0)
				return -1;
		}
	}
	else 
	{
		LOGTW_INFO << "No segments file exists: assuming wav.scp indexed by utterance.";
		std::vector<fs::path> split_segments;
		for (int n = 1; n <= nj; n++) {
			split_segments.push_back((logdir / ("wav_" + name + "." + std::to_string(n) + ".scp")));
		}
		if (SplitScp(scp, split_segments) < 0) return -1;

		//NOTE: add ',p' to the input rspecifier so that we can just skip over utterances that have bad wave data.

		std::vector<std::thread> _threads;
		std::vector<string_vec> _compute_mfc_options, _copy_feats_options;

		//NOTE: must keep the parameters to the function call started in different threads in order that the threads can access it
		std::vector<StrVec2Arg *> _args1, _args2;
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			//add all options for compute-mfcc-feats
			string_vec compute_mfc_options;
			compute_mfc_options.insert(compute_mfc_options.end(), vtln_opts.begin(), vtln_opts.end());
			compute_mfc_options.push_back("--print-args=false"); //NOTE: do not print arguments
			compute_mfc_options.push_back("--verbose=2"); //NOTE: verbose may be decreased from 2 to 1!
			compute_mfc_options.push_back("--config=" + mfcc_config.string());
			compute_mfc_options.push_back("scp,p:" + ((logdir / ("wav_" + name + ".JOBID.scp")).string()));
			//NOTE: the output must be written into a temporary file which will be passed to copy-feats
			compute_mfc_options.push_back("ark:" + ((logdir / ("wav_" + name + ".JOBID.mfctemp")).string()));
			//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
			for (std::string &s : compute_mfc_options)
			{//NOTE: accesing by ref for in place editing
				ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
			}
			_compute_mfc_options.push_back(compute_mfc_options);
			StrVec2Arg *args1 = new StrVec2Arg(_compute_mfc_options[JOBID-1]);
			_args1.push_back(args1);

			//add all options for copy-feats
			string_vec copy_feats_options;
			copy_feats_options.insert(copy_feats_options.end(), write_num_frames_opt.begin(), write_num_frames_opt.end());
			copy_feats_options.push_back("--print-args=false"); //NOTE: do not print arguments
			copy_feats_options.push_back("--compress="+ bool_as_text(compress));
			copy_feats_options.push_back("ark:" + ((logdir / ("wav_" + name + ".JOBID.mfctemp")).string()));
			copy_feats_options.push_back("ark,scp:"+ (mfccdir / ("raw_mfcc_" + name + ".JOBID.ark")).string() + "," + (mfccdir / ("raw_mfcc_"+name+".JOBID.scp")).string());
			//replace 'JOBID' with the current job ID of the thread
			for (std::string &s : copy_feats_options)
			{//NOTE: accesing by ref for in place editing
				ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
			}
			_copy_feats_options.push_back(copy_feats_options);
			StrVec2Arg *args2 = new StrVec2Arg(_copy_feats_options[JOBID-1]);
			_args2.push_back(args2);

			//logfile 
			fs::path log(logdir / ("make_mfcc_" + name + "." + std::to_string(JOBID) + ".log"));

			_threads.emplace_back(LaunchJob, 
				_args1[JOBID-1]->argc(), _args1[JOBID - 1]->argv(),			//params for ComputeMFCCFeats
				_args2[JOBID - 1]->argc(), _args2[JOBID - 1]->argv(),		//params for CopyFeats
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}
		//clean up
		try	{
			for (int JOBID = 1; JOBID <= nj; JOBID++) {
				delete _args1[JOBID - 1];
				delete _args2[JOBID - 1];
			}
			_args1.clear();
			_args2.clear();
		}
		catch (const std::exception&)
		{
			LOGTW_WARNING << " Could not free up memory (StrVec2Arg)."; 
		}
		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret_comp[JOBID - 1] < 0 || _ret_copy[JOBID - 1] < 0)
				return -1;
		}
	}

	// concatenate the files together.
	std::vector<fs::path> _infeats, _inutt2num;
	for (int JOBID = 1; JOBID <= nj; JOBID++) {
		//.scp
		_infeats.push_back(mfccdir / ("raw_mfcc_"+name+"."+ std::to_string(JOBID) +".scp"));
		//utt2num_frames
		if (write_utt2num_frames)
			_inutt2num.push_back(logdir / ("utt2num_frames." + std::to_string(JOBID)));
	}
	fs::path feats_scp(datadir / "feats.scp"), utt2num_frames(datadir / "utt2num_frames");
	//.scp
	if (MergeFiles(_infeats, feats_scp) < 0) return -1;
	//utt2num_frames
	if (write_utt2num_frames) {
		if (MergeFiles(_inutt2num, utt2num_frames) < 0) return -1;
		try {
			for (int JOBID = 1; JOBID <= nj; JOBID++)
				fs::remove(logdir / ("utt2num_frames." + std::to_string(JOBID)));
		} catch (const std::exception&) {}
	}
	//clean up temporary files
	try {
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (fs::exists(logdir / ("wav_" + name + "." + std::to_string(JOBID) + ".mfctemp")))
				fs::remove(logdir / ("wav_" + name + "." + std::to_string(JOBID) +".mfctemp"));
			if (fs::exists(logdir / ("wav_" + name + "." + std::to_string(JOBID) + ".scp")))
				fs::remove(logdir / ("wav_" + name + "." + std::to_string(JOBID) + ".scp"));
			if (fs::exists(logdir / ("segments." + std::to_string(JOBID) + ".temp")))
				fs::remove(logdir / ("segments." + std::to_string(JOBID) + ".temp"));
			if (fs::exists(logdir / ("segments." + std::to_string(JOBID))))
				fs::remove(logdir / ("segments." + std::to_string(JOBID)));
		}
	}
	catch (const std::exception&) {}

	StringTable tbl_feats, tbl_utt2spk;
	if (ReadStringTable((datadir / "feats.scp").string(), tbl_feats) < 0) return -1;
	if (ReadStringTable((datadir / "utt2spk").string(), tbl_utt2spk) < 0) return -1;
	int nf = tbl_feats.size();
	int nu = tbl_utt2spk.size();
	if (nu != nf) 
	{
		LOGTW_WARNING << "It seems not all of the feature files were successfully processed (" << nf << " != " << nu << ").";
		LOGTW_INFO << "Fixing data directory...";
		//call here automatically FixDataDir
		if (FixDataDir(datadir) < 0) {
			LOGTW_ERROR << " failed fixing data directory.";
			return -1;
		}
		LOGTW_INFO << "!!!!!!! Data directory is fixed with success. Please restart the software. !!!!!!!";
		return -1;
	}

	if ( nf < (nu - (nu / 20)) ) {
		LOGTW_ERROR << "Less than 95% of the features were successfully generated. Probably a serious error.";
		return -1;
	}

	LOGTW_INFO << "Succeeded creating MFCC features for " << name << ".";

	return 0;
}


//NOTE: this will be called from several threads
static void LaunchJob(int argc1, char *argv1[], int argc2, char *argv2[], fs::path log)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	try	{
		int ret1 = ComputeMFCCFeats(argc1, argv1, file_log);
		_ret_comp.push_back(ret1);
		int ret2 = CopyFeats(argc2, argv2, file_log);
		_ret_copy.push_back(ret2);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ComputeMFCCFeats, CopyFeats). Reason: " << ex.what();
		_ret_copy.push_back(-1);
		return;
	}
}

//NOTE: this will be called from several threads
static void LaunchJob_segments(int argc0, char *argv0[], int argc1, char *argv1[], int argc2, char *argv2[], fs::path log)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	try	{
		int ret0 = ExtractSegments(argc0, argv0, file_log);
		_ret_extr.push_back(ret0);
		int ret1 = ComputeMFCCFeats(argc1, argv1, file_log);
		_ret_comp.push_back(ret1);
		int ret2 = CopyFeats(argc2, argv2, file_log);
		_ret_copy.push_back(ret2);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ExtractSegments, ComputeMFCCFeats, CopyFeats). Reason: " << ex.what();
		_ret_copy.push_back(-1);
		return;
	}
}



