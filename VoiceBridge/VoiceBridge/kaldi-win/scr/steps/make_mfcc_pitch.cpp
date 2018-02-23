/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on : Copyright 2013 The Shenzhen Key Laboratory of Intelligent Media and Speech,
    PKU-HKUST Shenzhen Hong Kong Institution (Author: Wei Shi),
	2016  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/scr/kaldi_scr2.h"
#include "kaldi-win/src/kaldi_src.h"
#include <kaldi-win/utility/strvec2arg.h>

/*
	IMPORTANT NOTES: 
	 - using 'JOBID' in file names which will be replaced by the numerical id of the job (parallel process). The word JOBID
	   should not be in any input file name.
*/

static void LaunchJob(
	int JOBID,
	string_vec compute_mfc_options,
	string_vec copy_feats_options,
	string_vec compute_pitch_options,
	string_vec process_pitch_options,
	string_vec paste_feats_options,
	fs::path log
);

static void LaunchJob_segments(
	int JOBID,
	string_vec extract_options,
	string_vec compute_mfc_options,
	string_vec copy_feats_options,
	string_vec extract_pitch_options,
	string_vec compute_pitch_options,
	string_vec process_pitch_options,
	string_vec paste_feats_options,
	fs::path log
);
//the return values from each thread/job
std::vector<int> _ret;

/*
	Combines MFCC and Pitch features together
	NOTE: may increase or decrease the WER depending on the input data! Try always both MakeMfcc()
		  and MakeMfccPitch() and use the one which is best for your data!
*/
VOICEBRIDGE_API int MakeMfccPitch(
	fs::path datadir,					//data directory
	fs::path mfcc_config,				//mfcc config file path passed to compute-mfcc-feats
										//NOTE: when the config file is passed with the "--config" option specifier then
										//		it is read automatically when parsing the otions!
	fs::path pitch_config,				//pitch config file path (pitch.conf) : compute-kaldi-pitch-feats
	fs::path pitch_postprocess_config,	//pitch postprocess config file path (pitch.conf) : process-kaldi-pitch-feats
	int nj,								//default: 4, number of parallel jobs
	bool compress,						//default: true, compress mfcc features
	bool write_utt2num_frames,			//default: false, if true writes utt2num_frames
	int paste_length_tolerance		//default: 2, length tolerance passed to paste-feats
)
{
	fs::path logdir = datadir / "log";
	fs::path mfccdir = datadir / "mfcc_pitch";
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
	fs::path scp = datadir / "wav.scp";
	if (CheckFileExistsAndNotEmpty(scp, true) < 0) return -1;

	//required
	if (CheckFileExistsAndNotEmpty(mfcc_config, true) < 0) return -1;
	if (CheckFileExistsAndNotEmpty(pitch_config, true) < 0) return -1;

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

	std::vector<std::thread> _threads;

	if (fs::exists(datadir / "segments"))
	{
		LOGTW_INFO << "Segments file exists: using that.";
		std::vector<fs::path> split_segments;
		for (int n = 1; n <= nj; n++) {
			split_segments.push_back((logdir / ("segments." + std::to_string(n))));
		}
		if (SplitScp(datadir / "segments", split_segments) < 0) return -1;
				
		//parallel section
		for (int JOBID = 1; JOBID <= nj; JOBID++) 
		{
			//--- mfcc_feats:
			//add all options for extract-segments
			string_vec extract_options;
			extract_options.push_back("--print-args=false"); //NOTE: do not print arguments
			extract_options.push_back("scp,p:" + scp.string());
			extract_options.push_back((logdir / ("segments.JOBID")).string());
			extract_options.push_back("ark:" + ((logdir / ("segments_mfcc.JOBID.temp")).string())); //output

			//add all options for compute-mfcc-feats
			string_vec compute_mfc_options;
			compute_mfc_options.insert(compute_mfc_options.end(), vtln_opts.begin(), vtln_opts.end());
			compute_mfc_options.push_back("--print-args=false"); //NOTE: do not print arguments
			compute_mfc_options.push_back("--verbose=2"); //NOTE: verbose may be decreased from 2 to 1!
			compute_mfc_options.push_back("--config=" + mfcc_config.string());
			compute_mfc_options.push_back("ark:" + ((logdir / ("segments_mfcc.JOBID.temp")).string())); //input from extract-segments 
			compute_mfc_options.push_back("ark:" + ((logdir / ("wav_" + name + ".JOBID.mfctemp")).string())); //output

			//--- pitch_feats:
			//pitch extract-segments
			string_vec extract_pitch_options;
			extract_pitch_options.push_back("--print-args=false");
			extract_pitch_options.push_back("scp,p:" + scp.string());
			extract_pitch_options.push_back((logdir / ("segments.JOBID")).string());
			extract_pitch_options.push_back("ark:" + ((logdir / ("segments.JOBID.temp")).string())); //output

			//compute-kaldi-pitch-feats
			string_vec compute_pitch_options;
			compute_pitch_options.push_back("--print-args=false");
			compute_pitch_options.push_back("--verbose=2");
			compute_pitch_options.push_back("--config=" + pitch_config.string());
			compute_pitch_options.push_back("ark:" + ((logdir / ("segments.JOBID.temp")).string())); //input from extract-segments 
			compute_pitch_options.push_back("ark:" + ((logdir / ("comppitch.JOBID.temp")).string())); //output

			//process-kaldi-pitch-feats 
			string_vec process_pitch_options;
			process_pitch_options.push_back("--print-args=false");
			if(pitch_postprocess_config !="" && fs::exists(pitch_postprocess_config))
				process_pitch_options.push_back("--config=" + pitch_postprocess_config.string());
			process_pitch_options.push_back("ark:" + ((logdir / ("comppitch.JOBID.temp")).string())); //input from compute-kaldi-pitch-feats
			process_pitch_options.push_back("ark:" + ((logdir / ("procpitch.JOBID.temp")).string())); //output

			//add all options for paste-feats
			string_vec paste_feats_options;
			paste_feats_options.push_back("--print-args=false");
			paste_feats_options.push_back("--length-tolerance=" + std::to_string(paste_length_tolerance));
			paste_feats_options.push_back("ark:" + ((logdir / ("wav_" + name + ".JOBID.mfctemp")).string())); //output from compute-mfcc-feats
			paste_feats_options.push_back("ark,s,cs:" + ((logdir / ("procpitch.JOBID.temp")).string())); //output from process-kaldi-pitch-feats
			paste_feats_options.push_back("ark:" + (mfccdir / ("raw_mfcc_" + name + ".JOBID.temp")).string()); //output

			//add all options for copy-feats
			string_vec copy_feats_options;
			copy_feats_options.push_back("--print-args=false");
			copy_feats_options.push_back("--compress=" + bool_as_text(compress));
			copy_feats_options.insert(copy_feats_options.end(), write_num_frames_opt.begin(), write_num_frames_opt.end());
			copy_feats_options.push_back("ark:" + (mfccdir / ("raw_mfcc_" + name + ".JOBID.temp")).string()); //output from paste-feats
			copy_feats_options.push_back("ark,scp:" + (mfccdir / ("raw_mfcc_" + name + ".JOBID.ark")).string() + "," + (mfccdir / ("raw_mfcc_" + name + ".JOBID.scp")).string());

			//logfile 
			fs::path log(logdir / ("make_mfcc_"+name+"."+ std::to_string(JOBID) +".log"));

			_threads.emplace_back(LaunchJob_segments,
				JOBID,
				extract_options,
				compute_mfc_options,
				copy_feats_options,
				extract_pitch_options,
				compute_pitch_options,
				process_pitch_options,
				paste_feats_options,
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}

		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret[JOBID - 1] < 0)
				return -1;
		}
	}
	else 
	{
		LOGTW_INFO << "No segments file exists: assuming wav.scp indexed by utterance.";
		std::vector<fs::path> split_scps;
		for (int n = 1; n <= nj; n++) {
			split_scps.push_back((logdir / ("wav_" + name + "." + std::to_string(n) + ".scp")));
		}
		if (SplitScp(scp, split_scps) < 0) return -1;

		//NOTE: add ',p' to the input rspecifier so that we can just skip over utterances that have bad wave data.

		//parallel section
		for (int JOBID = 1; JOBID <= nj; JOBID++) 
		{
			//--- mfcc_feats:
			//add all options for extract-segments
			//add all options for compute-mfcc-feats
			string_vec compute_mfc_options;
			compute_mfc_options.insert(compute_mfc_options.end(), vtln_opts.begin(), vtln_opts.end());
			compute_mfc_options.push_back("--print-args=false"); //NOTE: do not print arguments
			compute_mfc_options.push_back("--verbose=2"); //NOTE: verbose may be decreased from 2 to 1!
			compute_mfc_options.push_back("--config=" + mfcc_config.string());
			compute_mfc_options.push_back("scp,p:" + ((logdir / ("wav_" + name + ".JOBID.scp")).string()));
			compute_mfc_options.push_back("ark:" + ((logdir / ("wav_" + name + ".JOBID.mfctemp")).string())); //output

			//--- pitch_feats:
			//compute-kaldi-pitch-feats
			string_vec compute_pitch_options;
			compute_pitch_options.push_back("--print-args=false");
			compute_pitch_options.push_back("--verbose=2");
			compute_pitch_options.push_back("--config=" + pitch_config.string());
			compute_pitch_options.push_back("scp,p:" + ((logdir / ("wav_" + name + ".JOBID.scp")).string())); //input
			compute_pitch_options.push_back("ark:" + ((logdir / ("comppitch.JOBID.temp")).string())); //output
			//process-kaldi-pitch-feats 
			string_vec process_pitch_options;
			process_pitch_options.push_back("--print-args=false");
			if (pitch_postprocess_config != "" && fs::exists(pitch_postprocess_config))
				process_pitch_options.push_back("--config=" + pitch_postprocess_config.string());
			process_pitch_options.push_back("ark:" + ((logdir / ("comppitch.JOBID.temp")).string())); //input from compute-kaldi-pitch-feats
			process_pitch_options.push_back("ark:" + ((logdir / ("procpitch.JOBID.temp")).string())); //output

			//add all options for paste-feats
			string_vec paste_feats_options;
			paste_feats_options.push_back("--print-args=false");
			paste_feats_options.push_back("--length-tolerance=" + std::to_string(paste_length_tolerance));
			paste_feats_options.push_back("ark:" + ((logdir / ("wav_" + name + ".JOBID.mfctemp")).string())); //output from compute-mfcc-feats
			paste_feats_options.push_back("ark,s,cs:" + ((logdir / ("procpitch.JOBID.temp")).string())); //output from process-kaldi-pitch-feats
			paste_feats_options.push_back("ark:" + (mfccdir / ("raw_mfcc_" + name + ".JOBID.temp")).string()); //output
			//add all options for copy-feats
			string_vec copy_feats_options;
			copy_feats_options.push_back("--print-args=false");
			copy_feats_options.push_back("--compress=" + bool_as_text(compress));
			copy_feats_options.insert(copy_feats_options.end(), write_num_frames_opt.begin(), write_num_frames_opt.end());
			copy_feats_options.push_back("ark:" + (mfccdir / ("raw_mfcc_" + name + ".JOBID.temp")).string()); //output from paste-feats
			copy_feats_options.push_back("ark,scp:" + (mfccdir / ("raw_mfcc_" + name + ".JOBID.ark")).string() + "," + (mfccdir / ("raw_mfcc_" + name + ".JOBID.scp")).string());

			//logfile 
			fs::path log(logdir / ("make_mfcc_" + name + "." + std::to_string(JOBID) + ".log"));

			_threads.emplace_back(LaunchJob,
				JOBID,
				compute_mfc_options,
				copy_feats_options,
				compute_pitch_options,
				process_pitch_options,
				paste_feats_options,
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}

		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret[JOBID - 1] < 0)
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
	DeleteAllMatching(logdir, boost::regex(".*(\\.temp)$"));
	DeleteAllMatching(mfccdir, boost::regex(".*(\\.temp)$"));

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

	LOGTW_INFO << "Succeeded creating MFCC & Pitch features for " << name << ".";

	return 0;
}


//
static void LaunchJob(
	int JOBID,
	string_vec compute_mfc_options,
	string_vec copy_feats_options,
	string_vec compute_pitch_options,
	string_vec process_pitch_options,
	string_vec paste_feats_options,
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";
	//replace JOBID in options
	for (std::string &s : compute_mfc_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : compute_pitch_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : process_pitch_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : copy_feats_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : paste_feats_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	int ret = 0;
	//process the data
	//1. mfcc_feats: compute-mfcc-feats
	//2. pitch_feats: compute-kaldi-pitch-feats -> process-kaldi-pitch-feats 
	//3. paste-feats -> copy-feats
	try {
		StrVec2Arg argsCompMfc(compute_mfc_options);

		ret = ComputeMFCCFeats(argsCompMfc.argc(), argsCompMfc.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed

		StrVec2Arg argsCompPitch(compute_pitch_options);
		StrVec2Arg argsProcPitch(process_pitch_options);

		ret = ComputeKaldiPitchFeats(argsCompPitch.argc(), argsCompPitch.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed

		ret = ProcessKaldiPitchFeats(argsProcPitch.argc(), argsProcPitch.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed

		StrVec2Arg argsCopyF(copy_feats_options);
		StrVec2Arg argsPasteF(paste_feats_options);

		ret = PasteFeats(argsPasteF.argc(), argsPasteF.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed

		ret = CopyFeats(argsCopyF.argc(), argsCopyF.argv(), file_log);
		_ret.push_back(ret);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in LaunchJob (PasteFeats -> CopyFeats). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
}

//
static void LaunchJob_segments(
	int JOBID,
	string_vec extract_options, 
	string_vec compute_mfc_options, 
	string_vec copy_feats_options,
	string_vec extract_pitch_options, 
	string_vec compute_pitch_options, 
	string_vec process_pitch_options, 
	string_vec paste_feats_options,
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";
	//replace JOBID in options
	for (std::string &s : extract_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : compute_mfc_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));	
	for (std::string &s : extract_pitch_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : compute_pitch_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : process_pitch_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : copy_feats_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : paste_feats_options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	int ret = 0;
	//process the data
	//1. mfcc_feats: extract-segments -> compute-mfcc-feats
	//2. pitch_feats: extract-segments -> compute-kaldi-pitch-feats -> process-kaldi-pitch-feats 
	//3. paste-feats -> copy-feats
	try	{
		StrVec2Arg argsExtrMfc(extract_options);
		StrVec2Arg argsCompMfc(compute_mfc_options);

		ret = ExtractSegments(argsExtrMfc.argc(), argsExtrMfc.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed
		
		ret = ComputeMFCCFeats(argsCompMfc.argc(), argsCompMfc.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed

		StrVec2Arg argsExtrPitch(extract_pitch_options);
		StrVec2Arg argsCompPitch(compute_pitch_options);
		StrVec2Arg argsProcPitch(process_pitch_options);

		ret = ExtractSegments(argsExtrPitch.argc(), argsExtrPitch.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed

		ret = ComputeKaldiPitchFeats(argsCompPitch.argc(), argsCompPitch.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed

		ret = ProcessKaldiPitchFeats(argsProcPitch.argc(), argsProcPitch.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed

		StrVec2Arg argsCopyF(copy_feats_options);
		StrVec2Arg argsPasteF(paste_feats_options);

		ret = PasteFeats(argsPasteF.argc(), argsPasteF.argv(), file_log);
		_ret.push_back(ret);
		if (ret < 0) return;  //do not proceed if failed

		ret = CopyFeats(argsCopyF.argc(), argsCopyF.argv(), file_log);
		_ret.push_back(ret);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in LaunchJob_segments (PasteFeats -> CopyFeats). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
}



