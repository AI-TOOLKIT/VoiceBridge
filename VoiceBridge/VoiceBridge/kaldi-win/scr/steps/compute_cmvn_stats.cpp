/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012-2016  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
*/

/*
 Compute cepstral mean and variance statistics per speaker. We do this in just one job; it's fast.
 NOTE: there is no option to do CMVN per utterance.  The idea is that if you did it per utterance it would not make sense to do
 per-speaker fMLLR on top of that (since you'd be doing fMLLR on top of different offsets).  Therefore what would be the use
 of the speaker information?  In this case you should probably make the speaker-ids identical to the utterance-ids.  The
 speaker information does not have to correspond to actual speakers, it's just the level you want to adapt at.
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"
#include <kaldi-win/utility/strvec2arg.h>

VOICEBRIDGE_API int ComputeCmvnStats(
	fs::path datadir,			//data directory
	bool fake,					//default: false, gives you fake cmvn stats that do no normalization.
	bool two_channel,			//default: false, is for two-channel telephone data, there must be no segments
								//file and reco2file_and_channel must be present. It will take only frames 
								//that are louder than the other channel.
	std::string fake_dims		//Generate stats that won't cause normalization for these dimensions (e.g. "13:14:15")
)
{
	fs::path logdir = datadir / "log";
	fs::path cmvndir = datadir / "data";
	std::string name = datadir.stem().string();

	try	{
		fs::create_directory(cmvndir);
		fs::create_directory(logdir);
	} catch (const std::exception& ex)	{
		LOGTW_ERROR << " " << ex.what() << ".";
		return -1;
	}

	if (!fs::exists(datadir / "feats.scp"))	{
		LOGTW_ERROR << " " << (datadir / "feats.scp").string() << " is required but can not be found.";
		return -1;
	}
	if (!fs::exists(datadir / "spk2utt")) {
		LOGTW_ERROR << " " << (datadir / "spk2utt").string() << " is required but can not be found.";
		return -1;
	}

	StringTable tbl_spk2utt;
	if (ReadStringTable((datadir / "spk2utt").string(), tbl_spk2utt) < 0) return -1;

	if (fake)
	{
		//add all options 
		string_vec options;
		options.push_back("--print-args=false"); //NOTE: do not print arguments
		options.push_back("scp:" + (datadir / "feats.scp").string());
		options.push_back((datadir / "feat_to_dim.temp").string()); //NOTE: save into temporary file
		StrVec2Arg args(options);
		FeatToDim(args.argc(), args.argv());
		//read in the output and delete temporary file
		StringTable tbl_feattodim;
		if (ReadStringTable((datadir / "feat_to_dim.temp").string(), tbl_feattodim) < 0) return -1;
		try	{
			fs::remove(datadir / "feat_to_dim.temp");
		} catch (const std::exception&){}
		int dim = StringToNumber<int>(tbl_feattodim[0][0], -1);
		if (dim < 0) {
			LOGTW_ERROR << "Failed creating fake CMVN stats because failed to get the feature dimension of the first feature file from " << (datadir / "feats.scp").string();
			return -1;
		}
		fs::ofstream file_faketemp((datadir / "fake.temp"), fs::ofstream::binary | fs::ofstream::out);
		if (!file_faketemp) {
			LOGTW_ERROR << "Failed creating fake CMVN stats because file is not accessible: " << (datadir / "fake.temp").string();
			return -1;
		}
		for (StringTable::const_iterator it(tbl_spk2utt.begin()), it_end(tbl_spk2utt.end()); it != it_end; ++it)
		{
			file_faketemp << (*it)[0] << " [" << "\n";
			for (int n = 0; n < dim; n++)
				file_faketemp << "0 ";
			file_faketemp << "1" << "\n";
			for (int n = 0; n < dim; n++)
				file_faketemp << "1 ";
			file_faketemp << "0 ]" << "\n";
		}
		file_faketemp.flush(); file_faketemp.close();

		//copy-matrix
		options.clear();
		options.push_back("--print-args=false"); //NOTE: do not print arguments
		options.push_back("ark:" + (datadir / "fake.temp").string());
		options.push_back("ark,scp:" + (cmvndir / ("cmvn_"+name+".ark")).string() + "," + (cmvndir / ("cmvn_"+name+".scp")).string());
		StrVec2Arg args1(options);
		//redirect logging to the log file:
		fs::path log(cmvndir / ("cmvn_" + name + ".ark.log"));
		fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
		if (!file_log) LOGTW_WARNING << " log file is not accessible " << log.string() << ".";
		if (CopyMatrix(args1.argc(), args1.argv(), file_log) < 0) {
			LOGTW_ERROR << "Failed creating fake CMVN stats. See file: " << (logdir / ("cmvn_"+name+".log")).string();
			return -1;
		}

		try {
			fs::remove(datadir / "fake.temp");
		}
		catch (const std::exception&) {}
	}
	else if (two_channel)
	{
		string_vec options;
		options.push_back("--print-args=false"); //NOTE: do not print arguments
		options.push_back((datadir / "reco2file_and_channel").string());
		options.push_back("scp:" + (datadir / "feats.scp").string());
		options.push_back("ark,scp:" + (cmvndir / ("cmvn_" + name + ".ark")).string() + "," + (cmvndir / ("cmvn_" + name + ".scp")).string());
		StrVec2Arg args(options);
		fs::path log(cmvndir / ("cmvn_" + name + ".log"));
		fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
		if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";
		if (ComputeCmvnStatsTwoChannel(args.argc(), args.argv(), file_log) < 0) {
			LOGTW_ERROR << "Error computing CMVN stats (using two-channel method).";
			return -1;
		}
	}
	else if (fake_dims!="")
	{
		string_vec options;
		options.push_back("--print-args=false"); //NOTE: do not print arguments
		options.push_back("--spk2utt=ark:" + (datadir / "spk2utt").string());
		options.push_back("scp:" + (datadir / "feats.scp").string());
		options.push_back("ark:" + (cmvndir / ("cmvn_" + name + ".ark.temp")).string()); //NOTE: temp output for as input to modify-cmvn-stats
		StrVec2Arg args(options);
		fs::path log(cmvndir / ("cmvn_" + name + ".log"));
		fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
		if (!file_log) LOGTW_WARNING << " log file is not accessible " << log.string() << ".";
		if (ComputeCmvnStats(args.argc(), args.argv(), file_log) < 0) {
			LOGTW_ERROR << "Error computing (partially fake) CMVN stats.";
			return -1;
		}
		file_log.flush(); file_log.close();
		//clear current options and replace with new
		options.clear();
		options.push_back("--print-args=false"); //NOTE: do not print arguments
		options.push_back(fake_dims);
		options.push_back("ark:" + (cmvndir / ("cmvn_" + name + ".ark.temp")).string());
		options.push_back("ark,scp:" + (cmvndir / ("cmvn_" + name + ".ark")).string() + "," + (cmvndir / ("cmvn_" + name + ".scp")).string());
		StrVec2Arg args1(options);
		fs::ofstream file_log1(log, fs::ofstream::binary | fs::ofstream::app);
		if (!file_log1) LOGTW_WARNING << " log file is not accessible " << log.string() << ".";
		if (ModifyCmvnStats(args1.argc(), args1.argv(), file_log1) < 0) {
			LOGTW_ERROR << "Error computing (partially fake) CMVN stats.";
			return -1;
		}
		try {
			fs::remove(cmvndir / ("cmvn_" + name + ".ark.temp"));
		}
		catch (const std::exception&) {}
	}
	else
	{
		string_vec options;
		options.push_back("--print-args=false"); //NOTE: do not print arguments
		options.push_back("--spk2utt=ark:" + (datadir / "spk2utt").string());
		options.push_back("scp:" + (datadir / "feats.scp").string());
		options.push_back("ark,scp:" + (cmvndir / ("cmvn_" + name + ".ark")).string() + "," + (cmvndir / ("cmvn_" + name + ".scp")).string());
		StrVec2Arg args(options);
		fs::path log(cmvndir / ("cmvn_" + name + ".log"));
		fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
		if (!file_log) LOGTW_WARNING << " log file is not accessible " << log.string() << ".";
		if (ComputeCmvnStats(args.argc(), args.argv(), file_log) < 0) {
			LOGTW_ERROR << "Error computing CMVN stats.";
			return -1;
		}
	}

	try	{
		fs::copy_file(cmvndir / ("cmvn_" + name + ".scp"), datadir / "cmvn.scp", fs::copy_option::overwrite_if_exists);
	} catch (const std::exception& ex) {
		LOGTW_ERROR << " " << ex.what() << ".";
		return -1;
	}

	StringTable tbl_cmvn;
	if (ReadStringTable((datadir / "cmvn.scp").string(), tbl_cmvn) < 0) return -1;	
	int nc = tbl_cmvn.size();
	int nu = tbl_spk2utt.size();
	if (nu != nc) {
		LOGTW_WARNING << "It seems not all of the speakers got cmvn stats (" << nc << " != " << nu << ").";
	}

	LOGTW_INFO << "Succeeded creating CMVN stats for " << name << ".";

	return 0;
}

