/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2013 Microsoft Corporation, Apache 2.0.
		   Johns Hopkins University (Author: Daniel Povey)
*/
#include "kaldi-win\scr\kaldi_scr.h"

/*
This creates its output in e.g. data/train/split50/{1,2,3,...50}, or if the per-utt option was given, 
in e.g. data/train/split50utt/{1,2,3,...50}. 
This will not split the data-dir if it detects that the output is newer than the input.
By default it splits per speaker (so each speaker is in only one split dir), but with the per-utt 
option it will ignore the speaker information while splitting.
*/
int SplitData(fs::path data,			//data directory
				int numsplit,			//
				bool per_utt)			//see description above
{
	bool split_per_spk = true;
	if (per_utt) split_per_spk = false;
	if (numsplit <= 0) {
		LOGTW_ERROR << "Invalid num-split argument " << numsplit << ".";
		return -1;
	}
	bool no_warn = false;
	if (!split_per_spk)
		//suppress warnings from filter_scps about 'some input lines were output to multiple files'.
		no_warn = true;

	int n = 0, nu=0, nf=0, nt=0;
	std::string feats("");
	std::string wavs("");
	std::vector<fs::path> utt2spks;
	std::string texts("");
	StringTable t_utt2spk, t_feats_scp, t_text;
	if (ReadStringTable((data / "utt2spk").string(), t_utt2spk) < 0) {
		LOGTW_ERROR << "Fail to open: " << (data / "utt2spk").string() << ".";
		return -1;
	}
	ReadStringTable((data / "feats.scp").string(), t_feats_scp); //may fail
	ReadStringTable((data / "text").string(), t_text); //may fail
	nu = t_utt2spk.size();
	nf = t_feats_scp.size();
	nt = t_text.size();
	if (fs::exists(data / "feats.scp") && nu != nf)
		LOGTW_WARNING << " #lines in utt2spk and feats.scp are " << nu << " <> " << nf << ".";
	if (fs::exists(data / "text") && nu != nt)
		LOGTW_WARNING << " #lines in utt2spk and text are " << nu << " <> " << nt << ".";
	std::string utt("");
	fs::path utt2spk_opt("");
	if (split_per_spk)
		utt2spk_opt = data / "utt2spk";
	else
		utt = "utt";
	//
	fs::path s1(data / ("split" + std::to_string(numsplit) + utt) / "1");
	bool need_to_split = false;
	if ( !(fs::exists(s1) && fs::is_directory(s1)) )
	{
		need_to_split = true;
	}
	else {
		need_to_split = false;
		string_vec _fn = {"utt2spk","spk2utt","spk2warp","feats.scp","text","wav.scp","cmvn.scp","spk2gender",
							"vad.scp","segments","reco2file_and_channel","utt2lang" };
		for each(std::string f in _fn) {
			if (fs::exists(data / f) && (!fs::exists(s1 / f) || fs::last_write_time(s1 / f) < fs::last_write_time(data / f)))
			{
				need_to_split = true;
			}
		}
	} ///else

	if (!need_to_split) return 0; //we do not need to split, exit

	LOGTW_INFO << "Need to split the data directory. Splitting...";

	for (int n = 1; n <= numsplit; n++) {
		utt2spks.push_back(data / ("split" + std::to_string(numsplit) + utt) / std::to_string(n) / "utt2spk");
		if (CreateDir(data / ("split" + std::to_string(numsplit) + utt) / std::to_string(n), true) < 0) return -1;
	}

	if (SplitScp(data / "utt2spk", utt2spks, 0, -1, utt2spk_opt) < 0) return -1;

	StringTable t_spk2uttn, t_utt2spkn;
	for (int n = 1; n <= numsplit; n++) {
		fs::path dsn(data / ("split" + std::to_string(numsplit) + utt) / std::to_string(n));		
		if (ReadStringTable((dsn / "utt2spk").string(), t_utt2spkn) < 0) return -1;
		utt2spk_to_spk2utt(t_spk2uttn, t_utt2spkn);
		if (SaveStringTable((dsn / "spk2utt").string(), t_spk2uttn) < 0) return -1;
	}

	std::string maybe_wav_scp("");
	if (!fs::exists(data / "segments"))
		maybe_wav_scp = "wav.scp"; //If there is no segments file, then wav file is indexed per utt.

	//split some things that are indexed by utterance.
	string_vec fs = { "feats.scp","text","vad.scp","utt2lang","utt2dur","utt2num_frames" };
	if (maybe_wav_scp != "") fs.push_back(maybe_wav_scp);
	for each(std::string f in fs) {
		if (fs::exists(data / f)) {
			if(FilterScps(1, numsplit,
				data / ("split"+ std::to_string(numsplit) + utt) / "JOBID" / "utt2spk",
				data / f,
				data / ("split" + std::to_string(numsplit) + utt) / "JOBID" / f
				) < 0 ) return -1;
		}
	}

	//split some things that are indexed by speaker.
	string_vec fs1 = { "spk2gender","spk2warp","cmvn.scp" };
	for each(std::string f in fs1) {
		if (fs::exists(data / f)) {
			if (FilterScps(1, numsplit,
				data / ("split" + std::to_string(numsplit) + utt) / "JOBID" / "spk2utt",
				data / f,
				data / ("split" + std::to_string(numsplit) + utt) / "JOBID" / f,
				no_warn
			) < 0) return -1;
		}
	}

	if (fs::exists(data / "segments"))
	{
		if (FilterScps(1, numsplit,
			data / ("split" + std::to_string(numsplit) + utt) / "JOBID" / "utt2spk",
			data / "segments",
			data / ("split" + std::to_string(numsplit) + utt) / "JOBID" / "segments"
		) < 0) return -1;
		StringTable t_dsnsegments;
		for (int n = 1; n <= numsplit; n++) {
			fs::path dsn(data / ("split" + std::to_string(numsplit) + utt) / std::to_string(n));
			//awk '{print $2;}' $dsn/segments | sort | uniq > $dsn/tmp.reco # recording-ids.
			if (ReadStringTable((dsn / "segments").string(), t_dsnsegments) < 0) return -1;
			StringTable t_tmpreco;			
			for (StringTable::const_iterator it(t_dsnsegments.begin()), it_end(t_dsnsegments.end()); it != it_end; ++it) {
				if ((*it).size() < 2) {
					LOGTW_ERROR << " expecting a second column in " << (dsn / "segments").string() << " but found only " << (*it).size() << ".";
					return -1;
				}
				std::vector<std::string> _v;
				_v.push_back((*it)[1]);
				t_tmpreco.push_back(_v);
			}
			if (SortStringTable(t_tmpreco, 0, 0, "string", "string", true) < 0) return -1; //sort on first column and make unique
			if (SaveStringTable((dsn / "tmp.reco").string(), t_tmpreco) < 0) return -1;
		}

		if (fs::exists(data / "reco2file_and_channel"))
		{
			if (FilterScps(1, numsplit,
				data / ("split" + std::to_string(numsplit) + utt) / "JOBID" / "tmp.reco",
				data / "reco2file_and_channel",
				data / ("split" + std::to_string(numsplit) + utt) / "JOBID" / "reco2file_and_channel",
				no_warn
			) < 0) return -1;
		}

		if (fs::exists(data / "wav.scp"))
		{
			if (FilterScps(1, numsplit,
				data / ("split" + std::to_string(numsplit) + utt) / "JOBID" / "tmp.reco",
				data / "wav.scp",
				data / ("split" + std::to_string(numsplit) + utt) / "JOBID" / "wav.scp",
				no_warn
			) < 0) return -1;
		}

		//clean up
		for (int n = 1; n <= numsplit; n++) {
			fs::path dsn(data / ("split" + std::to_string(numsplit) + utt) / std::to_string(n));
			try {
				fs::remove(dsn / "tmp.reco");
			} catch (const std::exception&){}
		}
	}

	return 0;
}
