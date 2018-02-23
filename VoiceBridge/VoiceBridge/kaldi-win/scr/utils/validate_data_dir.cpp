/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"
#include "kaldi-win/scr/Params.h"

/*
NOTE: The no_xxx options mean that the script does not require xxx.scp to be present, but it will check it if it is present.
no_spk_sort means that the script does not require the utt2spk to be sorted by the speaker-id in addition to being
sorted by utterance-id. By default, utt2spk is expected to be sorted by both, which can be achieved by making
the speaker-id prefixes of the utterance-ids

NOTE: file names in scp files may contain spaces. In this function we only use the first part of the scp files which is 
	  the utt_id, therefore ReadStringTable is ok to be used here (splits by space).
*/
int ValidateData(fs::path datadir,
	bool no_feats,	// = false
	bool no_wav,	// = false
	bool no_text,	// = false
	bool no_spk_sort	// = false
)
{

	fs::path spk2utt = datadir / "spk2utt";
	fs::path utt2spk = datadir / "utt2spk";
	if (CheckFileExistsAndNotEmpty(spk2utt, true) < 0) return -1;
	if (CheckFileExistsAndNotEmpty(utt2spk, true) < 0) return -1;
	StringTable table_;
	if (ReadStringTable(utt2spk.string(), table_) < 0) {
		LOGTW_ERROR << " fail to open: " << utt2spk.string() << ".";
		return -1;
	}
	for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it) {
		if ((*it).size() != 2) {
			LOGTW_ERROR << " data/utt2spk has wrong format. 2 columns are expected in " << utt2spk.string() << ".";
			return -1;
		}
	}
	if (ReadStringTable(spk2utt.string(), table_) < 0) {
		LOGTW_ERROR << "Fail to open: " << utt2spk.string() << ".";
		return -1;
	}
	if (table_.size() < 2 ) {
		LOGTW_WARNING << "You have only one speaker. This may cause several problems. " 
					  << "It is recommended to increase the number of speakers in " << utt2spk.string() << ".";
	}

	//create temporary directory
	fs::path tmpdir(datadir / ("kaldi_temp"));
	if (CreateDir(tmpdir) < 0) return -1;

	//utt2spk
	LOGTW_INFO << "  --> Validating utt2spk...";
	if (is_sortedonfield_and_uniqe(utt2spk) < 0) return -1;
	if (!no_spk_sort) {
		if (is_sortedonfield_and_uniqe(utt2spk, 1, false) < 0) {
			LOGTW_ERROR << " utt2spk is not in sorted order when sorted first on speaker-id " 
					  << "(fix this by making speaker-ids prefixes of utt-ids). File: " << utt2spk.string() << ".";
			return -1;
		}
	}

	//spk2utt
	LOGTW_INFO << "  --> Validating spk2utt..."; 
	fs::ofstream file_utts_temp(tmpdir / "utts", std::ios::binary);
	if (!file_utts_temp) {
		LOGTW_ERROR << "Can't open output file: " << (tmpdir / "utts").string() << ".";
		return -1;
	}
	if (is_sortedonfield_and_uniqe(spk2utt) < 0) return -1;
	StringTable table_utt2spk, table_spk2utt;
	if (ReadStringTable(utt2spk.string(), table_utt2spk) < 0) return -1;
	if (ReadStringTable(spk2utt.string(), table_spk2utt) < 0) return -1;
	table_.clear();
	spk2utt_to_utt2spk(table_spk2utt, table_);
	int indx = 0, col = 0;
	if (table_.size() != table_utt2spk.size()) {
		LOGTW_ERROR << " spk2utt and utt2spk do not seem to match.";
		return -1;
	}

	//sort table_ before comparing to table_spk2utt because it is sorted!
	if (SortStringTable(table_, 0, 0, "string", "string") < 0) return -1;

	for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it) {
		col = 0;
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) {
			if ((table_utt2spk[indx]).size()<(col+1) || table_utt2spk[indx][col] != *itc1) {
				LOGTW_ERROR << " spk2utt and utt2spk do not seem to match.";
				return -1;
			}
			col++;
		}
		//save utts to temp file
		file_utts_temp << (*it)[0] << "\n";
		indx++;
	}
	int num_utts = indx;
	file_utts_temp.flush(); file_utts_temp.close();

	//text
	StringTable table_text, table_text_utts, table_utts;
	LOGTW_INFO << "  --> Validating text...";
	fs::path data_txt = datadir / "text";
	if (CheckFileExistsAndNotEmpty(data_txt, false) < 0 && !no_text) return -1;
	if (fs::exists(data_txt)) {
		if (CheckUTF8AndWhiteSpace(data_txt, true) < 0) return -1;
		if (is_sortedonfield_and_uniqe(data_txt) < 0) return -1;		
		if (ReadStringTable(data_txt.string(), table_text) < 0) return -1;
		int text_len = table_text.size();
		std::vector<std::string> illegal_sym_list = { "<s>", "</s>", "#0" };
		fs::ofstream file_utts_txt_temp(tmpdir / "utts.txt", std::ios::binary);
		if (!file_utts_txt_temp) {
			LOGTW_ERROR << " can't open output file: " << (tmpdir / "utts.txt").string() << ".";
			return -1;
		}
		int k = 0;
		for each(std::string s in illegal_sym_list)
		{
			for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it) {
				for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) {
					if (*itc1 == s) {
						LOGTW_ERROR << " Error: in " << data_txt.filename() << ", text contains illegal symbol " << s << ".";
						return -1;
					}
				}
				//save first column to temp file because it is needed later; but only once!
				if(k==0) file_utts_txt_temp << (*it)[0] << "\n";
			}
			k++;
		}
		file_utts_txt_temp.flush(); file_utts_txt_temp.close();
		//StringTable table_text_utts, table_utts;
		if (ReadStringTable((tmpdir / "utts").string(), table_utts) < 0) return -1;
		if (ReadStringTable((tmpdir / "utts.txt").string(), table_text_utts) < 0) return -1;
		if (table_utts.size() != table_text_utts.size()) {
			LOGTW_ERROR << "Error in " << datadir.string() << ", utterance lists extracted from 'utt2spk' and 'text' "
					  << "differ. The lengths are " << table_utts.size() << " and " << table_text_utts.size() << ".";
			return -1;
		}
		indx = 0;
		for (StringTable::const_iterator it(table_utts.begin()), it_end(table_utts.end()); it != it_end; ++it) {
			col = 0;
			if ((table_text_utts[indx]).size()<(col + 1) || table_text_utts[indx][col] != (*it)[0]) {
				LOGTW_ERROR << "Error in " << datadir.string() << ", utterance lists extracted from 'utt2spk' and 'text' "
							<< "differ. Please check these files!";
				return -1;
			}
			indx++;
		}
	}

	//wav.scp
	LOGTW_INFO << "  --> Validating wav.scp...";
	fs::path path_wav_scp(datadir / "wav.scp");
	if (fs::exists(datadir / "segments") && !fs::exists(path_wav_scp))
	{
		LOGTW_ERROR << "Error in directory " << datadir.string() << ", 'segments' file exists but no 'wav.scp'.";
		return -1;
	}
	if (CheckFileExistsAndNotEmpty(path_wav_scp, false) < 0 && !no_wav) return -1;
	StringTable table_wav_scp;
	if (fs::exists(path_wav_scp))
	{
		if (is_sortedonfield_and_uniqe(path_wav_scp) < 0) return -1;		
		if (ReadStringTable(path_wav_scp.string(), table_wav_scp) < 0) return -1;
		if (ContainsString(table_wav_scp, "~")) {
			//NOTE: it's not a good idea to have any kind of tilde in wav.scp!
			LOGTW_ERROR << " Please do not use tilde (~) in your wav.scp.";
		}
		fs::path path_segments(datadir / "segments");
		StringTable table_segments;
		if (fs::exists(path_segments))
		{ // We have a segments file->interpret wav file as "recording-ids" not utterance - ids.	

			if (is_sortedonfield_and_uniqe(path_segments) < 0) return -1;
			if (ReadStringTable(path_segments.string(), table_segments) < 0) return -1;
			for (StringTable::const_iterator it(table_segments.begin()), it_end(table_segments.end()); it != it_end; ++it) {
				if ((*it).size() != 4 || ( StringToNumber<double>((*it)[3], -1) <= StringToNumber<double>((*it)[2]), -1) ) {
					LOGTW_ERROR << " Bad line in segments file.";
				}
			}
			int segments_len = table_segments.size();
			if (fs::exists(data_txt)) {
				//table_text, table_utts
				if (table_segments.size() != table_utts.size()) {
					LOGTW_ERROR << " Utterance list differs between data/text and data/segments. "
							  << "The lengths are " << table_segments.size() << " and " << table_utts.size() << ".";
					return -1;
				}
				indx = 0;
				for (StringTable::const_iterator it(table_text.begin()), it_end(table_text.end()); it != it_end; ++it) {
					col = 0;
					if ((table_utts[indx]).size() < (col + 1) || table_utts[indx][col] != (*it)[0]) {
						LOGTW_ERROR << " Utterance list differs between data/text and data/segments.";
						return -1;
					}
					indx++;
				}
			}

			//recordings
			fs::path path_recordings(tmpdir / "recordings");
			fs::path path_recordings_wav(tmpdir / "recordings_wav");
			fs::ofstream file_recordings(path_recordings, std::ios::binary);
			fs::ofstream file_recordings_wav(path_recordings_wav, std::ios::binary);
			if (!file_recordings) {
				LOGTW_ERROR << "Can't open output file: " << path_recordings.string() << ".";
				return -1;
			}
			if (!file_recordings_wav) {
				LOGTW_ERROR << "Can't open output file: " << path_recordings_wav.string() << ".";
				return -1;
			}
			std::vector<std::string> _v;
			for (StringTable::const_iterator it(table_segments.begin()), it_end(table_segments.end()); it != it_end; ++it) {
				_v.push_back((*it)[1]);
			}
			std::sort(_v.begin(), _v.end());
			_v.erase(std::unique(_v.begin(), _v.end()), _v.end());
			for each(std::string s in _v)
				file_recordings << s << "\n";
			file_recordings.flush(); file_recordings.close();
			for (StringTable::const_iterator it(table_wav_scp.begin()), it_end(table_wav_scp.end()); it != it_end; ++it) {
				file_recordings_wav << (*it)[0] << "\n";
			}
			file_recordings_wav.flush(); file_recordings_wav.close();
			//compare recordings/recordings_wav
			StringTable table_recordings, table_recordings_wav;
			if (ReadStringTable(path_recordings.string(), table_recordings) < 0) return -1;
			if (ReadStringTable(path_recordings_wav.string(), table_recordings_wav) < 0) return -1;
			if (table_recordings.size() != table_recordings_wav.size()) {
				LOGTW_ERROR << " in " << datadir.string() <<", recording-ids extracted from segments and wav.scp differ.";
				return -1;
			}
			for (StringTable::const_iterator it(table_recordings.begin()), it_end(table_recordings.end()); it != it_end; ++it) {
				col = 0;
				if ((table_recordings_wav[indx]).size()<(col + 1) || table_recordings_wav[indx][col] != (*it)[0]) {
					LOGTW_ERROR << " in " << datadir.string() << ", recording-ids extracted from segments and wav.scp differ.";
					return -1;
				}
				indx++;
			}

			//reco2file_and_channel
			fs::path path_reco2file_and_channel(datadir / "reco2file_and_channel");
			StringTable table_reco2;
			if (fs::exists(path_reco2file_and_channel))
			{//this file is needed only for ctm scoring; it's indexed by recording-id.
				if (is_sortedonfield_and_uniqe(path_reco2file_and_channel) < 0) return -1;
				if (ReadStringTable(path_reco2file_and_channel.string(), table_reco2) < 0) return -1;
				bool warning_issued = false;
				for (StringTable::const_iterator it(table_reco2.begin()), it_end(table_reco2.end()); it != it_end; ++it) {
					int NF = (*it).size();
					if (NF != 3 || ((*it)[2] != "A" && (*it)[2] != "B")) {
						if (NF == 3 && (*it)[2] == "1") {
							warning_issued = true;
						}
						else {
							LOGTW_ERROR << " bad line in " << path_reco2file_and_channel.string() << ".";
							return -1;
						}
					}
				}
				if (warning_issued)
					LOGTW_INFO << " The channel should be marked as A or B, not 1! You should change it ASAP!";

				//tmpdir/recordings_r2fc
				fs::path path_recordings_r2fc(tmpdir / "recordings_r2fc");
				fs::ofstream file_recordings_r2fc(path_recordings_r2fc, std::ios::binary);
				if (!file_recordings_r2fc) {
					LOGTW_ERROR << "Can't open output file: " << path_recordings_r2fc.string() << ".";
					return -1;
				}
				for (StringTable::const_iterator it(table_reco2.begin()), it_end(table_reco2.end()); it != it_end; ++it) {
					file_recordings_r2fc << (*it)[0] << "\n";
				}
				file_recordings_r2fc.flush(); file_recordings_r2fc.close();
				StringTable table_recordings_r2fc;
				if (ReadStringTable(path_recordings_r2fc.string(), table_recordings_r2fc) < 0) return -1;
				indx = 0;
				if (table_recordings.size() != table_recordings_r2fc.size()) {
					LOGTW_ERROR << " in " << datadir.string() << ", recording-ids extracted from segments and reco2file_and_channel.";
					return -1;
				}
				for (StringTable::const_iterator it(table_recordings.begin()), it_end(table_recordings.end()); it != it_end; ++it) {
					col = 0;
					if ((table_recordings_r2fc[indx]).size()<(col + 1) || table_recordings_r2fc[indx][col] != (*it)[0]) {
						LOGTW_ERROR << " in " << datadir.string() << ", recording-ids extracted from segments and reco2file_and_channel.";
						return -1;
					}
					indx++;
				}
			} ///if (fs::exists(path_reco2file_and_channel
		} ///if (fs::exists(path_segments
		else 
		{
			fs::path path_utts_wav(tmpdir / "utts_wav");
			fs::ofstream file_utts_wav(path_utts_wav, std::ios::binary);
			for (StringTable::const_iterator it(table_wav_scp.begin()), it_end(table_wav_scp.end()); it != it_end; ++it) {
				file_utts_wav << (*it)[0] << "\n";
			}
			file_utts_wav.flush(); file_utts_wav.close();
			StringTable table_utts_wav;
			if (ReadStringTable(path_utts_wav.string(), table_utts_wav) < 0) return -1;
			//compare tmpdir/utts and tmpdir/utts_wav}
			if (table_utts_wav.size() != table_utts.size()) {
				LOGTW_ERROR << " utterance lists extracted from utt2spk and wav.scp differ. "
					<< "The lengths are " << table_utts_wav.size() << " and " << table_utts.size() << ".";
				return -1;
			}
			indx = 0;
			for (StringTable::const_iterator it(table_utts_wav.begin()), it_end(table_utts_wav.end()); it != it_end; ++it) {
				col = 0;
				if ((table_utts[indx]).size() < (col + 1) || table_utts[indx][col] != (*it)[0]) {
					LOGTW_ERROR << " utterance lists extracted from utt2spk and wav.scp differ.";
					return -1;
				}
				indx++;
			}

			//reco2file_and_channel
			fs::path path_reco2file_and_channel(datadir / "reco2file_and_channel");
			StringTable table_reco2;
			if (fs::exists(path_reco2file_and_channel))
			{//this file is needed only for ctm scoring; it's indexed by recording-id.
				if (is_sortedonfield_and_uniqe(path_reco2file_and_channel) < 0) return -1;
				if (ReadStringTable(path_reco2file_and_channel.string(), table_reco2) < 0) return -1;
				bool warning_issued = false;
				for (StringTable::const_iterator it(table_reco2.begin()), it_end(table_reco2.end()); it != it_end; ++it) {
					int NF = (*it).size();
					if (NF != 3 || ((*it)[2] != "A" && (*it)[2] != "B")) {
						if (NF == 3 && (*it)[2] == "1") {
							warning_issued = true;
						}
						else {
							LOGTW_ERROR << " bad line in " << path_reco2file_and_channel.string() << ".";
							return -1;
						}
					}
				}
				if (warning_issued)
					LOGTW_INFO << " The channel should be marked as A or B, not 1! You should change it ASAP!";

				//tmpdir/utts_r2fc
				fs::path path_utts_r2fc(tmpdir / "utts_r2fc");
				fs::ofstream file_utts_r2fc(path_utts_r2fc, std::ios::binary);
				if (!file_utts_r2fc) {
					LOGTW_ERROR << " can't open output file: " << path_utts_r2fc.string() << ".";
					return -1;
				}
				for (StringTable::const_iterator it(table_reco2.begin()), it_end(table_reco2.end()); it != it_end; ++it) {
					file_utts_r2fc << (*it)[0] << "\n";
				}
				file_utts_r2fc.flush(); file_utts_r2fc.close();
				StringTable table_utts_r2fc;
				if (ReadStringTable(path_utts_r2fc.string(), table_utts_r2fc) < 0) return -1;
				indx = 0;
				if (table_utts.size() != table_utts_r2fc.size()) {
					LOGTW_ERROR << " in " << datadir.string() << ", utterance-ids extracted from segments and reco2file_and_channel differ.";
					return -1;
				}
				for (StringTable::const_iterator it(table_utts.begin()), it_end(table_utts.end()); it != it_end; ++it) {
					col = 0;
					if ((table_utts_r2fc[indx]).size()<(col + 1) || table_utts_r2fc[indx][col] != (*it)[0]) {
						LOGTW_ERROR << " in " << datadir.string() << ", utterance-ids extracted from segments and reco2file_and_channel differ.";
						return -1;
					}
					indx++;
				}
			} ///if (fs::exists(path_reco2file_and_channel
		}
	}

	//feats.scp
	fs::path path_feats_scp(datadir / "feats.scp");
	if (!fs::exists(path_feats_scp) && !no_feats)
	{
		LOGTW_ERROR << "No such file data/feats.scp (if this is by design, specify 'no_feats').";
		return -1;
	}
	if (fs::exists(path_feats_scp))
	{
		if (is_sortedonfield_and_uniqe(path_feats_scp) < 0) return -1;
		StringTable table_feats_scp;
		if (ReadStringTable(path_feats_scp.string(), table_feats_scp) < 0) return -1;

		fs::path path_utts_feats(tmpdir / "utts.feats");
		fs::ofstream file_utts_feats(path_utts_feats, std::ios::binary);
		if (!file_utts_feats) {
			LOGTW_ERROR << "Can't open output file: " << path_utts_feats.string() << ".";
			return -1;
		}
		for (StringTable::const_iterator it(table_feats_scp.begin()), it_end(table_feats_scp.end()); it != it_end; ++it) {
			file_utts_feats << (*it)[0] << "\n";
		}
		file_utts_feats.flush(); file_utts_feats.close();
		StringTable table_utts_feats;
		if (ReadStringTable(path_utts_feats.string(), table_utts_feats) < 0) return -1;
		indx = 0;
		if (table_utts.size() != table_utts_feats.size()) {
			LOGTW_ERROR << " in " << datadir.string() << ", utterance-ids extracted from utt2spk and features differ.";
			return -1;
		}
		for (StringTable::const_iterator it(table_utts.begin()), it_end(table_utts.end()); it != it_end; ++it) {
			col = 0;
			if ((table_utts_feats[indx]).size()<(col + 1) || table_utts_feats[indx][col] != (*it)[0]) {
				LOGTW_ERROR << " in " << datadir.string() << ", utterance-ids extracted from utt2spk and features differ.";
				return -1;
			}
			indx++;
		}
	}

	//cmvn.scp
	fs::path path_cmvn_scp(datadir / "cmvn.scp");
	if (fs::exists(path_cmvn_scp))
	{
		if (is_sortedonfield_and_uniqe(path_cmvn_scp) < 0) return -1;
		StringTable table_cmvn_scp;
		if (ReadStringTable(path_cmvn_scp.string(), table_cmvn_scp) < 0) return -1;
		//Compare the first field of table_spk2utt, table_cmvn_scp
		//NOTE: not saving the files just comparing in memory
		if (table_spk2utt.size() != table_cmvn_scp.size()) {
			LOGTW_ERROR << " in " << datadir.string() << ", speaker lists extracted from spk2utt and cmvn differ. "
					  << "The lengths are " << table_cmvn_scp.size() << " and " << table_spk2utt.size() << ".";
			return -1;
		}
		indx = 0;
		for (StringTable::const_iterator it(table_spk2utt.begin()), it_end(table_spk2utt.end()); it != it_end; ++it) {
			if ((table_cmvn_scp[indx]).size()<1 || table_cmvn_scp[indx][0] != (*it)[0]) {
				LOGTW_ERROR << " in " << datadir.string() << ", speaker lists extracted from spk2utt and cmvn differ.";
				return -1;
			}
			indx++;
		}
	}

	//spk2gender
	fs::path path_spk2gender(datadir / "spk2gender");
	if (fs::exists(path_spk2gender))
	{
		if (is_sortedonfield_and_uniqe(path_spk2gender) < 0) return -1;
		StringTable table_spk2gender;
		if (ReadStringTable(path_spk2gender.string(), table_spk2gender) < 0) return -1;
		for (StringTable::const_iterator it(table_spk2gender.begin()), it_end(table_spk2gender.end()); it != it_end; ++it) {
			int NF = (*it).size();
			if (!((NF == 2 && ((*it)[1] == "m" || (*it)[1] == "f")))) {
				LOGTW_ERROR << " Mal-formed spk2gender file " << path_spk2gender.string() << ".";
				return -1;
			}
		}
		if (table_spk2utt.size() != table_spk2gender.size()) {
			LOGTW_ERROR << " in " << datadir.string() << ", speaker lists extracted from spk2utt and spk2gender differ. "
				<< "The lengths are " << table_spk2gender.size() << " and " << table_spk2utt.size() << ".";
			return -1;
		}
		indx = 0;
		for (StringTable::const_iterator it(table_spk2utt.begin()), it_end(table_spk2utt.end()); it != it_end; ++it) {
			if ((table_spk2gender[indx]).size()<1 || table_spk2gender[indx][0] != (*it)[0]) {
				LOGTW_ERROR << " in " << datadir.string() << ", speaker lists extracted from spk2utt and spk2gender differ.";
				return -1;
			}
			indx++;
		}
	}

	//spk2warp
	fs::path path_spk2warp(datadir / "spk2warp");
	if (fs::exists(path_spk2warp))
	{
		if (is_sortedonfield_and_uniqe(path_spk2warp) < 0) return -1;
		StringTable table_spk2warp;
		if (ReadStringTable(path_spk2warp.string(), table_spk2warp) < 0) return -1;
		for (StringTable::const_iterator it(table_spk2warp.begin()), it_end(table_spk2warp.end()); it != it_end; ++it) {
			int NF = (*it).size();
			double d = StringToNumber<double>((*it)[1], -1);
			if (!((NF == 2 && (d > 0.5 && d < 1.5)))) {
				LOGTW_ERROR << " Mal-formed spk2warp file " << path_spk2warp.string() << ".";
				return -1;
			}
		}
		if (table_spk2utt.size() != table_spk2warp.size()) {
			LOGTW_ERROR << " in " << datadir.string() << ", speaker lists extracted from spk2utt and spk2warp differ. "
				<< "The lengths are " << table_spk2warp.size() << " and " << table_spk2utt.size() << ".";
			return -1;
		}
		indx = 0;
		for (StringTable::const_iterator it(table_spk2utt.begin()), it_end(table_spk2utt.end()); it != it_end; ++it) {
			if ((table_spk2warp[indx]).size()<1 || table_spk2warp[indx][0] != (*it)[0]) {
				LOGTW_ERROR << " in " << datadir.string() << ", speaker lists extracted from spk2utt and spk2warp differ.";
				return -1;
			}
			indx++;
		}
	}

	//utt2warp
	fs::path path_utt2warp(datadir / "utt2warp");
	if (fs::exists(path_utt2warp))
	{
		if (is_sortedonfield_and_uniqe(path_utt2warp) < 0) return -1;
		StringTable table_utt2warp;
		if (ReadStringTable(path_utt2warp.string(), table_utt2warp) < 0) return -1;
		for (StringTable::const_iterator it(table_utt2warp.begin()), it_end(table_utt2warp.end()); it != it_end; ++it) {
			int NF = (*it).size();
			double d = StringToNumber<double>((*it)[1], -1);
			if (!((NF == 2 && (d > 0.5 && d < 1.5)))) {
				LOGTW_ERROR << " Mal-formed utt2warp file " << path_utt2warp.string() << ".";
				return -1;
			}
		}
		if (table_spk2utt.size() != table_utt2warp.size()) {
			LOGTW_ERROR << " in " << datadir.string() << ", utterance lists extracted from utt2spk and utt2warp differ. "
					  << "The lengths are " << table_utt2warp.size() << " and " << table_spk2utt.size() << ".";
			return -1;
		}
		indx = 0;
		for (StringTable::const_iterator it(table_spk2utt.begin()), it_end(table_spk2utt.end()); it != it_end; ++it) {
			if ((table_utt2warp[indx]).size()<1 || table_utt2warp[indx][0] != (*it)[0]) {
				LOGTW_ERROR << " in " << datadir.string() << ", utterance lists extracted from utt2spk and utt2warp differ.";
				return -1;
			}
			indx++;
		}
	}

	//vad.scp
	fs::path path_vad_scp(datadir / "vad.scp");
	if (fs::exists(path_vad_scp))
	{
		if (is_sortedonfield_and_uniqe(path_vad_scp) < 0) return -1;
		StringTable table_vad_scp;
		if (ReadStringTable(path_vad_scp.string(), table_vad_scp) < 0) return -1;
		if (table_spk2utt.size() != table_vad_scp.size()) {
			LOGTW_ERROR << " in " << datadir.string() << ", vad.scp and utt2spk do not have identical utterance-id list. "
					  << "The lengths are " << table_vad_scp.size() << " and " << table_spk2utt.size() << ".";
			return -1;
		}
		indx = 0;
		for (StringTable::const_iterator it(table_spk2utt.begin()), it_end(table_spk2utt.end()); it != it_end; ++it) {
			if ((table_vad_scp[indx]).size()<1 || table_vad_scp[indx][0] != (*it)[0]) {
				LOGTW_ERROR << " in " << datadir.string() << ", vad.scp and utt2spk do not have identical utterance-id list.";
				return -1;
			}
			indx++;
		}
	}

	//utt2lang
	fs::path path_utt2lang(datadir / "utt2lang");
	if (fs::exists(path_utt2lang))
	{
		if (is_sortedonfield_and_uniqe(path_utt2lang) < 0) return -1;
		StringTable table_utt2lang;
		if (ReadStringTable(path_utt2lang.string(), table_utt2lang) < 0) return -1;
		if (table_spk2utt.size() != table_utt2lang.size()) {
			LOGTW_ERROR << " in " << datadir.string() << ", utt2lang and utt2spk do not have identical utterance-id list. "
					  << "The lengths are " << table_utt2lang.size() << " and " << table_spk2utt.size() << ".";
			return -1;
		}
		indx = 0;
		for (StringTable::const_iterator it(table_spk2utt.begin()), it_end(table_spk2utt.end()); it != it_end; ++it) {
			if ((table_utt2lang[indx]).size()<1 || table_utt2lang[indx][0] != (*it)[0]) {
				LOGTW_ERROR << " in " << datadir.string() << ", utt2lang and utt2spk do not have identical utterance-id list.";
				return -1;
			}
			indx++;
		}
	}

	//utt2uniq
	fs::path path_utt2uniq(datadir / "utt2uniq");
	if (fs::exists(path_utt2uniq))
	{
		if (is_sortedonfield_and_uniqe(path_utt2uniq) < 0) return -1;
		StringTable table_utt2uniq;
		if (ReadStringTable(path_utt2uniq.string(), table_utt2uniq) < 0) return -1;
		if (table_spk2utt.size() != table_utt2uniq.size()) {
			LOGTW_ERROR << " in " << datadir.string() << ", utt2uniq and utt2spk do not have identical utterance-id list. "
					  << "The lengths are " << table_utt2uniq.size() << " and " << table_spk2utt.size() << ".";
			return -1;
		}
		indx = 0;
		for (StringTable::const_iterator it(table_spk2utt.begin()), it_end(table_spk2utt.end()); it != it_end; ++it) {
			if ((table_utt2uniq[indx]).size()<1 || table_utt2uniq[indx][0] != (*it)[0]) {
				LOGTW_ERROR << " in " << datadir.string() << ", utt2uniq and utt2spk do not have identical utterance-id list.";
				return -1;
			}
			indx++;
		}
	}

	//utt2dur
	fs::path path_utt2dur(datadir / "utt2dur");
	if (fs::exists(path_utt2dur))
	{
		if (is_sortedonfield_and_uniqe(path_utt2dur) < 0) return -1;
		StringTable table_utt2dur;
		if (ReadStringTable(path_utt2dur.string(), table_utt2dur) < 0) return -1;
		if (table_spk2utt.size() != table_utt2dur.size()) {
			LOGTW_ERROR << " in " << datadir.string() << ", utterance-ids extracted from utt2spk and utt2dur file differ. "
				<< "The lengths are " << table_utt2dur.size() << " and " << table_spk2utt.size() << ".";
			return -1;
		}
		indx = 0;
		for (StringTable::const_iterator it(table_spk2utt.begin()), it_end(table_spk2utt.end()); it != it_end; ++it) {
			if ((table_utt2dur[indx]).size()<1 || table_utt2dur[indx][0] != (*it)[0]) {
				LOGTW_ERROR << " in " << datadir.string() << ", utterance-ids extracted from utt2spk and utt2dur file differ.";
				return -1;
			}
			indx++;
		}
		for (StringTable::const_iterator it(table_utt2dur.begin()), it_end(table_utt2dur.end()); it != it_end; ++it) {
			int NF = (*it).size();
			double d = StringToNumber<double>((*it)[1], -1);
			if (NF != 2 || !(d > 0)) {
				LOGTW_ERROR << " Mal-formed utt2dur file " << path_utt2dur.string() << ".";
				return -1;
			}
		}
	}

	LOGTW_INFO << "Successfully validated data-directory " << datadir.string() << ".";
	
	try	{
		fs::remove_all(tmpdir);
	} catch (const std::exception& ex) {
		LOGTW_WARNING << " Could not remove temporary directory. Reason: " << ex.what() << ".";
	}

	return 0;
}

