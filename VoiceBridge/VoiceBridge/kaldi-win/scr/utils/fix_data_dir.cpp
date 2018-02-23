/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Kaldi fix_data_dir.sh
*/

#include "kaldi-win\scr\kaldi_scr.h"

int filter_file(fs::path filter, fs::path file_to_filter);
int filter_recordings(fs::path datadir, fs::path tmpdir);
int filter_speakers(fs::path datadir, fs::path tmpdir, std::vector<fs::path> spk_extra_files = {});
int filter_utts(fs::path datadir, fs::path tmpdir, std::vector<fs::path> utt_extra_files = {});

/*
This function makes sure that only the segments present in all of "feats.scp", "wav.scp" [if present], segments [if present]
text, and utt2spk are present in any of them. It puts the original contents of data-dir into data-dir/.backup
*/
VOICEBRIDGE_API int FixDataDir(fs::path datadir, std::vector<fs::path> spk_extra_files, std::vector<fs::path> utt_extra_files)
{
	fs::path spk2utt = datadir / "spk2utt";
	fs::path utt2spk = datadir / "utt2spk";
	if (CheckFileExistsAndNotEmpty(spk2utt, true) < 0) return -1;
	if (CheckFileExistsAndNotEmpty(utt2spk, true) < 0) return -1;

	//create backup directory
	fs::path backupdir(datadir / ("backup"));
	if (CreateDir(backupdir) < 0) return -1;
	//create temporary directory
	fs::path tmpdir(datadir / ("kaldi_temp"));
	if (CreateDir(tmpdir) < 0) return -1;

	std::vector<std::string> _data = { "utt2spk", "spk2utt", "feats.scp", "text", "segments", "wav.scp", "cmvn.scp", 
	  "vad.scp", "reco2file_and_channel", "spk2gender", "utt2lang", "utt2uniq", "utt2dur", "utt2num_frames" };
	for each(std::string f in _data) {
		if (fs::exists(datadir / f)) {
			//backup first
			fs::copy_file(datadir / f, backupdir / f, fs::copy_option::overwrite_if_exists);
			int ret = is_sortedonfield_and_uniqe(datadir / f, 0, true, false);
			if (ret < -2) return -1; //fatal error
			if (ret < 0) 
			{
				//sort and make unique if it is not
				LOGTW_INFO << " file " << f << " is not in sorted order or not unique, fixing file...";
				StringTable table_;
				if (ReadStringTable((datadir / f).string(), table_) < 0) return -1;
				SortStringTable(table_, 0, 0, "string", "string", false);
				//save the file, overwrite existing (it is backed up before)
				if (SaveStringTable((datadir / f).string(), table_) < 0) return -1;
			}
		}
	}

	if (filter_recordings(datadir, tmpdir) < 0) return -1;
	if (filter_speakers(datadir, tmpdir, spk_extra_files) < 0) return -1;
	if (filter_utts(datadir, tmpdir, utt_extra_files) < 0) return -1;
	if (filter_speakers(datadir, tmpdir, spk_extra_files) < 0) return -1;
	if (filter_recordings(datadir, tmpdir) < 0) return -1;

	//convert the fixed utt2spk to spk2utt
	StringTable table_spk2utt, table_utt2spk;
	if (ReadStringTable((datadir / "utt2spk").string(), table_utt2spk) < 0) return -1;
	utt2spk_to_spk2utt(table_spk2utt, table_utt2spk);
	if (SaveStringTable((datadir / "spk2utt").string(), table_spk2utt) < 0) return -1;

	LOGTW_INFO << "Successfully fixed data-directory " << datadir.string() << ". Old files are kept in " << backupdir.string() << ".";

	try {
		fs::remove_all(tmpdir);
	}
	catch (const std::exception& ex) {
		LOGTW_WARNING << " Could not remove temporary directory. Reason: " << ex.what() << ".";
	}

	return 0;
}


int filter_file(fs::path filter, fs::path file_to_filter) 
{
	try	{
		fs::copy_file(file_to_filter, (file_to_filter.string() + ".tmp"), fs::copy_option::overwrite_if_exists);
	} catch (const std::exception& ex)	{
		LOGTW_ERROR << " Could not copy file. Reason: " << ex.what() << ".";
		return -1;
	}
	//
	if (FilterScp(filter, (file_to_filter.string() + ".tmp"), file_to_filter) < 0) {
		LOGTW_ERROR << " Failed to filter file " << filter.string() << ".";
		return -1;
	}
	//compare the two in order to see what is changed
	StringTable table_tmp, table_;
	if (ReadStringTable((file_to_filter.string() + ".tmp"), table_tmp) < 0) return -1;
	if (ReadStringTable(file_to_filter.string(), table_) < 0) return -1;
	if (table_tmp.size() != table_.size()) {
		LOGTW_INFO << " filtered " << file_to_filter.string() << " from " << table_tmp.size() << " to " 
			<< table_.size() << " lines based on filter " << filter.string() << ".";
	}

	try {
		fs::remove(file_to_filter.string() + ".tmp");
	}
	catch (const std::exception& ex) {
		LOGTW_WARNING << " Could not remove temporary file. Reason: " << ex.what() << ".";
	}
	return 0;
}

int filter_recordings(fs::path datadir, fs::path tmpdir)
{//We call this once before the stage when we filter on utterance-id, and once after.	
	fs::path path_segments(datadir / "segments");
	if (fs::exists(path_segments))
	{
		//We have a segments file -> we need to filter this and the file wav.scp, and reco2file_and_utt, 
		//if it exists, to make sure they have the same list of recording - ids.
		StringTable table_segments;
		if (ReadStringTable(path_segments.string(), table_segments) < 0) return -1;
		//
		fs::path path_wav_scp(datadir / "wav.scp");
		if (!fs::exists(path_wav_scp))
		{
			LOGTW_ERROR << "Error in directory " << datadir.string() << ", 'segments' file exists but no 'wav.scp'.";
			return -1;
		}
		if (CheckFileExistsAndNotEmpty(path_wav_scp, true) < 0) return -1;
		//
		fs::path path_recordings(tmpdir / "recordings");
		fs::ofstream file_recordings(path_recordings, std::ios::binary);
		if (!file_recordings) {
			LOGTW_ERROR << "Can't open output file: " << path_recordings.string() << ".";
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
		if (CheckFileExistsAndNotEmpty(path_recordings, true) < 0) return -1;
		//
		if (FilterScp(path_wav_scp, path_recordings, tmpdir / "recordings.tmp") < 0) {
			LOGTW_ERROR << "Failed to filter file " << path_wav_scp.string() << ".";
			return -1;
		}
		try {
			fs::copy_file(tmpdir / "recordings.tmp", path_recordings, fs::copy_option::overwrite_if_exists);
			fs::remove(tmpdir / "recordings.tmp");
			fs::copy_file(path_segments, datadir / "segments.tmp", fs::copy_option::overwrite_if_exists);
			fs::ofstream file_segments(path_segments, std::ios::binary | std::ios::out);
			if (!file_segments) {
				LOGTW_ERROR << "Can't open output file: " << path_segments.string() << ".";
				return -1;
			}
			for (StringTable::const_iterator it(table_segments.begin()), it_end(table_segments.end()); it != it_end; ++it) {
				file_segments << (*it)[1] << " " << (*it)[0] << " " << (*it)[2] << " " << (*it)[3] << "\n";
			}
			file_segments.flush(); file_segments.close();
			if (filter_file(path_recordings, path_segments)<0) return -1;
			fs::copy_file(path_segments, datadir / "segments.tmp", fs::copy_option::overwrite_if_exists);
			if (ReadStringTable((datadir / "segments.tmp").string(), table_segments) < 0) return -1;
			fs::ofstream file_segments2(path_segments, std::ios::binary | std::ios::out);
			if (!file_segments2) {
				LOGTW_ERROR << "Can't open output file: " << path_segments.string() << ".";
				return -1;
			}
			for (StringTable::const_iterator it(table_segments.begin()), it_end(table_segments.end()); it != it_end; ++it) {
				file_segments2 << (*it)[1] << " " << (*it)[0] << " " << (*it)[2] << " " << (*it)[3] << "\n";
			}
			file_segments2.flush(); file_segments2.close();
			fs::remove(datadir / "segments.tmp");
		}
		catch (const std::exception& ex) {
			LOGTW_ERROR << " " << ex.what() << ".";
			return -1;
		}

		if (filter_file(path_recordings, path_wav_scp)<0) return -1;
		if (fs::exists(datadir / "reco2file_and_channel"))
		{
			if (filter_file(path_recordings, datadir / "reco2file_and_channel")<0) return -1;
		}
	}

	return 0;
}

int filter_speakers(fs::path datadir, fs::path tmpdir, std::vector<fs::path> spk_extra_files)
{
	// throughout this program, we regard utt2spk as primary and spk2utt as derived, so...
	StringTable table_spk2utt, table_utt2spk;
	if (ReadStringTable((datadir / "utt2spk").string(), table_utt2spk) < 0) return -1;
	utt2spk_to_spk2utt(table_spk2utt, table_utt2spk);
	if (SaveStringTable((datadir / "spk2utt").string(), table_spk2utt) < 0) return -1;
	//
	fs::ofstream file_speakers(tmpdir / "speakers", std::ios::binary);
	if (!file_speakers) {
		LOGTW_ERROR << "Can't open output file: " << (tmpdir / "speakers").string() << ".";
		return -1;
	}
	for (StringTable::const_iterator it(table_spk2utt.begin()), it_end(table_spk2utt.end()); it != it_end; ++it) 
		file_speakers << (*it)[0] << "\n";
	file_speakers.flush(); file_speakers.close();
	//filter cmvn.scp
	if (fs::exists(datadir / "cmvn.scp"))
		if (filter_file(datadir / "cmvn.scp", tmpdir / "speakers")<0) return -1;

	//filter spk2gender
	if (fs::exists(datadir / "spk2gender"))
		if (filter_file(datadir / "spk2gender", tmpdir / "speakers")<0) return -1;

	if (filter_file(tmpdir / "speakers", datadir / "spk2utt")<0) return -1;
	if (ReadStringTable((datadir / "spk2utt").string(), table_spk2utt) < 0) return -1;
	spk2utt_to_utt2spk(table_spk2utt, table_utt2spk);

	//filter cmvn.scp
	if (fs::exists(datadir / "cmvn.scp"))
		if (filter_file(tmpdir / "speakers", datadir / "cmvn.scp")<0) return -1;

	//filter spk2gender
	if (fs::exists(datadir / "spk2gender"))
		if (filter_file(tmpdir / "speakers", datadir / "spk2gender")<0) return -1;

	//filter spk_extra_files
	for each(fs::path p in spk_extra_files) {
		if (fs::exists(p))
			if (filter_file(tmpdir / "speakers", p) < 0) return -1;
	}

	return 0;
}

int filter_utts(fs::path datadir, fs::path tmpdir, std::vector<fs::path> utt_extra_files) 
{

	StringTable table_spk2utt, table_utt2spk;
	if (ReadStringTable((datadir / "utt2spk").string(), table_utt2spk) < 0) return -1;
	if (ReadStringTable((datadir / "spk2utt").string(), table_spk2utt) < 0) return -1;
	//
	fs::ofstream file_utts_tmp(tmpdir / "utts", std::ios::binary | std::ios::out);
	if (!file_utts_tmp) {
		LOGTW_ERROR << "Can't open output file: " << (tmpdir / "utts").string() << ".";
		return -1;
	}
	for (StringTable::const_iterator it(table_utt2spk.begin()), it_end(table_utt2spk.end()); it != it_end; ++it) {
		file_utts_tmp << (*it)[0] << "\n";
	}
	file_utts_tmp.flush(); file_utts_tmp.close();
	//
	if (is_sortedonfield_and_uniqe(datadir / "utt2spk", 0, false, true) < 0) {
		LOGTW_ERROR << " utt2spk is not in sorted order (fix this yourself).";
		return -1;
	}
	if (is_sortedonfield_and_uniqe(datadir / "utt2spk", 1, false, true) < 0) {
		LOGTW_ERROR << " utt2spk is not in sorted order when sorted first on speaker-id (fix this by making speaker-ids prefixes of utt-ids).";
		return -1;
	}
	//
	if (is_sortedonfield_and_uniqe(datadir / "spk2utt", 0, false, true) < 0) {
		LOGTW_ERROR << " spk2utt is not in sorted order (fix this yourself).";
		return -1;
	}
	//
	if (fs::exists(datadir / "utt2uniq"))
	{
		if (is_sortedonfield_and_uniqe(datadir / "utt2uniq", 0, false, true) < 0) {
			LOGTW_ERROR << " utt2uniq is not in sorted order (fix this yourself).";
			return -1;
		}
	}
	//
	std::string maybe_wav("");
	if (!fs::exists(datadir / "segments")) {
		maybe_wav = "wav.scp"; // wav indexed by utts only if segments does not exist.
	}
	//feats.scp text segments utt2lang "maybe_wav"
	string_vec files = {"feats.scp","text","segments","utt2lang"};
	if (maybe_wav != "") files.push_back(maybe_wav);
	for each(std::string f in files) {
		if (fs::exists(datadir / f)) {
			if (FilterScp(datadir / f, (tmpdir / "utts").string(), (tmpdir / "utts").string() + ".tmp") < 0) {
				LOGTW_ERROR << " Failed to filter file " << (tmpdir / "utts").string() << ".";
				return -1;
			}
			try {
				fs::copy_file((tmpdir / "utts").string() + ".tmp", (tmpdir / "utts").string(), fs::copy_option::overwrite_if_exists);
				fs::remove((tmpdir / "utts").string() + ".tmp");
			}
			catch (const std::exception& ex) {
				LOGTW_ERROR << " Could not move file. Reason: " << ex.what() << ".";
				return -1;
			}
		}
	}

	if (!fs::exists(tmpdir / "utts") || fs::is_empty(tmpdir / "utts"))
	{
		LOGTW_ERROR << "No utterances remained: not proceeding further.";
		return -1;
	}
	//
	if (fs::exists(datadir / "utt2spk")) 
	{
		StringTable table_uttstmp;
		if (ReadStringTable((tmpdir / "utts").string(), table_uttstmp) < 0) return -1;
		int new_nutts = table_uttstmp.size();
		int old_nutts = table_utt2spk.size();
		if (new_nutts != old_nutts)
			LOGTW_INFO << " kept " << new_nutts << " utterances out of " << old_nutts << ".";
		else
			LOGTW_INFO << " kept all " << old_nutts << " utterances.";			
	}

	string_vec files1 = { "utt2spk","utt2uniq","feats.scp","vad.scp","text","segments","utt2lang","utt2dur","utt2num_frames" };
	if (maybe_wav != "") files1.push_back(maybe_wav);
	try
	{
		for each(std::string x in files1) {
			fs::path p(datadir / x);
			if (fs::exists(p)) {
				fs::copy_file(p, datadir / "backup" / p.filename(), fs::copy_option::overwrite_if_exists);
				if (FilterScp(tmpdir / "utts", p, p.string() + ".temp") < 0) {
					LOGTW_ERROR << " Failed to filter file " << p.string() << ".";
					return -1;
				}
				//
				StringTable table1, table2;
				if (ReadStringTable(p.string(), table1) < 0) return -1;
				if (ReadStringTable(p.string() + ".temp", table2) < 0) return -1;
				if (!IsTheSame(table1, table2)) {
					if (FilterScp(tmpdir / "utts", datadir / "backup" / p.filename(), p) < 0) {
						LOGTW_ERROR << " Failed to filter file " << (datadir / "backup" / p.filename()).string() << ".";
						return -1;
					}
				}
				try {
					fs::remove(p.string() + ".temp");
				} catch (const std::exception&) { }
			}
		}

		//filter utt_extra_files
		for each(fs::path p in utt_extra_files) {
			if (fs::exists(p)) {
				fs::copy_file(p, datadir / "backup" / p.filename(), fs::copy_option::overwrite_if_exists);
				if (FilterScp(tmpdir / "utts", p, p.string() + ".temp") < 0) {
					LOGTW_ERROR << " Failed to filter file " << p.string() << ".";
					return -1;
				}
				//
				StringTable table1, table2;
				if (ReadStringTable(p.string(), table1) < 0) return -1;
				if (ReadStringTable(p.string() + ".temp", table2) < 0) return -1;
				if (!IsTheSame(table1, table2)) {
					if (FilterScp(tmpdir / "utts", datadir / "backup" / p.filename(), p) < 0) {
						LOGTW_ERROR << " Failed to filter file " << (datadir / "backup" / p.filename()).string() << ".";
						return -1;
					}
				}
				try {
					fs::remove(p.string() + ".temp");
				}
				catch (const std::exception&) {}
			}
		}
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << " " << ex.what() << ".";
		return -1;
	}

	return 0;
}

