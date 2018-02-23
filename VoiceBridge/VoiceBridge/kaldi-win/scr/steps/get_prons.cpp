/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on :
	Apache 2.0.
	Copyright  2014  Johns Hopkins University (Author: Daniel Povey)
			   2014  Guoguo Chen
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"
#include "kaldi-win/utility/Utility2.h"

static void LaunchJobLatticeBest(
	int JOBID,
	string_vec options_latticealignwords,
	string_vec options,
	bool bLatticeAlignWords,
	bool bNbest,
	fs::path mdl,
	fs::path dir,
	std::string symtab, std::string input_txt, std::string output_txt, int field_begin, int field_end, std::string oov,	//params for Sym2Int
	fs::path sdata,
	fs::path log
);

//the return values from each thread/job
static std::vector<int> _ret;

/*
	GetProns()
	This function writes files prons.* in the directory provided, which must contain alignments (ali.*) 
	or lattices (lat.*).  These files are as output by nbest-to-prons (see its usage message).
*/
int GetProns(
	fs::path data,	//data directory
	fs::path lang,	//language directory
	fs::path dir,	//output directory
	int stage,		//stage number
	int lmwt)		//scale for LM, only applicable for lattice input (default: 10)
{
	//check if files exist
	if (CheckIfFilesExist(std::vector<fs::path> {data / "utt2spk", lang / "words.txt", dir / "num_jobs"}) < 0) return -1;

	//get nj
	std::string snj;
	int nj;
	try {
		snj = GetFirstLineFromFile((dir / "num_jobs").string());
	}
	catch (const std::exception&) {}
	nj = StringToNumber<int>(snj, -1);
	if (nj < 1) {
		LOGTW_ERROR << "Could not read number of jobs from file " << (dir / "num_jobs").string() << ".";
		return -1;
	}

	fs::path sdata(data / ("split" + std::to_string(nj)));
	if (!(fs::exists(sdata) && fs::is_directory(sdata) && fs::last_write_time(data / "feats.scp") < fs::last_write_time(sdata)))
	{
		//split data directory
		if (SplitData(data, nj) < 0) return -1;
	}

	//get oov
	if (CheckFileExistsAndNotEmpty(lang / "oov.int", true) < 0) return -1;
	StringTable t_oov;
	if (ReadStringTable((lang / "oov.int").string(), t_oov) < 0) return -1;
	std::string oov(t_oov[0][0]);

	//work out model name/path
	fs::path mdl;
	if (fs::exists(dir / "final.mdl"))
		mdl = dir / "final.mdl";
	else {
		if (fs::exists(dir.parent_path() / "final.mdl"))
			mdl = dir.parent_path() / "final.mdl";  // e.g.decoding directories.
		else {
			LOGTW_ERROR << "Expected dir/final.mdl or dir/../final.mdl to exist.";
			return -1;
		}
	}

	//prepare options for lattice-align-words or lattice-align-words-lexicon
	bool bLatticeAlignWords = true; //if true then will use lattice-align-words, if false will use lattice-align-words-lexicon
	string_vec options_latticealignwords, options_best;
	options_best.push_back("--print-args=false");
	options_latticealignwords.push_back("--print-args=false");
	if (fs::exists(lang / "phones" / "word_boundary.int")) {
		options_latticealignwords.push_back((lang / "phones" / "word_boundary.int").string());
		bLatticeAlignWords = true;
	}
	else {
		if (!fs::exists(lang / "phones" / "align_lexicon.int")) {
			LOGTW_ERROR << "Expected either lang/phones/word_boundary.int or lang/phones/align_lexicon.int to exist.";
			return -1;
		}
		else {
			options_latticealignwords.push_back((lang / "phones" / "align_lexicon.int").string());
			bLatticeAlignWords = false;
		}
	}
	options_latticealignwords.push_back(mdl.string());
	options_latticealignwords.push_back("ark:" + (dir / "best.JOBID.temp").string()); //input
	options_latticealignwords.push_back("ark:" + (dir / "law.JOBID.temp").string());  //output

	//Sym2Int options
	std::string symtab((lang / "words.txt").string());
	std::string input_txt((sdata / "JOBID" / "text").string());
	std::string output_txt((dir / "sym2int.JOBID.temp").string()); //output from Sym2Int
	//oov is defined above
	int field_begin = 1; //NOTE: zero based index of fields! 2nd field is index=1
	int field_end = -1;

	//remove former results
	if (stage <= 1) {
		try {
			for (int JOBID = 1; JOBID <= nj; JOBID++)
				if (fs::exists(dir / ("prons." + std::to_string(JOBID))))
					fs::remove(dir / ("prons." + std::to_string(JOBID)));
		}
		catch (const std::exception&) {}
	}

	bool bNbest = true; //if true linear-to-nbest is called, if false lattice-1best is called
	if (fs::exists(dir / "ali.1")) {
		LOGTW_INFO << "dir/ali.1 exists, so starting from alignments...";
		LOGTW_INFO << "dir/ali.1: " << (dir / "ali.1").string();
		bNbest = true;
		//prepare options
		options_best.push_back("ark:" + (dir / "ali.JOBID").string());
		options_best.push_back("ark:" + (dir / "sym2int.JOBID.temp").string()); //output from Sym2Int
		options_best.push_back("");	//deliberately empty
		options_best.push_back("");	//deliberately empty
		options_best.push_back("ark:" + (dir / "best.JOBID.temp").string()); //output from linear-to-nbest
	}
	else {
		if (!fs::exists(dir / "lat.1")) {
			LOGTW_ERROR << "Expected either dir/ali.1 or dir/lat.1 to exist.";
			LOGTW_INFO << "dir/ali.1: " << (dir / "ali.1").string();
			LOGTW_INFO << "dir/lat.1: " << (dir / "lat.1").string();
			return -1;
		}
		LOGTW_INFO << "dir/lat.1 exists, so starting from lattices...";
		LOGTW_INFO << "dir/lat.1: " << (dir / "lat.1").string();
		bNbest = false;
		//prepare options
		options_best.push_back("--lm-scale=" + std::to_string(lmwt));
		options_best.push_back("ark:" + (dir / "lat.JOBID").string());
		options_best.push_back("ark:" + (dir / "best.JOBID.temp").string()); //output from lattice-1best
	}

	if (stage <= 1) {
		//call parallel processing
		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(dir / "log" / ("nbest_to_prons." + std::to_string(JOBID) + ".log"));
			_threads.emplace_back(
				LaunchJobLatticeBest,
				JOBID,
				options_latticealignwords,
				options_best,
				bLatticeAlignWords,
				bNbest,
				mdl,
				dir,
				symtab, input_txt, output_txt, field_begin, field_end, oov,	//params for Sym2Int
				sdata,
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
		//---------------------------------------------------------------------

		//cleanup
		DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
		for (int JOBID = 1; JOBID <= nj; JOBID++)
			DeleteAllMatching(sdata / std::to_string(JOBID), boost::regex(".*(\\.temp)$"));

	}

	if (stage <= 2) {
		//skip the first 3 fields and add the rest and count the number of occurances => output into dir/pron_counts.int
		std::unordered_map<std::string, int> count;
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			StringTable t;
			if (ReadStringTable((dir / ("prons." + std::to_string(JOBID))).string(), t) < 0) return -1;
			for (std::vector<std::string> _s : t) {
				std::string s;
				if (_s.size() > 3) {
					for (int c = 3; c < _s.size(); c++) {
						if (c > 3) s.append(" ");
						s.append(_s[c]);
					}
					std::unordered_map<std::string, int>::iterator im = count.find(s);
					if (im == count.end()) {
						count.emplace(s, 1);
					}
					else im->second++;
				}
			}
		}
		try {
			fs::ofstream ofs(dir / "pron_counts.int", fs::ofstream::binary | fs::ofstream::out);
			if (!ofs) {
				LOGTW_ERROR << "Could not open file " << (dir / "pron_counts.int").string();
				return -1;
			}
			for (auto & p : count) {
				ofs << p.second << " " << p.first << "\n";
			}
			ofs.flush(); ofs.close();
		}
		catch (const std::exception&)
		{
			LOGTW_ERROR << "Could not write file " << (dir / "pron_counts.int").string();
			return -1;
		}
	}

	if (stage <= 3) {
		//1.
		StringTable t_symtab, t_i;
		if (ReadStringTable((lang / "words.txt").string(), t_symtab) < 0) return -1;
		if (ReadStringTable((dir / "pron_counts.int").string(), t_i) < 0) return -1;
		fs::path sym_out(dir / "int2sym_1.temp");
		if (Int2Sym(t_symtab, t_i, sym_out, 1, 1) < 0)  return -1;  //NOTE: zero based index of fields!

		//2.
		t_symtab.clear();
		t_i.clear();
		if (ReadStringTable((lang / "phones.txt").string(), t_symtab) < 0) return -1;
		if (ReadStringTable((dir / "int2sym_1.temp").string(), t_i) < 0)  return -1;
		sym_out = dir / ("int2sym_2.temp");
		if (Int2Sym(t_symtab, t_i, sym_out, 2, -1) < 0) return -1; //NOTE: zero based index of fields!

		//numerical reverse sort
		StringTable t_proncounts;
		if (ReadStringTable(sym_out.string(), t_proncounts) < 0) return -1;
		SortStringTable(t_proncounts, 0, 0, "number", "number", false, false, false);
		if (SaveStringTable((dir / "pron_counts.txt").string(), t_proncounts) < 0) return -1;

		//cleanup
		DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
	}

	static const boost::regex rexp("_[BISE]$");

	if (stage <= 4) {
		//remove the _B, _I, _S, _E markers from phones; this is often convenient if we want to go back to a 
		//word-position-independent source lexicon.
		if (fs::exists(lang / "phones" / "word_boundary.int")) {
			fs::ofstream ofs(dir / "pron_counts_nowb.txt", fs::ofstream::binary | fs::ofstream::out);
			if (!ofs) {
				LOGTW_ERROR << "Could not open file " << (dir / "pron_counts_nowb.txt").string();
				return -1;
			}
			StringTable t_proncounts;			
			if (ReadStringTable((dir / "pron_counts.txt").string(), t_proncounts) < 0) return -1;
			for (std::vector<std::string> _s : t_proncounts) {
				std::string line;
				line.append(_s[0]);
				line.append(" ");
				line.append(_s[1]);
				line.append(" ");
				for (int c = 2; c < _s.size(); c++) {
					if (c > 2) line.append(" ");
					std::string s = boost::regex_replace(_s[c], rexp, "");
					line.append(s);
				}
				ofs << line << "\n";
			}
			ofs.flush(); ofs.close();
		}
	}

	if (stage <= 5) {
		/*
		 Here we figure the count of silence before and after words (actually prons)
		 1. Create a text like file, but instead of putting words, we write "word pron" pairs.
		 We change the format of prons.*.gz from pron-per-line to utterance-per-line
		 (with "word pron" pairs tab-separated), and add <s> and </s> at the begin and end of each sentence.
		 The _B, _I, _S, _E markers are removed from phones.
		*/
		StringTable t_prons;
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			StringTable t;
			if (ReadStringTable((dir / ("prons." + std::to_string(JOBID))).string(), t) < 0) return -1;
			t_prons.insert(std::end(t_prons), std::begin(t), std::end(t));
		}
		StringTable t_symtab, t_i;
		if (ReadStringTable((lang / "words.txt").string(), t_symtab) < 0) return -1;
		fs::path sym_out(dir / "int2sym_1.temp");
		if (Int2Sym(t_symtab, t_prons, sym_out, 3, 3) < 0)  return -1;  //NOTE: zero based index of fields!
		t_symtab.clear();
		t_i.clear();
		if (ReadStringTable((lang / "phones.txt").string(), t_symtab) < 0) return -1;
		if (ReadStringTable((dir / "int2sym_1.temp").string(), t_i) < 0)  return -1;
		sym_out = dir / ("int2sym_2.temp");
		if (Int2Sym(t_symtab, t_i, sym_out, 4, -1) < 0) return -1; //NOTE: zero based index of fields!
		//
		StringTable t_cut;
		if (ReadStringTable(sym_out.string(), t_cut) < 0) return -1;
		std::string utter_id = "";
		//
		fs::ofstream ofs(dir / "pron_perutt_nowb.txt", fs::ofstream::binary | fs::ofstream::out);
		if (!ofs) {
			LOGTW_ERROR << "Could not open file " << (dir / "pron_perutt_nowb.txt").string();
			return -1;
		}
		for (std::vector<std::string> _s : t_cut) 
		{
			if (utter_id == "") { utter_id = _s[0]; ofs << utter_id << "\t<s>"; }
			else if (utter_id != _s[0]) {
				ofs << "\t</s>\n";
				utter_id = _s[0]; 
				ofs << utter_id << "\t<s>";
			}
			ofs << "\t" << _s[3];
			for (int n = 4; n < _s.size(); n++) {
				std::string s = boost::regex_replace(_s[n], rexp, "");
				ofs << " " << s;
			}
		}
		ofs << "\t</s>\n";
		ofs.flush(); ofs.close();

		/*
		 2. Collect bigram counts for words. To be more specific, we are actually collecting counts for 
		 "v ? w", where "?" represents silence or non-silence.
		*/
		static const boost::regex rexp2("<eps>[^\\t]*\\t");
		fs::ifstream ifs(dir / "pron_perutt_nowb.txt");
		if (!ifs) {
			LOGTW_ERROR << "Could not open file " << (dir / "pron_perutt_nowb.txt").string();
			return -1;
		}
		std::string line;
		std::unordered_map<std::string, int> bigram;
		while (std::getline(ifs, line)) {
			boost::algorithm::trim(line);
			std::string sline = boost::regex_replace(line, rexp2, "");			
			std::vector<std::string> _v;
			strtk::parse(sline, "\t", _v, strtk::split_options::compress_delimiters);
			for (int i = 0; i < _v.size() - 1; i++) {
				std::string s(_v[i] + "\t" + _v[i + 1]);
				std::unordered_map<std::string, int>::iterator im = bigram.find(s);
				if (im == bigram.end()) {
					bigram.emplace(s, 1);
				}
				else im->second++;
			}
		}
		ifs.close();
		fs::ofstream ofs2(dir / "pron_bigram_counts_nowb.txt", fs::ofstream::binary | fs::ofstream::out);
		if (!ofs2) {
			LOGTW_ERROR << "Could not open file " << (dir / "pron_bigram_counts_nowb.txt").string();
			return -1;
		}
		for (auto & pair : bigram)
			ofs2 << pair.second << "\t" << pair.first << "\n";
		ofs2.flush(); ofs2.close();
	
		/*
		 3. Collect bigram counts for silence and words. the count file has 4 fields
		    for counts, followed by the "word pron" pair. All fields are separated by
		    spaces:
		    <sil-before-count> <nonsil-before-count> <sil-after-count> <nonsil-after-count> <word> <phone1> <phone2 >...
		*/
		StringTable t_ppn;
		if (ReadStringTable((dir / "pron_perutt_nowb.txt").string(), t_ppn, "\t") < 0) return -1;
		static const boost::regex regex3("^<eps> ");
		boost::match_results<std::string::const_iterator> match_results;
		boost::match_flag_type flags = boost::match_default;
		std::unordered_map<std::string, int> sil_wpron, nonsil_wpron, wpron_sil, wpron_nonsil;
		std::map<std::string, int> words;
		for (std::vector<std::string> _s : t_ppn)
		{
			if (_s.size() < 3) {
				LOGTW_ERROR << "Bad line in " << (dir / "pron_perutt_nowb.txt").string();
				return -1;
			}
			//NOTE: skip id's
			for (int n = 1; n < _s.size() - 1; n++) {
				// First word is not silence, collect the wpron_sil and wpron_nonsil stats.
				if (!boost::regex_search(_s[n], match_results, regex3, flags))
				{
					if (boost::regex_search(_s[n+1], match_results, regex3, flags)) {
						//
						std::unordered_map<std::string, int>::iterator im = wpron_sil.find(_s[n]);
						if (im == wpron_sil.end()) {
							wpron_sil.emplace(_s[n], 1);
						}
						else im->second++;
					}
					else {
						//
						std::unordered_map<std::string, int>::iterator im = wpron_nonsil.find(_s[n]);
						if (im == wpron_nonsil.end()) {
							wpron_nonsil.emplace(_s[n], 1);
						}
						else im->second++;
					}
					//
					std::map<std::string, int>::iterator im = words.find(_s[n]);
					if (im == words.end()) {
						words.emplace(_s[n], 1);
					}
				}
				//Second word is not silence, collect the sil_wpron and nonsil_wpron stats.
				if (!boost::regex_search(_s[n+1], match_results, regex3, flags))
				{
					if (boost::regex_search(_s[n], match_results, regex3, flags)) {
						//
						std::unordered_map<std::string, int>::iterator im = sil_wpron.find(_s[n+1]);
						if (im == sil_wpron.end()) {
							sil_wpron.emplace(_s[n+1], 1);
						}
						else im->second++;
					}
					else {
						//
						std::unordered_map<std::string, int>::iterator im = nonsil_wpron.find(_s[n+1]);
						if (im == nonsil_wpron.end()) {
							nonsil_wpron.emplace(_s[n+1], 1);
						}
						else im->second++;
					}
					//
					std::map<std::string, int>::iterator im = words.find(_s[n+1]);
					if (im == words.end()) {
						words.emplace(_s[n+1], 1);
					}
				}
			}
		}

		//
		fs::ofstream ofs3(dir / "sil_counts_nowb.txt", fs::ofstream::binary | fs::ofstream::out);
		if (!ofs3) {
			LOGTW_ERROR << "Could not open file " << (dir / "sil_counts_nowb.txt").string();
			return -1;
		}
		for (auto & pair : words) {
			std::unordered_map<std::string, int>::iterator im1 = sil_wpron.find(pair.first);
			if (im1 == sil_wpron.end()) sil_wpron.emplace(pair.first, 0);
			std::unordered_map<std::string, int>::iterator im2 = nonsil_wpron.find(pair.first);
			if (im2 == nonsil_wpron.end()) nonsil_wpron.emplace(pair.first, 0);
			std::unordered_map<std::string, int>::iterator im3 = wpron_sil.find(pair.first);
			if (im3 == wpron_sil.end()) wpron_sil.emplace(pair.first, 0);
			std::unordered_map<std::string, int>::iterator im4 = wpron_nonsil.find(pair.first);
			if (im4 == wpron_nonsil.end()) wpron_nonsil.emplace(pair.first, 0);
			ofs3 << sil_wpron[pair.first] << " " << nonsil_wpron[pair.first] << " ";
			ofs3 << wpron_sil[pair.first] << " " << wpron_nonsil[pair.first] << " " << pair.first << "\n";
		}
		ofs3.flush(); ofs3.close();

		//cleanup
		DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
	}

	LOGTW_INFO << "Done writing prons to dir/prons.*, silence counts in ";
	LOGTW_INFO << "dir/sil_counts_nowb.txt and pronunciation counts in dir/pron_counts.{int,txt}";
	if (fs::exists(lang / "phones" / "word_boundary.int"))
		LOGTW_INFO << "and also in dir/pron_counts_nowb.txt. Destination directory: ";
	LOGTW_INFO << dir.string();

	return 0;
}

static void LaunchJobLatticeBest(
	int JOBID,
	string_vec options_latticealignwords,
	string_vec options_best, //options to linear-to-nbest or lattice-1best depending on bNbest
	bool bLatticeAlignWords,
	bool bNbest,
	fs::path mdl,
	fs::path dir,
	std::string symtab, std::string input_txt, std::string output_txt, int field_begin, int field_end, std::string oov,	//params for Sym2Int
	fs::path sdata,
	fs::path log
)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	//replace 'JOBID' with the current job ID of the thread
	for (std::string &s : options_latticealignwords) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_best) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	ReplaceStringInPlace(input_txt, "JOBID", std::to_string(JOBID));
	ReplaceStringInPlace(output_txt, "JOBID", std::to_string(JOBID));

	int ret = 0;

	//best to prons
	if (bNbest) {
		//sym2int
		StringTable t_symtab, t_input;
		if (ReadStringTable(symtab, t_symtab) < 0) {
			_ret.push_back(-1);
			return;
		}
		if (ReadStringTable(input_txt, t_input) < 0) {
			_ret.push_back(-1);
			return;
		}
		try {
			if (Sym2Int(t_symtab, t_input, output_txt, field_begin, field_end, oov) < 0) {
				_ret.push_back(-1);
				return;
			}
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (Sym2Int). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}

		//Linear To Nbest
		try {
			StrVec2Arg args(options_best);
			ret = LinearToNbest(args.argc(), args.argv(), file_log);
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (LinearToNbest). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
		if (ret < 0) {
			//do not proceed if failed
			_ret.push_back(ret);
			return;
		}
	}
	else {
		//Lattice 1 best
		try {
			StrVec2Arg args(options_best);
			ret = Lattice1best(args.argc(), args.argv(), file_log);
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (Lattice1best). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
		if (ret < 0) {
			//do not proceed if failed
			_ret.push_back(ret);
			return;
		}
	}

	//align words
	if (bLatticeAlignWords) {
		try {
			StrVec2Arg args(options_latticealignwords);
			ret = LatticeAlignWords(args.argc(), args.argv(), file_log);
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (LatticeAlignWords). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
		if (ret < 0) {
			//do not proceed if failed
			_ret.push_back(ret);
			return;
		}
	}
	else {
		try {
			StrVec2Arg args(options_latticealignwords);
			ret = LatticeAlignWordsLexicon(args.argc(), args.argv(), file_log);
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (LatticeAlignWordsLexicon). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
		if (ret < 0) {
			//do not proceed if failed
			_ret.push_back(ret);
			return;
		}
	}

	//Nbest To Prons
	try {
		string_vec options;
		options.push_back("--print-args=false");
		options.push_back(mdl.string());
		options.push_back("ark:" + (dir / ("law."+ std::to_string(JOBID) + ".temp")).string());
		options.push_back((dir / ("prons."+ std::to_string(JOBID))).string());
		StrVec2Arg args(options);
		ret = NbestToProns(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (NbestToProns). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
 
	_ret.push_back(ret);
}
