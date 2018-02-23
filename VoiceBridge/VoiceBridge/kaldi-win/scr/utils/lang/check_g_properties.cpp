/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"

/*
 This script checks that G.fst in the lang.fst directory is OK with respect to certain expected properties, 
 and returns nonzero exit status if a problem was detected. It is called from validate_lang.
 This only checks the properties of G that relate to disambiguation symbols, epsilons and forbidden symbols <s> and </s>.
*/

int IsCyclic(fs::path langdir, std::string sFst, bool & bIsCyclic);


int CheckGProperties(fs::path langdir)
{
	StringTable table_;

	fs::path path_G_fst(langdir / "G.fst");
	if (CheckFileExistsAndNotEmpty(path_G_fst, true) < 0) return -1;

	fs::path path_words_txt(langdir / "words.txt");
	if (CheckFileExistsAndNotEmpty(path_words_txt, true) < 0) return -1;
	StringTable table_words_txt;
	if (ReadStringTable(path_words_txt.string(), table_words_txt) < 0) {
		LOGTW_ERROR << " fail to open: " << path_words_txt.string() << ".";
		return -1;
	}
	std::vector<std::string> is_forbidden, is_disambig;
	std::string hash_zero = "";
	for (StringTable::const_iterator it(table_words_txt.begin()), it_end(table_words_txt.end()); it != it_end; ++it)
	{		
		std::string sym_ = (*it)[0];
		std::string int_ = (*it)[1];
		if (sym_ == "<s>" || sym_ == "</s>") {
			is_forbidden.push_back(int_);
		}
		if (sym_ == "#0") {
			hash_zero = int_;
		}
	}

	fs::path path_wdisambig_words_int(langdir / "phones\\wdisambig_words.int");
	if (CheckFileExistsAndNotEmpty(path_wdisambig_words_int, false) >= 0)
	{
		if (ReadStringTable(path_wdisambig_words_int.string(), table_) < 0) {
			LOGTW_ERROR << "Fail to open: " << path_wdisambig_words_int.string() << ".";
			return -1;
		}
		for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it)
		{
			std::string s;
			int c = 0;
			for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
			{
				if (c > 0) s.append(" ");
				s.append(*itc1);
				c++;
			}
			if (s != "") {
				is_disambig.push_back(s);
			}
		}		
	}
	else {
		is_disambig.push_back(hash_zero);
	}

	fs::path path_print_out(langdir / "G_fst_print.temp");
	fs::path path_compiled(langdir / "G_fst_compiled.temp");
	fstprint(path_G_fst.string(), path_print_out.string());
	if (CheckFileExistsAndNotEmpty(path_print_out, true) < 0) return -1;	
	bool has_epsilons = false;
	if (ReadStringTable(path_print_out.string(), table_) < 0) {
		LOGTW_ERROR << "Fail to open: " << path_print_out.string() << ".";
		return -1;
	}
	int indx = 0;
	for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it)
	{
		indx++;
		if ((*it).size() >= 4)
		{
			std::string A2((*it)[2]), A3((*it)[3]);
			if (std::find(is_forbidden.begin(), is_forbidden.end(), A2) != is_forbidden.end() ||
				std::find(is_forbidden.begin(), is_forbidden.end(), A3) != is_forbidden.end())
			{
				LOGTW_ERROR << " line " << indx << " in G.fst contains forbidden symbol <s> or </s>.";
				return -1;
			}
			else if (std::find(is_disambig.begin(), is_disambig.end(), A2) != is_forbidden.end())
			{
				if (is_positive_int(A3) && std::stoi(A3) != 0) {
					LOGTW_ERROR << " line " << indx << " in G.fst has disambig on input but no epsilon on output.";
					return -1;
				}
				//Check if the fst is cyclic
				std::string fstline;
				int c = 0;
				for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) {
					if (c > 0) fstline.append(" ");
					fstline.append(*itc1);
					c++;
				}
				bool bIsCyclic = false;
				if (IsCyclic(langdir, fstline, bIsCyclic) < 0) return -1;
				if (bIsCyclic) {
					LOGTW_ERROR << "G.fst has cycles containing only disambig symbols and epsilons. Would cause determinization failure.";
					return -1;
				}
			}
			else if (!is_positive_int(A2) || std::stoi(A2) == 0) {
				has_epsilons = true;
				//Check if the fst is cyclic
				std::string fstline;
				int c = 0;
				for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) {
					if (c > 0) fstline.append(" ");
					fstline.append(*itc1);
					c++;
				}
				bool bIsCyclic = false;
				if (IsCyclic(langdir, fstline, bIsCyclic) < 0) return -1;
				if (bIsCyclic) {
					LOGTW_ERROR << " G.fst has cycles containing only disambig symbols and epsilons. Would cause determinization failure.";
					return -1;
				}
			}
			else if(A2 != A3) {
				LOGTW_ERROR << " line " << indx << " in G.fst has inputs and outputs different but input is not disambig symbol.";
				return -1;
			}
		}
	}

	if (has_epsilons) {
		LOGTW_WARNING << "Validating lang: G.fst has epsilon-input arcs.  We don't expect these in most setups.";
	}

	LOGTW_INFO << "  --> successfully validated lang/G.fst.";

	return 0;
}

int IsCyclic(fs::path langdir, std::string sFst, bool & bIsCyclic)
{
	//must write out sFst into a file
	fs::path path_out(langdir / "G_fst.temp"), path_compiled(langdir / "G_fst_compiled.temp");
	//NOTE: ofstream write adds an '\r' in front of the '\n' automatically and some code in Kaldi crashes. To prevent this
	//		add std::ios::binary option to each ofstream to make sure that '\r' is not added!
	fs::ofstream file_(path_out, std::ios::binary);
	if (!file_) {
		LOGTW_ERROR << "Can't open output file: " << path_out.string() << ".";
		return -1;
	}
	file_ << sFst << "\n";
	file_.flush(); file_.close();
	try {
		fstcompile(false, "standard", "vector",
			path_out.string(), path_compiled.string(),
			"", "", "",
			false, false, false, false);
	}
	catch (const std::exception& message) {
		LOGTW_ERROR << "Error while compiling fst's, check file: " << path_out.string() << "\nDetail: " << message.what() << ".";
		return -1;
	}
	//get info
	fst::FstInfo info;
	if (fstinfo(path_compiled.string(), "any", "auto", true, true, &info) < 0) return -1;

	uint64 properties = info.Properties();
	//each property is set as a bit in properties
	//NOTE: properties are trinary - they are either true, false or unknown. For each such property, 
	//		there are two stored bits; one is set if true, the other is set if false and neither is set if unknown.
	//		Here: kOLabelSorted is set if true, kNotOLabelSorted is set if false, if neither is set then unkonwn!
	bIsCyclic = IsBitSet(properties, fst::kCyclic);

	return 0;
}
