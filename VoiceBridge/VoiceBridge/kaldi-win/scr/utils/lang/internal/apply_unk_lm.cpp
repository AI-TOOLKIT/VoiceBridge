
// TODO: not tested!

/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/
#include "kaldi-win\scr\kaldi_scr.h"
//#include "fst_ext.h"

/*
This function is called from the end of prepare_lang, inserts the unknown-word LM FST into the lexicon FSTs
<lang-dir>/L.fst and <lang-dir>/L_disambig.fst in place of the special disambiguation symbol #2 (which was inserted by
add_lex_disambig as a placeholder for this FST).

  input_unk_lm_fst :  A text-form FST, typically with the name unk_fst.txt.  We will remove all symbols from the
               output before applying it.
  lang_dir :  A partially built lang/ directory.  We modify L.fst and L_disambig.fst, and read only words.txt.
*/

int ApplyUnkLM(StringTable input_unk_lm_fst, fs::path lang_dir)
{
	//check files
	if (!(fs::exists(lang_dir / "L.fst") && !fs::is_empty(lang_dir / "L.fst"))) {
		LOGTW_ERROR << "File does not exist: " << (lang_dir / "L.fst").string() << ".";
		return -1;
	}
	if (!(fs::exists(lang_dir / "L_disambig.fst") && !fs::is_empty(lang_dir / "L_disambig.fst"))) {
		LOGTW_ERROR << "File does not exist: " << (lang_dir / "L_disambig.fst").string() << ".";
		return -1;
	}
	if (!(fs::exists(lang_dir / "words.fst") && !fs::is_empty(lang_dir / "words.fst"))) {
		LOGTW_ERROR << "File does not exist: " << (lang_dir / "words.fst").string() << ".";
		return -1;
	}
	if (!(fs::exists(lang_dir / "oov.int") && !fs::is_empty(lang_dir / "oov.int"))) {
		LOGTW_ERROR << "File does not exist: " << (lang_dir / "oov.int").string() << ".";
		return -1;
	}
	if (!(fs::exists(lang_dir / "phones.txt") && !fs::is_empty(lang_dir / "phones.txt"))) {
		LOGTW_ERROR << "File does not exist: " << (lang_dir / "phones.txt").string() << ".";
		return -1;
	}

	//phones.txt
	StringTable table_;
	if (ReadStringTable((lang_dir / "phones.txt").string(), table_) < 0) return -1;
	std::string s = table_[table_.size()-1][1];
	int unused_phone_label = -1;
	if (is_positive_int(s)) unused_phone_label = std::stoi(s);
	int label_to_replace = -1;
	for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it)
	{
		if ((*it)[0] == "#2") {
			label_to_replace = std::stoi((*it)[1]);
		}
	}
	if (unused_phone_label < 0 || label_to_replace < 0) {
		LOGTW_ERROR << "Error getting unused phone label or label for #2.";
		return -1;
	}

	//Now fstreplace works based on olabels, but we actually want to deal with ilabels,
	//so we need to invert all the FSTs before and after doing fstreplace.
	for (StringTable::iterator it(input_unk_lm_fst.begin()), it_end(input_unk_lm_fst.end()); it != it_end; ++it)
	{
		if ((*it).size() >= 4) {
			(*it)[3] = "<eps>";
		}
	}	
	//for in case the table is changed above with <eps> save the file into a temp file
	fs::path path_temp(lang_dir / "input_unk_lm_fst.temp");
	if (SaveStringTable(path_temp.string(), input_unk_lm_fst) < 0) return -1;
	fs::path path_temp_compiled(lang_dir / "input_unk_lm_fst_compiled.temp");
	fs::path path_isymbols(lang_dir / "phones.txt");
	fs::path path_osymbols(lang_dir / "words.txt");
	//compile
	try
	{
		fstcompile(false, "standard", "vector",
			path_temp.string(), path_temp_compiled.string(),
			path_isymbols.string(), path_osymbols.string(), "",
			false, false, false, false);
	}
	catch (const std::exception& message)
	{
		LOGTW_ERROR << "Error while compiling fst's, check file: " << path_temp.string() << "\nDetail: " << message.what() << ".";
		return -1;
	}

	fs::path path_unk_temp_fst(lang_dir / "unk_temp.fst");
	//invert
	if (fstinvert( path_temp_compiled.string(), path_unk_temp_fst.string() ) < 0) return -1;
	//get info
	fst::FstInfo *info = NULL;
	if( fstinfo(path_unk_temp_fst.string(), "any", "auto", true, true, info) < 0 ) return -1;
	int64 num_states_unk = info->NumStates();

	// ------ L.fst ------
	//now we insert the unknown-word LM FST into the lexicon FSTs - replace the placeholder added before
	//the rootlabel should just be an otherwise unused symbol. All the labels are olabels(word labels)..that is hardcoded in fstreplace.
	fs::path path_L_fst(lang_dir / "L.fst");
	fs::path path_L_fst_temp(lang_dir / "L.fst.temp");
	fs::path path_LI_fst(lang_dir / "LI.fst.temp");
	fs::path path_LR_fst(lang_dir / "LR.fst.temp");
	//must invert because of the replace
	if (fstinvert(path_L_fst.string(), path_LI_fst.string()) < 0) return -1;
	//replace
	FSTLABELMAP fstlabelmap;
	fstlabelmap.emplace((lang_dir / "unk_temp.fst").string(), std::to_string(label_to_replace));
	if (fstreplace(path_LI_fst.string(), path_LR_fst.string(), std::to_string(unused_phone_label), fstlabelmap) < 0) return -1;
	//invert it back
	if (fstinvert(path_LR_fst.string(), path_L_fst_temp.string()) < 0) return -1;
	//get info
	if (fstinfo(path_L_fst.string(), "any", "auto", true, true, info) < 0) return -1;
	int64 num_states_old = info->NumStates();
	if (fstinfo(path_L_fst_temp.string(), "any", "auto", true, true, info) < 0) return -1;
	int64 num_states_new = info->NumStates();
	int64 num_states_added = num_states_new - num_states_old;
	LOGTW_INFO << "In " << path_L_fst.string() << ", substituting in the unknown-word LM (which had " <<
				num_states_unk << " states added " << num_states_added << " new FST states.";
	try	{
		fs::copy_file(path_L_fst_temp, path_L_fst, fs::copy_option::overwrite_if_exists);
	}
	catch (const std::exception&) {
		LOGTW_ERROR << "Could not copy file " << path_L_fst_temp.string() << " to file " << path_L_fst.string() << ".";
		return -1;
	}

	// ------ L_disambig.fst ------
	//now we insert the unknown-word LM FST into the lexicon FSTs - replace the placeholder added before
	//the rootlabel should just be an otherwise unused symbol. All the labels are olabels(word labels)..that is hardcoded in fstreplace.
	fs::path path_L_disambig_fst(lang_dir / "L_disambig.fst");
	fs::path path_L_disambig_fst_temp(lang_dir / "L_disambig.fst.temp");
	fs::path path_L_disambigI_fst(lang_dir / "L_disambigI.fst.temp");
	fs::path path_L_disambigR_fst(lang_dir / "L_disambigR.fst.temp");
	//must invert because of the replace
	if (fstinvert(path_L_disambig_fst.string(), path_L_disambigI_fst.string()) < 0) return -1;
	//replace
	FSTLABELMAP fstlabelmap1;
	fstlabelmap1.emplace((lang_dir / "unk_temp.fst").string(), std::to_string(label_to_replace));
	if (fstreplace(path_L_disambigI_fst.string(), path_L_disambigR_fst.string(), std::to_string(unused_phone_label), fstlabelmap1) < 0) return -1;
	//invert it back
	if (fstinvert(path_L_disambigR_fst.string(), path_L_disambig_fst_temp.string()) < 0) return -1;
	//get info
	if (fstinfo(path_L_disambig_fst.string(), "any", "auto", true, true, info) < 0) return -1;
	num_states_old = info->NumStates();
	if (fstinfo(path_L_disambig_fst_temp.string(), "any", "auto", true, true, info) < 0) return -1;
	num_states_new = info->NumStates();
	num_states_added = num_states_new - num_states_old;
	LOGTW_INFO << "In " << path_L_disambig_fst.string() << ", substituting in the unknown-word LM (which had " << num_states_unk 
			  << " states added " << num_states_added << " new FST states.";
	try {
		fs::copy_file(path_L_disambig_fst_temp, path_L_disambig_fst, fs::copy_option::overwrite_if_exists);
	}
	catch (const std::exception&) {
		LOGTW_ERROR << "Could not copy file " << path_L_disambig_fst_temp.string() << " to file " << path_L_disambig_fst.string() << ".";
		return -1;
	}

	try	{
		fs::remove(lang_dir / "unk_temp.fst");
	}
	catch (const std::exception&){/*nothing*/}

	return 0;
}
