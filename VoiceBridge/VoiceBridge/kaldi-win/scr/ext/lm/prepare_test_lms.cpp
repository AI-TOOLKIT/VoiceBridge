/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2012 Microsoft Corporation
		   Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"
#include "kaldi-win/scr/Params.h"


VOICEBRIDGE_API int PrepareTestLms(const std::vector<std::string> & lms)
{

	if (lms.size() < 1) {
		LOGTW_WARNING << "No language model is requested for testing.";
		return 0;
	}

	for each(std::string lm in lms) {

		fs::path path_test(voicebridgeParams.pth_data / ("lang_test_" + lm));
		//remove everything in the test dir
		try {
			if (fs::exists(path_test)) {
				fs::remove_all(path_test);
			}
			//copy lang dir
			if (!CopyDir(voicebridgeParams.pth_lang, path_test))
			{
				//try once more
				LOGTW_WARNING << "Failed to copy language data. Trying again...";
				SleepWait(1); //wait 1 second
				if (!CopyDir(voicebridgeParams.pth_lang, path_test)) {
					LOGTW_ERROR << "Failed to copy language data from " << (voicebridgeParams.pth_lang).string() << " to " << path_test.string();
					return -1;
				}
			}
		}
		catch (const std::exception&) {
			LOGTW_ERROR << "Failed to copy language data from " << (voicebridgeParams.pth_lang).string() << " to " << path_test.string();
			return -1;
		}

		//
		fs::path arpa_rxfilename(voicebridgeParams.pth_project_input / "task.arpabo");
		fs::path fst_wxfilename(path_test / "G.fst");
		fs::path read_syms_filename(path_test / "words.txt");
		if (arpa2fst(arpa_rxfilename.string(), fst_wxfilename.string(), "#0", read_syms_filename.string()) < 0) return -1;

		//DEBUG ----------------------------------
		//fst::FstInfo info;
		//if (fstinfo(fst_wxfilename.string(), "any", "auto", true, true, &info) < 0) return -1;
		//----------------------------------------

		//FST is stochastic?
		if (fstisstochastic(fst_wxfilename.string()) < 0) {
			LOGTW_WARNING << " FST is not stochastic!" << fst_wxfilename.string() << ".";
			//return -1;
		}

	}

	//DIAGNOSTICS:
	//Everything int this section is only for diagnostic. Checking that G has no cycles with empty words on them (e.g. <s>, </s>);
	//this might cause determinization failure of CLG. #0 is treated as an empty word.

	//NOTE: just check the first language model because they have all the same input data
	fs::path path_test(voicebridgeParams.pth_data / ("lang_test_" + lms[0]));
	fs::path fst_wxfilename(path_test / "G.fst");
	fs::path read_syms_filename(path_test / "words.txt");
	//setup temporary directory
	fs::path path_tempdir(voicebridgeParams.pth_project_base / "tempdir");
	try {
		fs::create_directory(path_tempdir);
	}
	catch (const std::exception&) {
		LOGTW_ERROR << "Can not create directory " << path_tempdir.string();
		return -1;
	}
	//
	StringTable table_lex;
	fs::path path_select_empty_fst_compiled(path_tempdir / "select_empty_fst_compiled.temp");
	if (ReadStringTable((voicebridgeParams.pth_project_input / "lexicon.txt").string(), table_lex) < 0) return -1;
	fs::ofstream file_emptyfst(path_tempdir / "select_empty.fst.txt", std::ios::binary);
	if (!file_emptyfst) {
		LOGTW_ERROR << "Can't open output file: " << (path_tempdir / "select_empty.fst.txt").string();
		return -1;
	}
	for (StringTable::const_iterator it(table_lex.begin()), it_end(table_lex.end()); it != it_end; ++it)
	{
		if ((*it).size() == 1) {
			file_emptyfst << "0 0 " << (*it)[0] << " " << (*it)[0] << "\n";
		}
	}
	file_emptyfst << "0 0 #0 #0\n";
	file_emptyfst << "0\n";
	file_emptyfst.flush(); file_emptyfst.close();
	try
	{
		if (fstcompile(false, "standard", "vector",
			(path_tempdir / "select_empty.fst.txt").string(), path_select_empty_fst_compiled.string(),
			read_syms_filename.string(), read_syms_filename.string(), "",
			false, false, false, false) < 0) return -1;
	}
	catch (const std::exception& message)
	{
		LOGTW_ERROR << "Error while compiling fst's, check file: " << (path_tempdir / "select_empty.fst.txt").string() << "\nDetail: " << message.what();
		return -1;
	}
	//sort the fst's and write the final output
	fs::path path_select_empty_FST(path_tempdir / "select_empty.fst");
	VectorFstClass * pofst = NULL;
	fst::FstInfo info;
	try {
		if (fstarcsort("olabel", path_select_empty_fst_compiled.string(), path_select_empty_FST.string()) < 0) return -1;
		if (fstcompose(path_select_empty_FST.string(), fst_wxfilename.string(), (path_tempdir / "select_empty_compose.fst").string(), pofst) < 0) return -1;
		if (fstinfo((path_tempdir / "select_empty_compose.fst").string(), "any", "auto", true, true, &info) < 0) return -1;
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << "Error while getting fst info." << " Detail: " << ex.what();
		return -1;
	}

	uint64 properties = info.Properties();
	//each property is set as a bit in properties
	//NOTE: properties are trinary - they are either true, false or unknown. For each such property, 
	//		there are two stored bits; one is set if true, the other is set if false and neither is set if unknown.
	//		Here: kOLabelSorted is set if true, kNotOLabelSorted is set if false, if neither is set then unkonwn!
	bool bIsCyclic = IsBitSet(properties, fst::kCyclic);
	if (bIsCyclic) {
		LOGTW_ERROR << "Language model has cycles with empty words.";
		return -1;
	}

	try	{
		if (fs::exists(voicebridgeParams.pth_project_base / "tempdir")) {
			fs::remove_all(voicebridgeParams.pth_project_base / "tempdir");
			fs::remove(voicebridgeParams.pth_project_base / "tempdir");
		}
	}
	catch (const std::exception&) {}

	LOGTW_INFO << "Language models for test preparation succeeded!";
	return 0;
}


