/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/
#include "Params.h"

VOICEBRIDGE_API VoiceBridge::Params voicebridgeParams;

VoiceBridge::Params::Params(void) {};

VOICEBRIDGE_API bool VoiceBridge::Params::Init(std::string strain_base_name, std::string stest_base_name,
	std::string sproject_base_dir, std::string sproject_input_dir,
	std::string swaves_dir, std::string soov_word)
{
	//adjustable
	train_base_name = strain_base_name;
	test_base_name = stest_base_name;
	project_base_dir = sproject_base_dir;
	project_input_dir = sproject_input_dir;
	waves_dir = swaves_dir;
	oov_word = soov_word;
	//fixed
	task_arpabo_name = "task.arpabo";
	phones_txt_name = "phones.txt";
	//init all paths
	pth_project_base = fs::path(project_base_dir, fs::native);
	pth_project_input = fs::path(project_input_dir, fs::native);

	//basic check
	if (!fs::exists(pth_project_base)) return false;
	if (!fs::exists(pth_project_input)) return false;

	pth_local = (pth_project_base / "data/local");
	pth_data = (pth_project_base / "data");
	pth_dict = (pth_local / "dict");
	pth_lang = (pth_data / "lang");
	pth_silence_phones_txt = (pth_dict / "silence_phones.txt");
	pth_optional_silence_txt = (pth_dict / "optional_silence.txt");
	pth_nonsilence_phones_txt = (pth_dict / "nonsilence_phones.txt");
	pth_lexicon_txt = (pth_dict / "lexicon.txt");
	pth_lexiconp_txt = (pth_dict / "lexiconp.txt");
	pth_lexiconp_silprob_txt = (pth_dict / "lexiconp_silprob.txt");
	pth_silprob_txt = (pth_dict / "silprob.txt");
	pth_extra_questions_txt = (pth_dict / "extra_questions.txt");

	//redirects std::cerr to LOGTW_INFO in order to see the errors sent to std::cerr by OpenFst!
	const redirect_outputs _RO(LOGTW_ERROR, std::cerr);
	//BindStdHandlesToConsole();

	//redirect all Kaldi messages to the global twin logging module
	ReplaceKaldiLogHandlerEx(true);

	return true;
};
