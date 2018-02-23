/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/
#pragma once

#include "kaldi_scr.h"

namespace VoiceBridge {

	/*
		The Params class holds all path and name parameters for VoiceBridge for easy access on global level.
	*/
	class Params
	{
	public:
		Params(void);
		VOICEBRIDGE_API bool Init(std::string strain_base_name, std::string stest_base_name,
			std::string sproject_base_dir, std::string sproject_input_dir,
			std::string swaves_dir, std::string soov_word = "<SIL>");

		//publicly accesible paths
		fs::path pth_project_base;
		fs::path pth_project_input;
		fs::path pth_local;
		fs::path pth_data;
		fs::path pth_dict;
		fs::path pth_lang;
		fs::path pth_silence_phones_txt;
		fs::path pth_optional_silence_txt;
		fs::path pth_nonsilence_phones_txt;
		fs::path pth_lexicon_txt;
		fs::path pth_lexiconp_txt;
		fs::path pth_lexiconp_silprob_txt;
		fs::path pth_silprob_txt;
		fs::path pth_extra_questions_txt;
		//
		std::string train_base_name;
		std::string test_base_name;
		std::string project_base_dir;
		std::string project_input_dir;
		std::string waves_dir;
		std::string oov_word;
		std::string task_arpabo_name; //fixed 
		std::string phones_txt_name; //fixed 
	};
}

extern VOICEBRIDGE_API VoiceBridge::Params voicebridgeParams;
