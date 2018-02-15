/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/

#pragma once
#include <kaldi-win/stdafx.h>
#include "kaldi-win/utility/Utility.h"
#include "kaldi-win/src/fstbin/fst_ext.h"

using string_vec = std::vector<std::string>;
using UMAPSS = std::unordered_map<std::string, std::string>;
using MSYMLINK = std::map<std::string, std::string>;

VOICEBRIDGE_API int MakeMfccPitch(
	fs::path datadir,						//data directory
	fs::path mfcc_config,					//mfcc config file path
	fs::path pitch_config,					//pitch config file path (pitch.conf)
	fs::path pitch_postprocess_config="",	//pitch postprocess config file path (pitch.conf)
	int nj = 4,								//default: 4, number of parallel jobs
	bool compress = true,					//default: true, compress mfcc features
	bool write_utt2num_frames = false,		//default: false, if true writes utt2num_frames	
	int paste_length_tolerance = 2			//default: 2
);

