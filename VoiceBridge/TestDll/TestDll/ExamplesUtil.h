/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/
#pragma once

#include "stdafx.h"
#include "VoiceBridge.h"

extern int concurentThreadsSupported;

bool NeedToRetrainModel(fs::path project_model_dir, 
	fs::path project_input_dir, 
	fs::path project_waves_dir, 
	fs::path project_config, //the config file for the model training
	fs::path project_base_model_dir=""); //the model dir on which the final.mdl in the project_model_dir depends on; if not define then not checked
bool NeedToDecode(fs::path model_dir, fs::path decode_dir);
