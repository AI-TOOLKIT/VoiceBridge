/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
Based on: see below
*/
#pragma once

#include "stdafx.h"
#include "kaldi-win/utility/Utility.h"

VOICEBRIDGE_API int EvaluateNgram(int argc, char* argv[]);
VOICEBRIDGE_API int EstimateNgram(int argc, char* argv[]);
VOICEBRIDGE_API int InterpolateNgram(int argc, char* argv[]);
