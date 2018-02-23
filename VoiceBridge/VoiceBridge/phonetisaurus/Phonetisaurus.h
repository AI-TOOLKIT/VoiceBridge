#pragma once
/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
	Based on Phonetisaurus: Copyright (c) [2012-], Josef Robert Novak, All rights reserved.
*/

#include "stdafx.h"
#include "..\kaldi-win\stdafx.h"

namespace Phonetisaurus 
{
	VOICEBRIDGE_API int TrainModel(fs::path refDictionary, fs::path outModel, int ngramOrder=6);
	VOICEBRIDGE_API int GetPronunciation(fs::path refDictionary, fs::path model, fs::path intxtfile, fs::path outtxtfile, bool forceModel=true);
}

