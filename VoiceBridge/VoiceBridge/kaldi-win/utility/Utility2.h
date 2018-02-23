/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/
#pragma once

#include "stdafx.h"
#include <kaldi-win/stdafx.h>

template <typename Map>
bool map_compare(Map const &lhs, Map const &rhs) {
	// No predicate needed because there is operator== for pairs already.
	return lhs.size() == rhs.size()
		&& std::equal(lhs.begin(), lhs.end(),
			rhs.begin());
}

int GetOptionsVector(std::string in, std::vector<std::string> & _ov);
int CheckIfFilesExist(std::vector<fs::path> _f);
int SaveOptionsToFile(fs::path p, std::vector<std::string> _o);

//VOICEBRIDGE_API std::string GetWindowsShortPath(fs::path p);
