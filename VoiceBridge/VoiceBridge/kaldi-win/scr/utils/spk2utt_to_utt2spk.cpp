/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2011 Microsoft Corporation, Apache 2
*/

#include "kaldi-win\scr\kaldi_scr.h"

//NOTE: expects > 1 columns; the output utt2spk contains 2 columns! 
//		Each column in spk2utt (indx > 0) is mapped to the first field as key.
int spk2utt_to_utt2spk(StringTable & spk2utt, StringTable & utt2spk) 
{
	//make sure that the output table is empty
	utt2spk.clear();
	for (StringTable::const_iterator it(spk2utt.begin()), it_end(spk2utt.end()); it != it_end; ++it)
	{
		if ((*it).size() < 2) {
			LOGTW_ERROR << "Expected at least 2 columns in spk2utt but found " << (*it).size() << ".";
			return -1;
		}
		string_vec::const_iterator itc1(it->begin()), itc1_end(it->end());
		std::string s((*it)[0]);
		++itc1;
		for (itc1, itc1_end; itc1 != itc1_end; ++itc1) {
			std::vector<std::string> _v = { *itc1, s };
			utt2spk.push_back(_v);
		}
	}

	return 0;
}
