/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2011 Microsoft Corporation, Apache 2
*/

#include "kaldi-win\scr\kaldi_scr.h"

//converts an utt2spk file to a spk2utt file.
int utt2spk_to_spk2utt(StringTable & spk2utt, StringTable & utt2spk)
{
	//make sure that the output table is empty
	spk2utt.clear();
	string_vec speakers;
	using spk2utt2_map = std::unordered_map<std::string, string_vec>;
	spk2utt2_map hashtable;
	for (StringTable::const_iterator it(utt2spk.begin()), it_end(utt2spk.end()); it != it_end; ++it)
	{
		if ((*it).size() != 2) {
			LOGTW_ERROR << " There should be 2 columns in the utt2spk file!";
			return -1;
		}
		if (std::find(speakers.begin(), speakers.end(), (*it)[1]) == speakers.end())
		{ //not found, add it
			speakers.push_back((*it)[1]);
		}
		spk2utt2_map::iterator itm = hashtable.find((*it)[1]);
		if (itm == hashtable.end())
		{//did not find
			std::vector<std::string> uttvec;
			uttvec.push_back((*it)[0]);
			hashtable.emplace((*it)[1], uttvec);
		}
		else {
			itm->second.push_back((*it)[0]);
		}
	}
	for (string_vec::const_iterator it(speakers.begin()), it_end(speakers.end()); it != it_end; ++it)
	{
		spk2utt2_map::iterator itm = hashtable.find(*it);
		if (itm != hashtable.end()) {
			std::vector<std::string> _s;
			_s.push_back(*it);
			for (string_vec::const_iterator itu(itm->second.begin()), itu_end(itm->second.end()); itu != itu_end; ++itu)
			{
				_s.push_back(*itu);
			}
			spk2utt.push_back(_s);
		}
	}

	return 0;
}
