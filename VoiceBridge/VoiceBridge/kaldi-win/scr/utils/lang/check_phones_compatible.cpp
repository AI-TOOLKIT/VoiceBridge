/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2016 Hang Lyu, Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"
#include "kaldi-win\utility\Utility2.h"

/*
	checks if two phones tables are the same or not (except for possible difference in disambiguation symbols)
*/
int CheckPhonesCompatible(fs::path table_first, fs::path table_second)
{
	if (!fs::exists(table_first)) {
		LOGTW_ERROR << "File does not exist: " << table_first.string();
		return -1;
	}
	if (!fs::exists(table_second)) {
		LOGTW_ERROR << "File does not exist: " << table_second.string();
		return -1;
	}
	std::map<std::string, std::string> t_1, t_2; //automatically sorted and unique
	StringTable st1, st2;
	if (ReadStringTable(table_first.string(), st1) < 0) {
		LOGTW_ERROR << "Can not read file: " << table_first.string();
		return -1;
	}
	if (ReadStringTable(table_second.string(), st2) < 0) {
		LOGTW_ERROR << "Can not read file: " << table_second.string();
		return -1;
	}
	for (std::vector<std::string> _v : st1) {
		if (_v.size() != 2) {
			LOGTW_ERROR << "Wrong file format: " << table_first.string();
			return -1;
		}
		if(_v[0].find("#") == std::string::npos) //do not compare disambiguation symbols
			t_1.emplace(_v[0], _v[1]);
	}
	for (std::vector<std::string> _v : st2) {
		if (_v.size() != 2) {
			LOGTW_ERROR << "Wrong file format: " << table_second.string();
			return -1;
		}
		if (_v[0].find("#") == std::string::npos) //do not compare disambiguation symbols
			t_2.emplace(_v[0], _v[1]);
	}

	//compare the two maps
	if(!map_compare(t_1, t_2)) {
		LOGTW_ERROR << "Phone symbol tables " << table_first.string() << " and " << table_second.string() << " are not compatible.";
		return -1;
	}

	return 0;
}
