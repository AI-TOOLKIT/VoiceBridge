/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2015 Johns Hopkins University (Author: Yenda Trmal <jtrmal@gmail.com>), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"

static std::string rjustify(std::string str, int maxlen) {
	return str.insert(0, maxlen - str.length(), ' ');
}
static std::string ljustify(std::string str, int maxlen) {
	return str.append(maxlen - str.length(), ' ');
}

/*
	The program generates global statistic on how many time was each word recognized correctly, confused as another word, 
	incorrectly deleted or inserted. The output will contain similar info as the sclite dtl file, the format is,
	however, completely different.

	Input:
		UTT-A ref  word-A   <eps>  word-B  word-C  word-D  word-E
		UTT-A hyp  word-A  word-A  word-B   <eps>  word-D  word-X
	Output:
		correct       word-A  word-A  1
		correct       word-B  word-B  1
		correct       word-D  word-D  1
		deletion      word-C  <eps>   1
		insertion     <eps>   word-A  1
		substitution  word-E  word-X  1
*/
int WerOpsDetails(fs::path in,
	fs::path out,
	std::string special_symbol) 
{
	int max_size = 16;
	using MAPSI = std::map<std::string, int>;
	using UMAPSVS = std::unordered_map<std::string, std::vector<std::string>>;
	using UMAPSS = std::unordered_map<std::string, std::string>;
	using MAPSMAPSI = std::map<std::string, MAPSI>; //NOTE: automatically sorted by the first and also by second key, not unique keys!
	using MAPSUMAPSVS = std::map<std::string, UMAPSVS>; //NOTE: automatically sorted by the first key, not unique keys!
	MAPSMAPSI EDIT_OPS;
	MAPSUMAPSVS UTT;

	StringTable t_in;
	if (ReadStringTable(in.string(), t_in) < 0) return -1;
	//prepare output file
	fs::ofstream f_out(out, fs::ofstream::binary | fs::ofstream::out);
	if (!f_out) {
		LOGTW_ERROR << "Could not write to file " << out.string();
		return -1;
	}

	for (StringTable::const_iterator it(t_in.begin()), it_end(t_in.end()); it != it_end; ++it) 
	{
		if ((*it).size()<2 || ((*it)[1] != "hyp" && (*it)[1] != "ref")) continue;

		std::string it0 = ConvertToUTF8((*it)[0]);
		std::string it1 = ConvertToUTF8((*it)[1]);

		MAPSUMAPSVS::iterator itm = UTT.find(it0);
		if (itm != UTT.end()) {
			UMAPSVS::iterator itm1 = itm->second.find(it1);
			if (itm1 != itm->second.end()) {
				LOGTW_ERROR << "The input stream contains duplicate entry " << it0 << " " << it1;
				return -1;
			}
		}
		int c = 0;
		std::vector<std::string> _v;
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) 
		{
			if (c < 2) { c++; continue; } //skip the first two columns
			_v.push_back(ConvertToUTF8(*itc1));
		}
		//
		if (itm != UTT.end()) 
		{//it exist already, just add the new element to it, do not overwrite it
			itm->second.emplace(it1, _v );
		}
		else {
			UTT.emplace(it0, UMAPSVS({ { it1, _v } }));
		}
	}

	for (auto & pair : UTT)
	{
		std::string utterance(pair.first);
		UMAPSVS::iterator itm = pair.second.find("hyp");
		if (itm == pair.second.end()) {
			LOGTW_ERROR << "The input stream does not contain entry 'hyp' for utterance " << utterance;
			return -1;
		}
		itm = pair.second.find("ref");
		if (itm == pair.second.end()) {
			LOGTW_ERROR << "The input stream does not contain entry 'ref' for utterance " << utterance;
			return -1;
		}

		string_vec _hyp = UTT[utterance]["hyp"];
		string_vec _ref = UTT[utterance]["ref"];
		if (_hyp.size() != _ref.size()) {
			LOGTW_ERROR << "The 'ref' an 'hyp' entries do not have the same number of fields.";
			return -1;
		}

		for (int i = 0; i < _hyp.size(); i++) 
		{
			//allocate memory if it does not exit
			MAPSMAPSI::iterator itm = EDIT_OPS.find(_ref[i]);
			if (itm == EDIT_OPS.end()) {
				EDIT_OPS.emplace(_ref[i], MAPSI({ {_hyp[i], 0} }));
			}
			else {
				MAPSI::iterator itm1 = itm->second.find(_hyp[i]);
				if (itm1 == itm->second.end()) {
					itm->second.emplace(_hyp[i], 0);
				}
			}
			//
			EDIT_OPS[_ref[i]][_hyp[i]] += 1;
		}
	}

	int word_len = 0;
	int ops_len = 0;
	for (auto & pairRef : EDIT_OPS) {
		std::string refw(pairRef.first);
		
		for (auto & pairHyp : EDIT_OPS[refw]) {
			std::string hypw(pairHyp.first);

			int q = refw.length() > hypw.length() ? refw.length() : hypw.length();
			if (q > max_size) {
				//DEBUG: refw, hypw, q
				//LOGTW_INFO << refw.length() << hypw.length() << q;
			}
			word_len = q > word_len ? q : word_len;

			int d = (to_string_with_precision<int>((EDIT_OPS[refw][hypw]), 0)).length();

			ops_len = d > ops_len ? d : ops_len;
		}
	}

	if (word_len > max_size) {
		// "Affects only whitespace: we are limiting the width to " << max_size << ", max word len was " << word_len;
		word_len = max_size;
	}
	//to the screen:
	for (auto & pairRef : EDIT_OPS) {
		std::string refw(pairRef.first);
		for (auto & pairHyp : EDIT_OPS[refw]) {
			std::string hypw(pairHyp.first);
			if (refw == hypw) {
				LOGTW_INFO << "correct       " << ljustify(refw, word_len) << "    " << ljustify(hypw, word_len) << " " << ljustify(to_string_with_precision<int>(EDIT_OPS[refw][hypw], 0), ops_len);
			} else if(refw == special_symbol) {
				LOGTW_INFO << "insertion     " << ljustify(refw, word_len) << "    " << ljustify(hypw, word_len) << " " << ljustify(to_string_with_precision<int>(EDIT_OPS[refw][hypw], 0), ops_len);
			} else if(hypw == special_symbol) {
				LOGTW_INFO << "deletion      " << ljustify(refw, word_len) << "    " << ljustify(hypw, word_len) << " " << ljustify(to_string_with_precision<int>(EDIT_OPS[refw][hypw], 0), ops_len);
			}
			else {
				LOGTW_INFO << "substitution  " << ljustify(refw, word_len) << "    " << ljustify(hypw, word_len) << " " << ljustify(to_string_with_precision<int>(EDIT_OPS[refw][hypw], 0), ops_len);
			}
		}
	}
	//to output file:
	for (auto & pairRef : EDIT_OPS) {
		std::string refw(pairRef.first);
		for (auto & pairHyp : EDIT_OPS[refw]) {
			std::string hypw(pairHyp.first);
			if (refw == hypw) {
				f_out << "correct       " << ljustify(refw, word_len) << "    " << ljustify(hypw, word_len) << " " << ljustify(to_string_with_precision<int>(EDIT_OPS[refw][hypw], 0), ops_len) << "\n";
			}
			else if (refw == special_symbol) {
				f_out << "insertion     " << ljustify(refw, word_len) << "    " << ljustify(hypw, word_len) << " " << ljustify(to_string_with_precision<int>(EDIT_OPS[refw][hypw], 0), ops_len) << "\n";
			}
			else if (hypw == special_symbol) {
				f_out << "deletion      " << ljustify(refw, word_len) << "    " << ljustify(hypw, word_len) << " " << ljustify(to_string_with_precision<int>(EDIT_OPS[refw][hypw], 0), ops_len) << "\n";
			}
			else {
				f_out << "substitution  " << ljustify(refw, word_len) << "    " << ljustify(hypw, word_len) << " " << ljustify(to_string_with_precision<int>(EDIT_OPS[refw][hypw], 0), ops_len) << "\n";
			}
		}
	}

	f_out.flush(); f_out.close();
	return 0;
}
