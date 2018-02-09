/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"
#include "kaldi-win/scr/Params.h"

using distinguished_map = std::multimap<std::string, std::string>;

int CheckLexiconPair(fs::path lex1, int num_prob_cols1, int num_skipped_cols1, fs::path lex2, int num_prob_cols2, int num_skipped_cols2);
int CheckExtraQuestions(fs::path lp, distinguished_map & distinguished);

//external parameters
//NOTE: _silencephones, _nonsilencephones must be read before use with: ReadSilenceAndNonSilencePhones
static std::vector<std::string> * _silencephones = NULL;
static std::vector<std::string> * _nonsilencephones = NULL;
//

/*
	Validate the dictionary
*/
int ValidateDict(fs::path pthdict)
{
	//replace all voicebridgeParams constants with these:
	fs::path pth_silence_phones_txt = (pthdict / "silence_phones.txt");
	fs::path pth_optional_silence_txt = (pthdict / "optional_silence.txt");
	fs::path pth_nonsilence_phones_txt = (pthdict / "nonsilence_phones.txt");
	fs::path pth_lexicon_txt = (pthdict / "lexicon.txt");
	fs::path pth_lexiconp_txt = (pthdict / "lexiconp.txt");
	fs::path pth_lexiconp_silprob_txt = (pthdict / "lexiconp_silprob.txt");
	fs::path pth_silprob_txt = (pthdict / "silprob.txt");
	fs::path pth_extra_questions_txt = (pthdict / "extra_questions.txt");	

	//NOTE: _silencephones and _nonsilencephones are filled in below in CheckDuplicates()!
	_silencephones = Get_silencephones();
	_nonsilencephones = Get_nonsilencephones();

	//Checking silence_phones.txt -------------------------------
	if (CheckFileExistsAndNotEmpty(pth_silence_phones_txt, true) < 0) return -1;
	if (CheckUTF8AndWhiteSpace(pth_silence_phones_txt, true) < 0) return -1;
	if (CheckDuplicates(pth_silence_phones_txt, true, *_silencephones) < 0) return -1;
	LOGTW_INFO << pth_silence_phones_txt.string() << " is OK!";

	//Checking optional_silence.txt------------------------------
	if (CheckFileExistsAndNotEmpty(pth_optional_silence_txt, true) < 0) return -1;
	if (CheckUTF8AndWhiteSpace(pth_optional_silence_txt, true) < 0) return -1;
	StringTable txt = readData(pth_optional_silence_txt.string());
	if (txt.size() < 1) {
		LOGTW_ERROR << " phone not found in " << pth_optional_silence_txt.string() << ".";
		return -1;
	}
	else if (txt.size() > 1 || txt[0].size()>1) {
		LOGTW_ERROR << " only 1 phone is expected in " << pth_optional_silence_txt.string() << ".";
		return -1;
	}
	LOGTW_INFO << pth_optional_silence_txt.string() << " is OK!";

	//Checking nonsilence_phones.txt -------------------------------
	if (CheckFileExistsAndNotEmpty(pth_nonsilence_phones_txt, true) < 0) return -1;
	if (CheckUTF8AndWhiteSpace(pth_nonsilence_phones_txt, true) < 0) return -1;
	if (CheckDuplicates(pth_nonsilence_phones_txt, true, *_nonsilencephones) < 0) return -1;
	LOGTW_INFO << pth_nonsilence_phones_txt.string() << " is OK!";

	//Checking disjoint -------------------------------
	//Checking disjoint: silence_phones.txt, nonsilence_phones.txt
	if (!IsDisjoint(*_silencephones, *_nonsilencephones)) {
		LOGTW_ERROR << " silence_phones.txt and nonsilence_phones.txt has overlap.";
		return -1;
	}
	LOGTW_INFO << pth_silence_phones_txt.string() << " and\n" << pth_nonsilence_phones_txt.string() << " do not overlap! OK!";

	//check_lexicon
	bool bLexicon = false;
	if (fs::exists(pth_lexicon_txt)) {
		if (CheckLexicon(pth_lexicon_txt, 0, 0) < 0) return -1;
		bLexicon = true;
	}
	if (fs::exists(pth_lexiconp_txt)) {
		if (CheckLexicon(pth_lexiconp_txt, 1, 0) < 0) return -1;
		bLexicon = true;
	}
	if (fs::exists(pth_lexiconp_silprob_txt)) {
		//NOTE: If dict / lexiconp_silprob.txt exists, we expect dict / silprob.txt to also exist!
		if (!fs::exists(pth_silprob_txt)) {
			LOGTW_ERROR << "Failed to open " << pth_silprob_txt.string() << ".";
			return -1;
		}
		if (CheckLexicon(pth_lexiconp_silprob_txt, 2, 2) < 0) return -1;
		if (CheckSilprob(pth_silprob_txt) < 0) return -1;
		//there must be at least one lexicon
		if (!bLexicon) {
			LOGTW_ERROR << "Neither lexicon.txt or lexiconp.txt exist in directory " << pth_lexicon_txt.branch_path() << ".";
			return -1;
		}
		//If more than one lexicon exist, we have to check if they correspond to each other
		//It could be that the user overwrote one and we need to regenerate the other, but we do not know which is which.
		if (fs::exists(pth_lexicon_txt) && fs::exists(pth_lexiconp_txt)) {
			if (CheckLexiconPair(pth_lexicon_txt, 0, 0, pth_lexiconp_txt, 1, 0) < 0) return -1;
		}
		if (fs::exists(pth_lexiconp_txt) && fs::exists(pth_lexiconp_silprob_txt)) {
			if (CheckLexiconPair(pth_lexiconp_txt, 1, 0, pth_lexiconp_silprob_txt, 2, 2) < 0) return -1;
		}
	}

	//Checking extra_questions.txt -------------------------------
	/*
	Keep track of all phone-pairs including nonsilence that are distinguished (split apart) by extra_questions.txt,
	as distinguished[p1,p2] = 1.  This will be used to make sure that we don't have pairs of phones on the same
	line in nonsilence_phones.txt that can never be distinguished from each other by questions.  (If any two
	phones appear on the same line in nonsilence_phones.txt, they share a tree root, and since the automatic
	question-building treats all phones that appear on the same line of nonsilence_phones.txt as being in the same
	group, we can never distinguish them without resorting to questions in extra_questions.txt.
	*/
	distinguished_map distinguished;
	if (fs::exists(pth_extra_questions_txt) && !fs::is_empty(pth_extra_questions_txt))
	{
		if (CheckExtraQuestions(pth_extra_questions_txt, distinguished) < 0) return -1;
		LOGTW_INFO << "  --> extra_questions.txt is validated with succes!";
	}

	/*
	check nonsilence_phones.txt again for phone-pairs that are never distnguishable.
	(note: this situation is normal and expected for silence phones, so we don't check it.)
	*/
	int num_warn_nosplit = 0;
	StringTable nsp_txt = readData(pth_nonsilence_phones_txt.string());
	for (StringTable::const_iterator it(nsp_txt.begin()), it_end(nsp_txt.end()); it != it_end; ++it)
	{
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
		{
			for (string_vec::const_iterator itc2(it->begin()), itc2_end(it->end()); itc2 != itc2_end; ++itc2) {
				std::string s1(*itc1);
				std::string s2(*itc2);

				if (s1 != s2) {					
					//there can be several items with the same key, do not get only the first one but all of them!
					std::pair <distinguished_map::iterator, distinguished_map::iterator> ret;
					ret = distinguished.equal_range(s1);
					bool bFound = false;
					for (distinguished_map::iterator it = ret.first; it != ret.second; ++it) {
						if (s2 == it->second) {
							bFound = true;
							break;
						}
					}
					if (!bFound) //NOTE: not found in distinguished!
					{
						LOGTW_WARNING << " phones " << s1 << " and " << s2 << " share a tree root but can never be distinguished by extra_questions.txt.";
						num_warn_nosplit++;
					}
				}
			}
		}
	}

	if (num_warn_nosplit > 0) {
		LOGTW_WARNING << "NOTE WARNINGS ABOVE: You can build a system with this setup but some phones will be acoustically indistinguishable!";
	}

	return 0;
}

//validate extra_questions.txt and store distinguished
int CheckExtraQuestions(fs::path lp, distinguished_map & distinguished)
{
	_silencephones = Get_silencephones();
	_nonsilencephones = Get_nonsilencephones();
	//NOTE: ValidateDict() or ReadSilenceAndNonSilencePhones() must be first called in order to read in _silencephones, _nonsilencephones!
	if (_silencephones->size() < 1 || _nonsilencephones->size() < 1) {
#ifdef _DEBUG
		LOGTW_ERROR << "***DEVELOPER NOTE: ValidateDict() or ReadSilenceAndNonSilencePhones() must be first called in order to read in _silencephones, _nonsilencephones!";
#endif // DEBUG
		LOGTW_ERROR << "No silence and/or non-silence phones detected.";
		return -1;
	}

	if (CheckUTF8AndWhiteSpace(lp, true) < 0) return -1;
	if (CheckLexicon(lp, 0, 0) < 0) return -1;
	StringTable exqs = readData(lp.string());
	std::vector<std::string> _col_hash;

	for (StringTable::const_iterator it(exqs.begin()), it_end(exqs.end()); it != it_end; ++it)
	{
		_col_hash.clear();
		for (string_vec::const_iterator itc(it->begin()), itc_end(it->end()); itc != itc_end; ++itc)
		{
			std::string s(*itc);
			_col_hash.push_back(s);
		}
		for (string_vec::const_iterator itc(it->begin()), itc_end(it->end()); itc != itc_end; ++itc)
		{
			// Update distinguished hash.
			for (std::string& s : *_nonsilencephones) {
				if (std::find(_col_hash.begin(), _col_hash.end(), s) == _col_hash.end())
				{//not found
				 // for each p1 in this question and p2 not in this question(and in nonsilence phones)... mark p1, p2 as being split apart
					distinguished.emplace(*itc, s);
					distinguished.emplace(s, *itc);
				}
			}
		}
	}

	return 0;
}


//check if the two lexicons are compatible
int CheckLexiconPair(fs::path plex1, int num_prob_cols1, int num_skipped_cols1, fs::path plex2, int num_prob_cols2, int num_skipped_cols2)
{
	StringTable lex1 = readData(plex1.string());
	StringTable lex2 = readData(plex2.string());
	if (lex1.size() != lex2.size()) {
		LOGTW_ERROR << " " << plex1.string() << " and " << plex2.string() << " have different number of lines.";
		return -1;
	}
	for (int r = 0; r<lex1.size(); r++)
	{
		for (int c = 0; c<lex1[r].size(); c++)
		{
			if (lex1[r][0] != lex2[r][0]) {
				LOGTW_ERROR << " " << plex1.string() << " and " << plex2.string() << " mismatch at line " << (r + 1) << ". Sorting?";
				return -1;
			}
		}
		std::string slex1, slex2;
		for (int n = num_prob_cols1 + num_skipped_cols1; n < lex1[r].size(); n++)
		{
			slex1.append(lex1[r][n]);
		}
		for (int n = num_prob_cols2 + num_skipped_cols2; n < lex2[r].size(); n++)
		{
			slex2.append(lex2[r][n]);
		}
		if (slex1 != slex2) {
			LOGTW_ERROR << " " << plex1.string() << " and " << plex2.string() << " mismatch at line " << (r + 1) << ". Sorting?";
			return -1;
		}
	}

	return 0;
}
