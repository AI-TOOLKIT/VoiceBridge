/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/

#include "kaldi-win\scr\kaldi_scr.h"

//NOTE: these variables are used in several compilation units and therefore must be accessed with the below accessor functions
//		call ReadSilenceAndNonSilencePhones() before using the variable and access them with the two pointer functions
static std::vector<std::string> _silencephones; 
static std::vector<std::string> _nonsilencephones;
std::vector<std::string> * Get_silencephones() { return &_silencephones; }
std::vector<std::string> * Get_nonsilencephones() {	return &_nonsilencephones; }

//returns -2 if error; -1 if it does not exist; returns 0 if everything is OK (exists and not empty)
int CheckFileExistsAndNotEmpty(fs::path file, bool bShowError) 
{
	try
	{
		if (!fs::exists(file) || fs::is_empty(file))
		{
			if(bShowError)
				LOGTW_ERROR << " the file " << file.string() << " does not exist or empty.";
			//else LOGTW_INFO << " the file " << file.string() << " does not exist or empty.";
			//NOTE: removed info about missing file because it can be confusing. If bShowError=false then it does not matter.
			return -1;
		}
	}
	catch (std::exception const& e)
	{
		LOGTW_FATALERROR << " " << e.what() << ".";
		return -2;
	}
	catch (...)
	{
		LOGTW_FATALERROR << " Unknown Error.";
		return -2;
	}
	return 0;
}

//check if valid strings in text file
int CheckUTF8AndWhiteSpace(fs::path _path, bool check_last_char_is_nl)
{
	try {
		StringTable txt = readData(_path.string());
		for (StringTable::const_iterator it(txt.begin()), it_end(txt.end()); it != it_end; ++it)
		{
			for (string_vec::const_iterator itc(it->begin()), itc_end(it->end()); itc != itc_end; ++itc)
			{
				std::string s = ConvertToUTF8(*itc);
				if (s.empty() || !validate_utf8_whitespaces(s)) {
					LOGTW_ERROR << " " << _path.string() << " contains disallowed UTF-8 whitespace character(s) or not UTF-8 compatible.";
					return -1;
				}
			}
		}
		if (check_last_char_is_nl == true)
		{
			fs::ifstream ifs(_path);
			if (ifs.is_open()) {
				ifs.seekg(-1, std::ios::end);
				char *t = new char[1];
				ifs.read(t, 1);
				if (*t != '\n') {
					LOGTW_ERROR << " " << _path.string() << " does not end in newline.";
					return -1;
				}
			}
		}
	}
	catch (std::exception const& e)
	{
		LOGTW_FATALERROR << " " << e.what() << ".";
		return -1;
	}
	catch (...)
	{
		LOGTW_FATALERROR << " Unknown Error.";
		return -1;
	}
	return 0;
}


//checks if every string in the text file is unique and puts all strings in the _v vector
//if check_valid_ending == true it check for some special character endings
// (disambiguation symbols; phones ending in _B, _E, _S or _I will cause problems with word - position - dependent systems in silence_phones.txt)
int CheckDuplicates(fs::path _path, bool check_valid_ending, std::vector<std::string> & _v)
{
	try
	{
		StringTable txt = readData(_path.string());
		for (StringTable::const_iterator it(txt.begin()), it_end(txt.end()); it != it_end; ++it)
		{
			for (string_vec::const_iterator itc(it->begin()), itc_end(it->end()); itc != itc_end; ++itc)
			{
				std::string s(*itc);
				if (check_valid_ending && s.length() > 1) {
					std::string ss = s.substr(s.length() - 2);
					if (ss == "_B" || ss == "_E" || ss == "_S" || ss == "_I")
					{
						LOGTW_ERROR << " the characters _B, _E, _S and _I are not allowed as last characters in " << _path.string() << ".";
						return -1;
					}
				}
				_v.push_back(s);
			}
		}
		if (!is_unique(_v)) {
			LOGTW_ERROR << " duplicates in " << _path.string() << ".";
			return -1;
		}
	}
	catch (std::exception const& e)
	{
		LOGTW_FATALERROR << " " << e.what() << ".";
		return -1;
	}
	catch (...)
	{
		LOGTW_FATALERROR << " Unknown Error.";
		return -1;
	}
	return 0;
}


//checks the lexicons for consistency
int CheckLexicon(fs::path lp, int num_prob_cols, int num_skipped_cols)
{
	//NOTE: ValidateDict() or ReadSilenceAndNonSilencePhones() must be first called in order to read in _silencephones, _nonsilencephones!
	if (_silencephones.size() < 1 || _nonsilencephones.size() < 1) {
#ifdef _DEBUG
		LOGTW_ERROR << "***DEVELOPER NOTE: ValidateDict() or ReadSilenceAndNonSilencePhones() must be first called in order to read in _silencephones, _nonsilencephones!";
#endif // DEBUG
		LOGTW_ERROR << " no silence and/or non-silence phones detected.";
		return -1;
	}

	if (CheckUTF8AndWhiteSpace(lp, true) < 0) return -1;

	std::ifstream ifs(lp.string());
	if (!ifs) {
		LOGTW_ERROR << " Error opening file: " << lp.string() << ".";
		return -1;
	}
	std::string line;
	std::vector<std::string> _vLines;
	int nLine = 0, nextCol = 0;
	while (std::getline(ifs, line)) {
		nextCol = 0;
		nLine++;

		//duplicate:
		if (std::find(_vLines.begin(), _vLines.end(), line) == _vLines.end()) {
			_vLines.push_back(line);
		}
		else {
			LOGTW_ERROR << " Duplicate lines in file: " << lp.string() << ".";
			return -1;
		}

		//extract the columns with strtk and check each word
		std::vector<std::string> _words;
		strtk::parse(line, " \t", _words, strtk::split_options::compress_delimiters);

		if (_words.size() < 1) {
			LOGTW_ERROR << " empty lexicon line in file: " << lp.string() << ".";
			return -1;
		}

		//forbidden word: 
		if (_words[0].find("<s>", 0) != std::string::npos ||
			_words[0].find("</s>", 0) != std::string::npos ||
			_words[0].find("<eps>", 0) != std::string::npos ||
			_words[0].find("#0", 0) != std::string::npos)
		{
			LOGTW_ERROR << " Forbidden word in " << _words[0] << " (<s>, </s>, <eps>, #0) in file: " << lp.string() << ".";
			return -1;
		}

		//the first column is the <> tag
		nextCol++;
		int __num_prob_cols = num_prob_cols + nextCol;
		for (int n = nextCol; n < __num_prob_cols; n++) {
			double d = std::stod(_words[n].c_str());
			if (!(d > 0.0 && d <= 1.0)) {
				LOGTW_ERROR << " bad pron-prob in lexicon-line " << nLine << "in file: " << lp.string() << ".";
				return -1;
			}
			nextCol++;
		}
		int __num_skipped_cols = num_skipped_cols + nextCol;
		for (int n = nextCol; n < __num_skipped_cols; n++) { nextCol++; }
		if (_words.size() < nextCol + 1) {
			LOGTW_ERROR << " " << lp.string() << " contains word " << _words[0] << " with empty pronunciation.";
			return -1;
		}

		//check if the word is either in silence.txt or in nonsilence.txt
		for (int n = nextCol; n < _words.size(); n++)
		{
			if (std::find(_silencephones.begin(), _silencephones.end(), _words[n]) == _silencephones.end() &&
				std::find(_nonsilencephones.begin(), _nonsilencephones.end(), _words[n]) == _nonsilencephones.end())
			{ //not found, add it
				LOGTW_ERROR << " " << _words[n] << " is not in silence.txt and neither in nonsilence.txt.";
				return -1;
			}
		}
	}

	return 0;
}


//checks silprob.txt for consistency
int CheckSilprob(fs::path lp)
{
	//NOTE: not checking for Carriage Return (^M) characters in the C++ version

	if (CheckUTF8AndWhiteSpace(lp, true) < 0) return -1;

	std::ifstream ifs(lp.string());
	if (!ifs) {
		LOGTW_ERROR << " Error opening file: " << lp.string() << ".";
		return -1;
	}
	std::string line;
	//std::vector<std::string> _vLines;
	int nLine = 0, nextCol = 0;
	while (std::getline(ifs, line)) {
		nextCol = 0;
		nLine++;

		//NOTE: duplicate checking - probably not needed and maybe would not be good
		//duplicate:
		//if (std::find(_vLines.begin(), _vLines.end(), line) == _vLines.end()) {
		//	_vLines.push_back(line);
		//}
		//else {
		//	LOGTW_ERROR << " Duplicate lines in file: " << lp.string() << ".";
		//	return -1;
		//}

		//extract the columns with strtk and check each word
		std::vector<std::string> _words;
		strtk::parse(line, " \t", _words, strtk::split_options::compress_delimiters);

		if (_words.size() != 2) {
			LOGTW_ERROR << " bad line (" << nLine << ") in file: " << lp.string() << ".";
			return -1;
		}
		double d = std::stod(_words[1].c_str());
		if (_words[0] == "<s>" || _words[0] == "overall") {
			if (!(d > 0.0 && d <= 1.0)) {
				LOGTW_ERROR << " bad probability at line " << nLine << " in file: " << lp.string() << ".";
				return -1;
			}
		}
		else if (_words[0] == "</s>_s" || _words[0] == "</s>_n") {
			if (d <= 0.0) {
				LOGTW_ERROR << " bad correction term at line " << nLine << " in file: " << lp.string() << ".";
				return -1;
			}
		}
		else {
			LOGTW_ERROR << " unexpected line " << nLine << " in file: " << lp.string() << ".";
			return -1;
		}
	}

	return 0;
}

//Return the integer value from the second field of the StringTable if the first field matches the supplied key.
//The return value is placed into 'val' (pointer).
//return -1 when not found; display error message
int GetSecondFieldFromStringTable(StringTable table, std::string key, std::string path, int * val)
{
	for (StringTable::const_iterator it(table.begin()), it_end(table.end()); it != it_end; ++it)
	{
		if ((*it)[0] == key)
		{
			*val = std::stoi((*it)[1]);
			return 0;
		}
	}
	LOGTW_ERROR << " key " << key << " not found in table in file: " << path << ".";
	return -1;
}

int ReadSilenceAndNonSilencePhones(fs::path fSilencephones, fs::path fNonSilencephones)
{
	//NOTE: these files have one word per line.
	StringTable t_silencephones, t_nonsilencephones;
	if (ReadStringTable(fSilencephones.string(), t_silencephones) < 0) {
		LOGTW_ERROR << " could not read file: " << fSilencephones.string() << ".";
		return -1;
	}
	if (ReadStringTable(fNonSilencephones.string(), t_nonsilencephones) < 0) {
		LOGTW_ERROR << " could not read file: " << fNonSilencephones.string() << ".";
		return -1;
	}
	_silencephones.clear();
	_nonsilencephones.clear();
	for (StringTable::const_iterator it(t_silencephones.begin()), it_end(t_silencephones.end()); it != it_end; ++it)
		_silencephones.push_back((*it)[0]);
	for (StringTable::const_iterator it(t_nonsilencephones.begin()), it_end(t_nonsilencephones.end()); it != it_end; ++it)
		_nonsilencephones.push_back((*it)[0]);

	if (_silencephones.size() < 1) {
		LOGTW_ERROR << " no silence phones found in file: " << fSilencephones.string() << ".";
		return -1;
	}
	if (_nonsilencephones.size() < 1) {
		LOGTW_ERROR << " no non-silence phones found in file: " << fNonSilencephones.string() << ".";
		return -1;
	}

	return 0;
}
