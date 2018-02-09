/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.


Based on : Copyright 2010-2011  Microsoft Corporation, Apache 2.0.
           2013-2016  Johns Hopkins University (author: Daniel Povey)
           2015  Hainan Xu
           2015  Guoguo Chen
*/

#include "kaldi-win\scr\kaldi_scr.h"

/*
Adds disambiguation symbols to a lexicon. Outputs still in the normal lexicon format.
Disambig syms are numbered #1, #2, #3, etc. (#0 reserved for symbol in grammar).
Outputs the number of disambig syms to the standard output. With the --pron-probs option, expects the second field
of each lexicon line to be a pron-prob. With the --sil-probs option, expects three additional
fields after the pron-prob, representing various components of the silence probability model.
*/
//first_allowed_disambig <n>  The number of the first disambiguation symbol that this function is allowed to add.  
//The default value is 1, but you can set this to a larger value using this option.
//sil_probs: silence probability model (Expect 3 extra fields after the pron-probs)
//pron_probs: pronunciation probabilities (Expect pronunciation probabilities in the 2nd field)
//Return: max_disambig
int AddLexDisambig(bool pron_probs, bool sil_probs, int first_allowed_disambig, fs::path lexiconp_silprob, fs::path lexiconp_silprob_disambig)
{
	if (first_allowed_disambig < 1) {
		LOGTW_ERROR << "Invalid first allowed disambig: " << first_allowed_disambig << ".";
		return -1;
	}
	//check input lexicon
	if (!fs::exists(lexiconp_silprob) || fs::is_empty(lexiconp_silprob)) {
		LOGTW_ERROR << "Error opening lexicon or the file is empty: " << lexiconp_silprob.string() << ".";
		return -1;
	}
	//open output lexicon file
	//NOTE: ofstream write adds an '\r' in front of the '\n' automatically and some code in Kaldi crashes. To prevent this
	//		add std::ios::binary option to each ofstream to make sure that '\r' is not added!
	fs::ofstream file_lexiconp_silprob_disambig(lexiconp_silprob_disambig, std::ios::binary);
	if (!file_lexiconp_silprob_disambig) {
		LOGTW_ERROR << "Can't open output file: " << lexiconp_silprob_disambig.string() << ".";
		return -1;
	}

	//sil_probs should be with pron_probs option!
	if (sil_probs) pron_probs = true;
	
	//holds the count of each phone-sequence in the lexicon
	using MAPCPS = std::unordered_map<std::string, int>;
	MAPCPS count_phonesequence;	
	//holds the subsequences of all phone-sequences
	std::vector<std::string> issubseq;
	StringTable table_lexiconp_silprob;
	try
	{
		//(1) read the input data from srcdir/lexiconp_silprob.txt (pth_lexiconp_silprob_txt) and make new file
		table_lexiconp_silprob = readData(lexiconp_silprob.string());
	}
	catch (std::exception const& e)
	{
		LOGTW_FATALERROR << " " << e.what() << ".";
		return -1;
	}
	catch (...)
	{
		LOGTW_FATALERROR << "Unknown Error.";
		return -1;
	}

	if (table_lexiconp_silprob.size() < 1)
	{
		LOGTW_ERROR << "There is no data in file: " << lexiconp_silprob.string() << ".";
		return -1;
	}
	int c = 0, nCols = 0, r = 0, startindx_phonesequence = 0;
	char *stopstring; //just used for the conversion function
	for (StringTable::const_iterator it(table_lexiconp_silprob.begin()), it_end(table_lexiconp_silprob.end()); it != it_end; ++it)
	{			
		nCols = (int)(table_lexiconp_silprob[r].size());
		if (pron_probs && nCols < 3)
		{
			LOGTW_ERROR << "There are not enough data columns for pronunciation probabilities in file: " << lexiconp_silprob.string() << ".";
			return -1;
		}
		else if (sil_probs && nCols < 6)
		{
			LOGTW_ERROR << "There are not enough data columns for silence probability model in file: " << lexiconp_silprob.string() << ".";
			return -1;
		}
		//check values
		if (pron_probs) {				
			double p = std::strtod( (*it)[1].c_str(), &stopstring);
			if (!(p > 0.0 && p <= 1.0)) { 
				LOGTW_ERROR << "Wrong pronunciation probability at line " << (r+1) << " in file: " << lexiconp_silprob.string() << ".";
				return -1;
			}
			startindx_phonesequence = 2;
		}
		//check values
		if (sil_probs) {
			//NOTE: strtod will return 0 if can not be converted!
			double p = std::strtod((*it)[2].c_str(), &stopstring);
			double cor1 = std::strtod((*it)[3].c_str(), &stopstring);
			double cor2 = std::strtod((*it)[4].c_str(), &stopstring);
			if (!(p > 0.0 && p <= 1.0)) {
				LOGTW_ERROR << "Wrong silence pronunciation probability at line " << (r + 1) << " in file: " << lexiconp_silprob.string() << ".";
				return -1;
			}
			if (cor1 <= 0.0 || cor2 <= 0.0) {
				LOGTW_ERROR << "Wrong correction value at line " << (r + 1) << " in file: " << lexiconp_silprob.string() << ".";
				return -1;
			}
			startindx_phonesequence = 5;
		}
		c = 0;
		//(2) Work out the count of each phone-sequence in the lexicon.
		std::string phonesequence;
		for (int i = startindx_phonesequence; i<(*it).size(); i++)
		{
			if (c > 0) phonesequence.append(" "); //Separator!
			phonesequence.append((*it)[i]);
			// (3) For each left sub-sequence of each phone-sequence, note down that it exists(for identifying prefixes of longer strings).
			if (i < (*it).size() - 1) {
				//add it only once
				std::vector<std::string>::iterator its = std::find(issubseq.begin(), issubseq.end(), phonesequence);
				if (its == issubseq.end())
					issubseq.push_back(phonesequence);
			}
			c++;
		}
		MAPCPS::iterator itm = count_phonesequence.find(phonesequence);
		if (itm != count_phonesequence.end()) 
		{//already exists, increment count
			itm->second++;
		}
		else 
		{//does not exist, add it and set count to 1
			count_phonesequence.emplace(phonesequence, 1);
		}
		r++;
	}

	// (4) For each entry in the lexicon: if the phone sequence is unique and is not a prefix of another word, 
	// no disambig symbol. Else output #1, or #2, #3, ... if the same phone - seq  has already been assigned a disambig symbol.
	/*
	 Format of lexiconp_disambig.txt:

	 !SIL	1.0   SIL_S
	 <SPOKEN_NOISE>	1.0   SPN_S #1
	 <UNK>	1.0  SPN_S #2
	 <NOISE>	1.0  NSN_S
	 !EXCLAMATION-POINT	1.0  EH2_B K_I S_I K_I L_I AH0_I M_I EY1_I SH_I AH0_I N_I P_I OY2_I N_I T_E
	*/
	// max_disambig will always be the highest-numbered disambiguation symbol that has been used so far.
	int max_disambig = first_allowed_disambig - 1;
	int cur_disambig = 0;
	std::vector<int> reserved_for_the_empty_string;
	MAPCPS last_used_disambig_symbol_of;

	for (StringTable::const_iterator it(table_lexiconp_silprob.begin()), it_end(table_lexiconp_silprob.end()); it != it_end; ++it)
	{
		std::string word = (*it)[0];
		double pron_prob, sil_word_prob, word_sil_correction, prev_nonsil_correction;
		std::string phonesequence;
		if (pron_probs) {
			pron_prob = std::strtod((*it)[1].c_str(), &stopstring);
			startindx_phonesequence = 2;
		}
		if (sil_probs) {
			sil_word_prob = std::strtod((*it)[2].c_str(), &stopstring);
			word_sil_correction = std::strtod((*it)[3].c_str(), &stopstring);
			prev_nonsil_correction = std::strtod((*it)[4].c_str(), &stopstring);
			startindx_phonesequence = 5;
		}
		//work out the phonesequence
		for (int i = startindx_phonesequence; i<(*it).size(); i++)
		{
			if (c > 0) phonesequence.append(" "); //Separator!
			phonesequence.append((*it)[i]);
			c++;
		}
		std::vector<std::string>::iterator its = std::find(issubseq.begin(), issubseq.end(), phonesequence);
		MAPCPS::iterator itm = count_phonesequence.find(phonesequence);		
		if ( !(its == issubseq.end() && itm != count_phonesequence.end() && itm->second == 1) )
		{
			if (phonesequence == "") {
				//need disambig symbols for the empty string that are not use anywhere else.
				max_disambig++;
				reserved_for_the_empty_string.push_back(max_disambig);
				phonesequence = "#"+std::to_string(max_disambig);
			}
			else {
				MAPCPS::iterator itlu = last_used_disambig_symbol_of.find(phonesequence);
				if (itlu == last_used_disambig_symbol_of.end()) {
					cur_disambig = first_allowed_disambig;
				}
				else {
					cur_disambig = itlu->second;
					// Get a number that has not been used yet for this phone sequence.
					cur_disambig++;
				}
				while (std::find(reserved_for_the_empty_string.begin(), reserved_for_the_empty_string.end(), cur_disambig)
								!= reserved_for_the_empty_string.end()) 
				{
					cur_disambig++;
				}
				if (cur_disambig > max_disambig) { max_disambig = cur_disambig; }
				//NOTE: the below code will first check if the key exist and then if it exists updates the value, if it does not exist then it adds a new key
				//		and sets the value. This is probably the best behaviour here. The key should exist anyway.
				last_used_disambig_symbol_of[phonesequence] = cur_disambig; 
				phonesequence.append(" #");
				phonesequence.append(std::to_string(cur_disambig));
			}
		}

		//write file
		if (pron_probs) {
			if (sil_probs) {
				file_lexiconp_silprob_disambig << word << "\t" 
											   << pron_prob << "\t" 
											   << sil_word_prob << "\t" 
											   << word_sil_correction << "\t" 
											   << prev_nonsil_correction << "\t" 
											   << phonesequence << "\n";
			}
			else {
				file_lexiconp_silprob_disambig << word << "\t" << pron_prob << "\t" << phonesequence << "\n"; 
			}
		}
		else {
			file_lexiconp_silprob_disambig << word << "\t" << phonesequence << "\n";
		}
	}
	file_lexiconp_silprob_disambig.flush(); file_lexiconp_silprob_disambig.close();

	return max_disambig;
}
