/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"

/*
This code replaces the existing pronunciation of the unknown word in the provided lexicon, with a pronunciation
consisting of three disambiguation symbols: #1 followed by #2 followed by #3.
The #2 will later be replaced by a phone-level LM by apply_unk_lm.sh (called later on by prepare_lang.sh).
Caution: this script is sensitive to the basename of the lexicon: it should be called either lexiconp.txt, in which
case the format is 'word pron-prob p1 p2 p3 ...' or lexiconp_silprob.txt, in which case the format is
'word pron-prob sil-prob1 sil-prob2 sil-prob3 p1 p2 p3....'. 
It is an error if there is not exactly one pronunciation of the unknown word in the lexicon.
*/
//lexicon_file: Filename of the lexicon file to operate on (this is both an input and output of this script).
//unk_word: The printed form of the unknown / OOV word, normally '<unk>'.
//return: 0 if OK, -1 if error
int ModifyUnkPron(fs::path lexicon_file, std::string unk_word)
{
	if (lexicon_file.filename() != "lexiconp.txt" && lexicon_file.filename() != "lexiconp_silprob.txt") {
		LOGTW_ERROR << "Expected the lexicon file name to be either 'lexiconp.txt' or 'lexiconp_silprob.txt' got: " << lexicon_file.filename() << ".";
		return -1;
	}
	//the lexiconp.txt format is: word pron-prob p1 p2 p3...
	//lexiconp_silprob.txt has 3 extra real - valued fields after the pron - prob.
	int num_fields_before_pron = 5;
	if (lexicon_file.filename() == "lexiconp.txt")
		num_fields_before_pron = 2;

	StringTable split_lines;
	try
	{
		split_lines = readData(lexicon_file.string());
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
	if (split_lines.size() < 1) {
		LOGTW_ERROR << "Lexicon file is empty. " << lexicon_file.string() << ".";
		return -1;
	}
	//determine unk_index
	int unk_index = -1, row = 0;
	for (StringTable::const_iterator it(split_lines.begin()), it_end(split_lines.end()); it != it_end; ++it)
	{
		if ((*it).size() <= num_fields_before_pron) {
			LOGTW_ERROR << "There are not enough fields in " << lexicon_file.string() << " at line " << (row+1) << ".";
			return -1;
		}
		if ((*it)[0] == unk_word) {
			if (unk_index != -1) {
				LOGTW_ERROR << "Expected there to be exactly one pronunciation of the "
                     "unknown word " << unk_word << " in " << lexicon_file.string() << ", but there are more than one.";
				return -1;
			}
			unk_index = row;
		}
		row++;
	}
	if (unk_index == -1) {
		LOGTW_ERROR << "Expected there to be exactly one pronunciation of the "
			"unknown word " << unk_word << " in " << lexicon_file.string() << ", but there are none.";
		return -1;
	}

	// now modify the pron.
	std::vector<std::string> mod_pron;
	int c = 0;
	for (string_vec::const_iterator itc(split_lines[unk_index].begin()), itc_end(split_lines[unk_index].end()); itc != itc_end; ++itc)
	{
		if (c > num_fields_before_pron - 1) break;
		mod_pron.push_back(*itc);
		c++;
	}
	mod_pron.push_back("#1");
	mod_pron.push_back("#2");
	mod_pron.push_back("#3");
	split_lines[unk_index] = mod_pron;
	//save the modified data into the same file back
	//NOTE: ofstream write adds an '\r' in front of the '\n' automatically and some code in Kaldi crashes. To prevent this
	//		add std::ios::binary option to each ofstream to make sure that '\r' is not added!
	fs::ofstream file_lexicon_file(lexicon_file, std::ios::binary);
	if (!file_lexicon_file) {
		LOGTW_ERROR << "Can't open output file: " << lexicon_file.string() << ".";
		return -1;
	}
	for (StringTable::const_iterator it(split_lines.begin()), it_end(split_lines.end()); it != it_end; ++it)
	{
		int c = 0;
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
		{
			if (c > 0) file_lexicon_file << " "; //Separator!
			file_lexicon_file << *itc1;
			c++;
		}
		file_lexicon_file << "\n";
	}
	file_lexicon_file.flush(); file_lexicon_file.close();

	return 0;
}
