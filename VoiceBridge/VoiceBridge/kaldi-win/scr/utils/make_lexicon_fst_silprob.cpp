/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2012 Microsoft Corporation  | license: Apache 2.0.
		   2012  Johns Hopkins University (Author: Daniel Povey), 
		   2015  Hainan Xu, 2015  Guoguo Chen
*/

//Makes lexicon FST, in text form, from lexicon which contains(optional) probabilities of pronuniations, 
//and (mandatory)probabilities of silence before and after the pronunciation.This script is almost the same with
//the make_lexicon_fst.pl script except for the word - dependent silprobs part

/*
This is almost the same as the utils/make_lexicon_fst except here we include word-dependent silence probabilities
when making the lexicon FSTs. For details, see paper http://danielpovey.com/files/2015_interspeech_silprob.pdf
The lexiconp_silprob_disambig.txt file should have each line like

	word p(pronunciation|word) p(sil-after|word) correction-term-for-sil
	correction-term-for-no-sil phone-1 phone-2 ... phone-N

The pronunciation would have to include disambiguation symbols the 2 correction terms above are computed to reflect how much a
word affects the probability of a [non-]silence before it. Please see the paper (link given above) for detailed descriptions
for how the 2 terms are computed. The silprob.txt file contains 4 lines:

	<s> p(sil-after|<s>)
	</s>_s correction-term-for-sil-for-</s>
	</s>_n correction-term-for-no-sil-for-</s>
	overall p(overall-sil)
*/

#include "kaldi-win\scr\kaldi_scr.h"

int MakeLexiconFstSilprob(StringTable lexfn, StringTable silprobfile, fs::path output_txt,
						  std::string silphone, std::string sildisambig)
{
	double silbeginprob = -1.0;
	double silendcorrection = -1.0;
	double nonsilendcorrection = -1.0;
	double siloverallprob = -1.0;

	//open output file
	//NOTE: ofstream write adds an '\r' in front of the '\n' automatically and some code in Kaldi crashes. To prevent this
	//		add std::ios::binary option to each ofstream to make sure that '\r' is not added!
	fs::ofstream file_output_txt(output_txt, std::ios::binary);
	if (!file_output_txt) {
		LOGTW_ERROR << " can't open output file: " << output_txt.string() << ".";
		return -1;
	}

	int n = 0;
	for (StringTable::const_iterator it(silprobfile.begin()), it_end(silprobfile.end()); it != it_end; ++it)
	{
		n++;
		std::string w((*it)[0]);
		if (w == "<s>") {
			silbeginprob = StringToNumber<double>((*it)[1], -1.0);
		}
		else if (w == "</s>_s") {
			silendcorrection = StringToNumber<double>((*it)[1], -1.0);
			if (silendcorrection <= 0) {
				LOGTW_ERROR << " Bad correction term in file silprob at line " << n << ".";
				return -1;
			}
		}
		else if (w == "</s>_n") {
			nonsilendcorrection = StringToNumber<double>((*it)[1], -1.0);
			if (nonsilendcorrection <= 0) {
				LOGTW_ERROR << " Bad correction term in file silprob at line " << n << ".";
				return -1;
			}
		}
		else if (w == "overall") {
			siloverallprob = StringToNumber<double>((*it)[1], -1.0);
		}
	}

	int startstate = 0;
	int nonsilstart = 1;
	int silstart = 2;
	int nextstate = 3;

	if (!(silbeginprob > 0.0 && silbeginprob <= 1.0))
	{
		LOGTW_ERROR << "Wrong value " << silbeginprob << " detected in lexicon file.";
		return -1;
	}

	double cost = -std::log(silbeginprob);
	file_output_txt << startstate << "\t" << silstart << "\t" << silphone << "\t" << "<eps>" << "\t" << std::setprecision(8) << cost << "\n"; // will change these	
	cost = -log(1.0 - silbeginprob);
	file_output_txt << startstate << "\t" << nonsilstart << "\t" << sildisambig << "\t" << "<eps>" << "\t" << std::setprecision(8) << cost << "\n";

	n = 0;
	for (StringTable::const_iterator it(lexfn.begin()), it_end(lexfn.end()); it != it_end; ++it)
	{
		n++;
		if ((*it).size() < 5) {
			LOGTW_ERROR << " There is not enough data in line " << n << ".";
			return -1;
		}		
		std::string w((*it)[0]);
		double pron_prob = StringToNumber<double>((*it)[1], -1.0);
		if (!(pron_prob > 0.0 && pron_prob <= 1.0)) {
			LOGTW_ERROR << " Bad pronunciation probability in line " << n << ".";
			return -1;
		}
		double wordsilprob = StringToNumber<double>((*it)[2], -1.0);
		double silwordcorrection = StringToNumber<double>((*it)[3], -1.0);
		double nonsilwordcorrection = StringToNumber<double>((*it)[4], -1.0);
		if (!(wordsilprob > 0.0 && wordsilprob <= 1.0) || !(silwordcorrection > 0.0) || !(nonsilwordcorrection > 0.0))
		{
			LOGTW_ERROR << " Bad word pronunciation probability in line " << n << ".";
			return -1;
		}
		double pron_cost = -std::log(pron_prob);
		double wordsilcost = -std::log(wordsilprob);
		double wordnonsilcost = -std::log(1.0 - wordsilprob);
		double silwordcost = -std::log(silwordcorrection);
		double nonsilwordcost = -std::log(nonsilwordcorrection);
		bool bFirst = true;
		int newstate, oldstate;
		int c = 0, last = (int)(*it).size() - 1;
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
		{
			if (c < 5) {
				c++;
				continue;
			}
			std::string p(*itc1);
			if (bFirst) {
				newstate = nextstate++;
				// for nonsil before w
				cost = nonsilwordcost + pron_cost;
				file_output_txt << nonsilstart << "\t" << newstate << "\t" << p << "\t" << w << "\t" << std::setprecision(8) << cost << "\n";
				// for sil before w
				cost = silwordcost + pron_cost;
				file_output_txt << silstart << "\t" << newstate << "\t" << p << "\t" << w << "\t" << std::setprecision(8) << cost << "\n";
				bFirst = false;
			}
			else {
				oldstate = nextstate - 1;
				file_output_txt << oldstate << "\t" << nextstate << "\t" << p << "\t" << "<eps>" << "\n";
				nextstate++;
			}
			if (c == last) {
				oldstate = nextstate - 1;
				// for no sil after w
				cost = wordnonsilcost;
				file_output_txt << oldstate << "\t" << nonsilstart << "\t" << sildisambig << "\t" << "<eps>" << "\t" << std::setprecision(8) << cost << "\n";
				// for sil after w
				cost = wordsilcost;
				file_output_txt << oldstate << "\t" << silstart << "\t" << silphone << "\t" << "<eps>" << "\t" << std::setprecision(8) << cost << "\n";
			}
			c++;
		}
	}

	cost = -std::log(silendcorrection);
	file_output_txt << silstart << "\t" << std::setprecision(8) << cost << "\n";
	cost = -std::log(nonsilendcorrection);
	file_output_txt << nonsilstart << "\t" << std::setprecision(8) << cost << "\n";

	return 0;
}
