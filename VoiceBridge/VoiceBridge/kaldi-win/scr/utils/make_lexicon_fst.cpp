/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2012 Microsoft Corporation  | license: Apache 2.0.
2012  Johns Hopkins University (Author: Daniel Povey),
*/

//Makes lexicon FST, in text form, from lexicon (pronunciation probabilities optional).

/*
Creates a lexicon FST that transduces phones to words, and may allow optional silence.
Note: ordinarily, each line of lexicon.txt is:
	word phone1 phone2 ... phoneN;
if the --pron-probs option is used, each line is:
	word pronunciation-probability phone1 phone2 ... phoneN.
The probability 'prob' will typically be between zero and one, and note that it's generally helpful to normalize 
so the largest one for each word is 1.0, but this is your responsibility.
The silence disambiguation symbol, e.g. something like #5, is used only when creating a lexicon with
disambiguation symbols, e.g. L_disambig.fst, and was introduced to fix a particular case of non-determinism 
of decoding graphs.
*/

#include "kaldi-win\scr\kaldi_scr.h"

int MakeLexiconFst(StringTable lexfn, fs::path output_txt,
				   bool pron_probs, double silprob, std::string silphone, std::string sildisambig)
{
	//open output file
	//NOTE: ofstream write adds an '\r' in front of the '\n' automatically and some code in Kaldi crashes. To prevent this
	//		add std::ios::binary option to each ofstream to make sure that '\r' is not added!
	fs::ofstream file_output_txt(output_txt, std::ios::binary);
	if (!file_output_txt) {
		LOGTW_ERROR << " can't open output file: " << output_txt.string() << ".";
		return -1;
	}

	int n = 0;
	int startstate = 0;
	int nonsilstart = 1;
	int silstart = 2;
	int nextstate = 0; //next unallocated state.
	int loopstate = 0;
	int silstate = 0;
	int disambigstate = 0;
	double silcost = 0, nosilcost = 0;

	if (silprob < 0 || silprob >= 1.0) {
		LOGTW_ERROR << " wrong silprob value " << silprob << " detected in lexicon file. ( 1.0 > silprob >=0.0 ).";
		return -1;
	}
	if (silprob > 0.0) {
		silcost = -log(silprob);
		nosilcost = -log(1.0 - silprob);
	}

	if (silprob == 0.0) //No optional silences: just have one (loop+final) state which is numbered zero.
	{		
		loopstate = 0;
		nextstate = 1;
		n = 0;
		for (StringTable::const_iterator it(lexfn.begin()), it_end(lexfn.end()); it != it_end; ++it)
		{
			int c = 0, last = (int)(*it).size() - 1;
			n++;
			if ((*it).size() < 1) {
				LOGTW_ERROR << " There is not enough data in line " << n << ".";
				return -1;
			}
			//check <eps>
			for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
			{
				if (*itc1 == "<eps>") {
					LOGTW_ERROR << " Bad lexicon line " << n << " (<eps> is forbidden).";
					return -1;
				}
			}

			std::string w((*it)[c]); 
			c++; //next column
			std::string pron_cost_string("");
			double pron_prob=0, pron_cost=0;
			if (!pron_probs) {
				pron_cost = 0.0;
			}
			else {
				pron_prob = StringToNumber<double>((*it)[c], -1.0); 
				c++; //next column
				if (!(pron_prob > 0.0 && pron_prob <= 1.0)) {
					LOGTW_ERROR << " Bad pronunciation probability in line " << n << ".";
					return -1;
				}
				pron_cost = -std::log(pron_prob);
			}
			if (pron_cost != 0.0) { 
				pron_cost_string = std::to_string(pron_cost);
			}
			else { pron_cost_string = ""; }

			int s = loopstate;

			std::string word_or_eps = w;
			int col = 0, ns;
			for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
			{
				if (col < c) continue; //these columns are read/used already
				if (col != last) {
					ns = nextstate++;
				}
				else {
					ns = loopstate;
				}
				if(pron_cost_string != "")
					file_output_txt << s << "\t" << ns << "\t" << (*it)[col] << "\t" << word_or_eps << "\t" << pron_cost_string << "\n";
				else 
					file_output_txt << s << "\t" << ns << "\t" << (*it)[col] << "\t" << word_or_eps << "\n";
				word_or_eps = "<eps>";
				pron_cost_string = ""; // so we only print it on the first arc of the word.
				s = ns;
				col++;
			}
		}
		file_output_txt << loopstate << "\t" << 0 << "\n"; //final - cost.
	}
	else
	{
		startstate = 0;
		loopstate = 1;
		silstate = 2;   // state from where we go to loopstate after emitting silence.
		//no silence:
		file_output_txt << startstate << "\t" << loopstate << "\t" << "<eps>" << "\t" << "<eps>" << "\t" << std::setprecision(8) << nosilcost << "\n";

		if (sildisambig == "") {
			//silence.
			file_output_txt << startstate << "\t" << loopstate << "\t" << silphone << "\t" << "<eps>" << "\t" << std::setprecision(8) << silcost << "\n";
			//no cost.
			file_output_txt << silstate << "\t" << loopstate << "\t" << silphone << "\t" << "<eps>" << "\n";
			nextstate = 3;
		}
		else {
			disambigstate = 3;
			nextstate = 4;
			//silence
			file_output_txt << startstate << "\t" << disambigstate << "\t" << silphone << "\t" << "<eps>" << "\t" << std::setprecision(8) << silcost << "\n";
			//no cost
			file_output_txt << silstate << "\t" << disambigstate << "\t" << silphone << "\t" << "<eps>" << "\n";
			//silence disambiguation symbol.
			file_output_txt << disambigstate << "\t" << loopstate << "\t" << sildisambig << "\t" << "<eps>" << "\n";
		}

		n = 0;		
		for (StringTable::const_iterator it(lexfn.begin()), it_end(lexfn.end()); it != it_end; ++it)
		{
			int c = 0, last = (int)(*it).size() - 1;
			std::string w((*it)[c]);
			c++; //next column
			std::string pron_cost_string("");
			double pron_prob = 0, pron_cost = 0;
			if (!pron_probs) {
				pron_cost = 0.0;
			}
			else {
				pron_prob = StringToNumber<double>((*it)[c], -1.0);
				c++; //next column
				if (!(pron_prob > 0.0 && pron_prob <= 1.0)) {
					LOGTW_ERROR << " Bad pronunciation probability in line " << n << ".";
					return -1;
				}
				pron_cost = -std::log(pron_prob);
			}
			if (pron_cost != 0.0) {
				pron_cost_string = std::to_string(pron_cost);
			}
			else { pron_cost_string = ""; }

			int s = loopstate;
			std::string word_or_eps = w;
			int col = 0, ns;

			for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
			{
				if (col < c) {
					col++;
					continue; //these columns are read/used already
				}
				if (col != last) {
					ns = nextstate++;
					if (pron_cost_string != "")
						file_output_txt << s << "\t" << ns << "\t" << (*it)[col] << "\t" << word_or_eps << "\t" << pron_cost_string << "\n";
					else
						file_output_txt << s << "\t" << ns << "\t" << (*it)[col] << "\t" << word_or_eps << "\n";
					word_or_eps = "<eps>";
					pron_cost_string = ""; 
					pron_cost = 0.0; // so we only print it the 1st time.
					s = ns;
				} 
				else if(silphone == "" || (*it)[col] == silphone) {
					// This is non - deterministic but relatively compact, and avoids epsilons.
					double local_nosilcost = nosilcost + pron_cost;
					double local_silcost = silcost + pron_cost;
					file_output_txt << s << "\t" << loopstate << "\t" << (*it)[col] << "\t" << word_or_eps << "\t" << std::setprecision(8) << local_nosilcost << "\n";
					file_output_txt << s << "\t" << silstate << "\t" << (*it)[col] << "\t" << word_or_eps << "\t" << std::setprecision(8) << local_silcost << "\n";
				}
				else {
					// no point putting opt - sil after silence word.
					if (pron_cost_string != "")
						file_output_txt << s << "\t" << loopstate << "\t" << (*it)[col] << "\t" << word_or_eps << "\t" << pron_cost_string << "\n";
					else
						file_output_txt << s << "\t" << loopstate << "\t" << (*it)[col] << "\t" << word_or_eps << "\n";
				}
				col++;
			}
		}

		file_output_txt << loopstate << "\t" << 0 << "\n"; //final - cost.
	}

	return 0;
}
