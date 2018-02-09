/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
Based on : Phonetisaurus: Copyright (c) [2012-], Josef Robert Novak, All rights reserved.
*/
/* Copyright (c) [2012-], Josef Robert Novak, All rights reserved.
   Redistribution and use in source and binary forms, with or without
   modification, are permitted #provided that the following conditions
   are met:
   * Redistributions of source code must retain the above copyright 
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above 
     copyright notice, this list of #conditions and the following 
     disclaimer in the documentation and/or other materials provided 
     with the distribution.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
   OF THE POSSIBILITY OF SUCH DAMAGE. */

//
//IMPORTANT: these two headers must proceed the rest of the headers otherwise VS crashes while compiling because of 
//			 the redefinition of some symbols as BYTE
#include "..\kaldi-win\utility\Utility.h"
#include "..\kaldi-win\stdafx.h"
//----------------------------------------------------
#include <fst/fstlib.h>
#include "PhonetisaurusScript.h"
#include "utilp.h"
//#include "iomanip"


using namespace fst;

typedef unordered_map<int, vector<PathData> > RMAP;

using MAPSS = std::map<std::string, std::string>; //automatically sorted and unique 

static int PrintPathData(const vector<PathData>& results, string FLAGS_word,
	const SymbolTable* osyms, fs::ofstream & ofs, MAPSS & _refdict, bool forceModel, bool print_scores = true, bool nlog_probs = true)
{
	try	{
		//search for the pronunciation in the reference dictionary first if requested
		if (!forceModel) {
			if (_refdict.size() < 1) {
				LOGTW_WARNING << "Reference dictionary is empty! Using model...";
			}
			else {
				MAPSS::const_iterator it = _refdict.find(FLAGS_word);
				if (it != _refdict.end()) {
					std::stringstream ss;
					ss << FLAGS_word << "\t" << it->second;
					//LOGTW_INFO << ss.str();
					ofs << ss.str() << "\n";
					return 0;
				}
			}
		}

		//not found in the reference dictionary or forcing model:
		for (int i = 0; i < results.size(); i++)
		{
			std::stringstream ss;
			ss << FLAGS_word << "\t";
			if (print_scores == true) {
				if (nlog_probs == true)
					ss << results[i].PathWeight << "\t";
				else
					ss << std::setprecision(3) << exp(-results[i].PathWeight) << "\t";
			}
			//
			for (int j = 0; j < results[i].Uniques.size(); j++) {
				ss << osyms->Find(results[i].Uniques[j]);
				if (j < results[i].Uniques.size() - 1) {
					ss << " ";
				}
			}
			//NOTE: there can be several solutions (pronounciations for the same word!)
			//LOGTW_INFO << ss.str();
			ofs << ss.str() << "\n";
		}
		return 0;
	}
	catch (const std::exception& e)
	{
		LOGTW_ERROR << "Failed to write output file. Reason: " << e.what();
		return -1;
	}
}

static int EvaluateWordlist(PhonetisaurusScript& decoder, vector<string> corpus,
	int FLAGS_beam, int FLAGS_nbest, bool FLAGS_reverse,
	string FLAGS_skip, double FLAGS_thresh, string FLAGS_gsep,
	bool FLAGS_write_fsts, bool FLAGS_print_scores,
	bool FLAGS_accumulate, double FLAGS_pmass,
	bool FLAGS_nlog_probs, fs::ofstream & ofs, bool forceModel, MAPSS & _refdict)
{
	for (int i = 0; i < corpus.size(); i++) {
		vector<PathData> results = decoder.Phoneticize(corpus[i], FLAGS_nbest,
			FLAGS_beam, FLAGS_thresh,
			FLAGS_write_fsts,
			FLAGS_accumulate, FLAGS_pmass);
		int ret = PrintPathData(results, corpus[i],
			decoder.osyms_,
			ofs,
			_refdict,
			forceModel,
			FLAGS_print_scores,
			FLAGS_nlog_probs);

		if (ret < 0) return -1;
	}
	return 0;
}

DEFINE_string(outfile_P, "", "Output text file path.");
DEFINE_string(refdict_P, "", "Reference dictionary file path.");
DEFINE_bool(forcemodel_P, true, "If false the reference dict will first be used to search for the pronunciation.");
DEFINE_string (model_P, "", "Input FST G2P model.");
DEFINE_string (word_P, "", "Input word to phoneticize.");
DEFINE_string (wordlist_P, "", "Input wordlist to phoneticize");
DEFINE_string (gsep_P, "", "Grapheme separator.");
DEFINE_string (skip_P, "_", "Phoneme skip marker.");
DEFINE_int32 (nbest_P, 1, "N-best hypotheses to output.");
DEFINE_int32 (beam_P, 10000, "Decoder beam.");
DEFINE_double (thresh_P, 99.0, "N-best comparison threshold.");
DEFINE_double (pmass_P, 0.0, "Percent of probability mass (0.0 < p <= 1.0).");
DEFINE_bool (write_fsts_P, false, "Write the output FSTs for debugging.");
DEFINE_bool (reverse_P, false, "Reverse input word.");
DEFINE_bool (print_scores_P, false, "Print scores in output.");
DEFINE_bool (accumulate_P, false, "Accumulate weights for unique output prons.");
DEFINE_bool (nlog_probs_P, true, "Default scores vals are negative logs. Otherwise exp (-val).");

/*
	Returns the pronunciation for a word or for a list of words when a trained model and a reference dictionary 
	(--refdict_P) are supplied.
	The reference dictionary is only needed in case the --forcemodel_P parameter is set to false.
	The output is daved into a text file given by --outfile_P.
*/
int G2Pfst(int argc, char* argv[]) 
{
	string usage = "";
	set_new_handler(FailedNewHandler);
	PhonetisaurusSetFlags(usage.c_str(), &argc, &argv, false);

	if (FLAGS_model_P.compare("") == 0) {
		LOGTW_DEBUG << "You must supply an FST model with --model";
		return -1;
	}
	else {
		std::ifstream model_ifp(FLAGS_model_P);
		if (!model_ifp.good()) {
			LOGTW_ERROR << "Failed to open model file '" << FLAGS_model_P << "'";
			return -1;
		}
	}

	if (!FLAGS_forcemodel_P && FLAGS_refdict_P.compare("") == 0) {
		LOGTW_DEBUG << "You must supply a reference dictionary with --refdict_P";
		return -1;
	}
	else if(!FLAGS_forcemodel_P) {
		std::ifstream ref_ifp(FLAGS_refdict_P);
		if (!ref_ifp.good()) {
			LOGTW_ERROR << "Failed to open reference dictionary '" << FLAGS_refdict_P << "'";
			return -1;
		}
	}

	if (FLAGS_pmass_P < 0.0 || FLAGS_pmass_P > 1) {
		LOGTW_DEBUG << "--pmass must be a float value between 0.0 and 1.0.";
		return -1;
	}
	if (FLAGS_pmass_P == 0.0)
		FLAGS_pmass_P = 99.0;
	else
		FLAGS_pmass_P = -log(FLAGS_pmass_P);

	bool use_wordlist = false;
	if (FLAGS_wordlist_P.compare("") != 0) {
		std::ifstream wordlist_ifp(FLAGS_wordlist_P);
		if (!wordlist_ifp.good()) {
			LOGTW_ERROR << "Failed to open wordlist file '" << FLAGS_wordlist_P << "'";
			return -1;
		}
		else {
			use_wordlist = true;
		}
	}

	if (FLAGS_wordlist_P.compare("") == 0 && FLAGS_word_P.compare("") == 0) {
		LOGTW_DEBUG << "Either --wordlist or --word must be set!";
		return -1;
	}

	//in case the reference dictionary may be used for searching for the pron then load the reference dictionary in memory
	//for fast searching
	MAPSS _refdict;
	if (!FLAGS_forcemodel_P) {
		fs::ifstream ifs(FLAGS_refdict_P);
		if (!ifs) {
			LOGTW_ERROR << "Could not open " << FLAGS_refdict_P;
			return -1;
		}
		std::string line;
		//must also convert the dictionary to Phonetisaurus format
		static const boost::regex rexp("\\([0-9]+\\)");
		while (std::getline(ifs, line)) {
			//remove space from the start and tail.
			boost::algorithm::trim(line);
			std::vector<std::string> _w;
			//use regex to remove (n) from repeated words
			line = boost::regex_replace(line, rexp, "");
			//there can be a comment in the line: remove the comment
			size_t pos = line.find_first_of('#', 0);
			if (pos != std::string::npos)
				line = line.substr(0, pos-1);
			//split line
			strtk::parse(line, " ", _w, strtk::split_options::compress_delimiters);
			int c = 1;
			std::stringstream pron;
			for (int i = 1; i < _w.size(); i++) {
				if (c > 1) pron << " ";
				pron << ConvertToCaseUtf8(_w[i], true); //NOTE: phonems upper case!
				c++;
			}
			_refdict.emplace(ConvertToCaseUtf8(_w[0], false), pron.str()); //automatically sorted and unique; Lower case!
		}
	}

	fs::ofstream ofs(FLAGS_outfile_P);
	if (!ofs) {
		LOGTW_ERROR << "Failed to open file '" << FLAGS_outfile_P << "'";
		return -1;
	}

	if (use_wordlist == true) {
		vector<string> corpus;
		LoadWordList(FLAGS_wordlist_P, &corpus);

		PhonetisaurusScript decoder(FLAGS_model_P, FLAGS_gsep_P);
		if (EvaluateWordlist(
			decoder, corpus, FLAGS_beam_P, FLAGS_nbest_P, FLAGS_reverse_P,
			FLAGS_skip_P, FLAGS_thresh_P, FLAGS_gsep_P, FLAGS_write_fsts_P,
			FLAGS_print_scores_P, FLAGS_accumulate_P, FLAGS_pmass_P,
			FLAGS_nlog_probs_P, ofs, FLAGS_forcemodel_P, _refdict) < 0) return -1;
	}
	else {
		PhonetisaurusScript decoder(FLAGS_model_P, FLAGS_gsep_P);
		vector<PathData> results = decoder.Phoneticize(
			FLAGS_word_P, FLAGS_nbest_P, FLAGS_beam_P, FLAGS_thresh_P,
			FLAGS_write_fsts_P, FLAGS_accumulate_P, FLAGS_pmass_P
		);
		if (PrintPathData(results, FLAGS_word_P,
			decoder.osyms_,
			ofs,
			_refdict,
			FLAGS_forcemodel_P,
			FLAGS_print_scores_P,
			FLAGS_nlog_probs_P) < 0) return -1;
	}

	return 0;
}
