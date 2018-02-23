/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal), Apache 2.0.
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"
#include <kaldi-win/utility/strvec2arg.h>

/*
	Computes the CER (Character Error Rate)
	NOTE: when run both WER and CER then call WER first and then CER with stage=2, in this way the 
		  the lattice decoding won't be run twice.
*/
int ScoreKaldiCER(
	fs::path data,											//data directory <data-dir>
	fs::path lang_or_graph,									//language or graph directory <lang-dir|graph-dir>
	fs::path dir,											//decode directory <decode-dir>
	UMAPSS wer_ref_filter,									//for filtering the text; contains a 'key' which is a regex and the result will be replaced in the input text with the 'value' property in the map
	UMAPSS wer_hyp_filter,									//for filtering the text; contains a 'key' which is a regex and the result will be replaced in the input text with the 'value' property in the map
	int nj,													//
	int stage, //= 0										//start scoring script from part-way through.
	bool decode_mbr, //= false								//maximum bayes risk decoding (confusion network).
	bool stats, // = true									//output statistics
	double beam, // = 6										//Pruning beam [applied after acoustic scaling]
	std::string word_ins_penalty, // = 0.0, 0.5, 1.0		//word insertion penalty
	int min_lmwt, // = 7									//minumum LM-weight for lattice rescoring
	int max_lmwt, // = 17									//maximum LM-weight for lattice rescoring
	std::string iter // = final								//which model to use; default final.mdl
)
{

	//TODO:...


	return 0;
}

