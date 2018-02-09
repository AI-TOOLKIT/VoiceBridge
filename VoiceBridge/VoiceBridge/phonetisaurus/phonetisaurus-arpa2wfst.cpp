/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
Based on: see below
*/
/*
 phonetisaurus-arpa2wfst.cc

 Copyright (c) [2012-], Josef Robert Novak
 All rights reserved.

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
   OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/
#include "kaldi-win/utility/Utility.h"

#include "ARPA2WFST.h"
#include "utilp.h"

using namespace fst;

DEFINE_string(lm_W, "", "Input ARPA format LM.");
DEFINE_string(eps_W, "<eps>", "Epsilon symbol.");
DEFINE_string(sb_W, "<s>", "Sentence begin token.");
DEFINE_string(se_W, "</s>", "Sentence end token.");
DEFINE_string(split_W, "}", "Character separating grapheme/phoneme info.");
DEFINE_string(skip_W, "_", "Character indicating insertions/deletions.");
DEFINE_string(tie_W, "|", "Character separating multi-token subsequences.");
DEFINE_string(ssyms_W, "", "Output filename for state symbols tables (default: do not print).");
DEFINE_string(ofile_W, "", "Output file for writing. (STDOUT)");

/*
	Convert arpa model to wfst format
*/
int Arpa2Wfst(int argc, char* argv[])
{
	string usage = "";
	set_new_handler(FailedNewHandler);
	PhonetisaurusSetFlags(usage.c_str(), &argc, &argv, false);

	if (FLAGS_lm_W.compare("") == 0) {
		LOGTW_ERROR << "You must supply an ARPA format lm for conversion!";
		return 0;
	}

	LOGTW_INFO << "Initializing..." << endl;
	ARPA2WFST* converter = new ARPA2WFST(FLAGS_lm_W, FLAGS_eps_W, FLAGS_sb_W, FLAGS_se_W, FLAGS_split_W, FLAGS_skip_W, FLAGS_tie_W);
	LOGTW_INFO << "Converting..." << endl;
	converter->arpa_to_wfst();

	converter->arpafst.Write(FLAGS_ofile_W);

	if (FLAGS_ssyms_W.compare("") != 0) {
		converter->ssyms->WriteText(FLAGS_ssyms_W);
	}

	delete converter;

	return 0;
}
