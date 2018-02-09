/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
			See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

#include "kaldi-win/src/kaldi_src.h"

//various properties of a model, of any type
int AmInfo(std::string model_in_filename,
	int & nofphones,
	int & nofpdfs,
	int & noftransitionids,
	int & noftransitionstates
	) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

	TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
    }

    KALDI_LOG << "number of phones " << trans_model.GetPhones().size() << '\n';
	KALDI_LOG << "number of pdfs " << trans_model.NumPdfs() << '\n';
	KALDI_LOG << "number of transition-ids " << trans_model.NumTransitionIds()
              << '\n';
	KALDI_LOG << "number of transition-states "
              << trans_model.NumTransitionStates() << '\n';

	nofphones = trans_model.GetPhones().size();
	nofpdfs = trans_model.NumPdfs();
	noftransitionids = trans_model.NumTransitionIds();
	noftransitionstates = trans_model.NumTransitionStates();
	return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}


