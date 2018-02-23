/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :  Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
			See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "gmm/am-diag-gmm.h"

#include "kaldi-win/src/kaldi_src.h"

int GmmBoostSilence(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Modify GMM-based model to boost (by a certain factor) all\n"
        "probabilities associated with the specified phones (could be\n"
        "all silence phones, or just the ones used for optional silence).\n"
        "Note: this is done by modifying the GMM weights.  If the silence\n"
        "model shares a GMM with other models, then it will modify the GMM\n"
        "weights for all models that may correspond to silence.\n"
        "\n"
        "Usage:  gmm-boost-silence [options] <silence-phones-list> <model-in> <model-out>\n"
        "e.g.: gmm-boost-silence --boost=1.5 1:2:3 1.mdl 1_boostsil.mdl\n";
    
    bool binary_write = true;
    BaseFloat boost = 1.5;
        
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("boost", &boost, "Factor by which to boost silence probs");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "wrong arguments.";
		return -1;
    }
    
    std::string
        silence_phones_string = po.GetArg(1),
        model_rxfilename = po.GetArg(2),
        model_wxfilename = po.GetArg(3);
    
    std::vector<int32> silence_phones;
    if (silence_phones_string != "") {
      SplitStringToIntegers(silence_phones_string, ":", false, &silence_phones);
      std::sort(silence_phones.begin(), silence_phones.end());
      KALDI_ASSERT(IsSortedAndUniq(silence_phones) && "Silence phones non-unique.");
    } else {
      KALDI_WARN << "gmm-boost-silence: no silence phones specified, doing nothing.";
    }
    
    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    { // Do the modification to the am_gmm object.
      std::vector<int32> pdfs;
      bool ans = GetPdfsForPhones(trans_model, silence_phones, &pdfs);
      if (!ans) {
        KALDI_WARN << "The pdfs for the silence phones may be shared by other phones "
                   << "(note: this probably does not matter.)";
      }
      for (size_t i = 0; i < pdfs.size(); i++) {
        int32 pdf = pdfs[i];
        DiagGmm &gmm = am_gmm.GetPdf(pdf);
        Vector<BaseFloat> weights(gmm.weights());
        weights.Scale(boost);
        gmm.SetWeights(weights);
        gmm.ComputeGconsts();
      }
	  if (file_log)
		  file_log << "Boosted weights for " << pdfs.size() << " pdfs, by factor of " << boost << "\n";
	  else
      KALDI_LOG << "Boosted weights for " << pdfs.size() << " pdfs, by factor of " << boost;
    }
    
    {
      Output ko(model_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }

	if (file_log)
		file_log << "Wrote model to " << model_wxfilename << "\n";
	else
	KALDI_LOG << "Wrote model to " << model_wxfilename;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what() << '\n';
    return -1;
  }

  return 0;
}


