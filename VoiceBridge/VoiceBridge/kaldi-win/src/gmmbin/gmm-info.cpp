/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :  Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
			See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

#include "kaldi-win/src/kaldi_src.h"

/*
ZSO03012018: the option to write the info to a file is added! <info-out> (out_filename)
*/
int GmmInfo(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Write to standard output various properties of GMM-based model\n"
        "Usage:  gmm-info [options] <model-in> <info-out>\n"
        "e.g.:\n"
        " gmm-info 1.mdl out.info\n"
        "See also: gmm-global-info, am-info\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() < 1) {
		//po.PrintUsage();
		//exit(1);
		LOGTW_ERROR << "wrong arguments.";
		return -1;
    }

    std::string model_in_filename = po.GetArg(1);
	std::string out_filename = po.GetArg(2);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

	fs::ofstream file_out(out_filename, fs::ofstream::binary | fs::ofstream::out);
	if (out_filename != "" && file_out) {
		file_out << "number-of-phones " << trans_model.GetPhones().size() << '\n';
		file_out << "number-of-pdfs " << trans_model.NumPdfs() << '\n';
		file_out << "number-of-transition-ids " << trans_model.NumTransitionIds() << '\n';
		file_out << "number-of-transition-states " << trans_model.NumTransitionStates() << '\n';
		file_out << "feature-dimension " << am_gmm.Dim() << '\n';
		file_out << "number-of-gaussians " << am_gmm.NumGauss() << '\n';
		file_out.flush(); file_out.close();
	}
	else {
		LOGTW_INFO << "number of phones " << trans_model.GetPhones().size() << '.';
		LOGTW_INFO << "number of pdfs " << trans_model.NumPdfs() << '.';
		LOGTW_INFO << "number of transition-ids " << trans_model.NumTransitionIds() << '.';
		LOGTW_INFO << "number of transition-states " << trans_model.NumTransitionStates() << '.';
		LOGTW_INFO << "feature dimension " << am_gmm.Dim() << '.';
		LOGTW_INFO << "number of gaussians " << am_gmm.NumGauss() << '.';
	}
    return 0;
  } catch(const std::exception &e) {
	  LOGTW_FATALERROR << e.what() << '.';
    return -1;
  }
}


