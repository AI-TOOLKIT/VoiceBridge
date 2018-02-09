/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :  Copyright 2009-2011  Saarland University;  Microsoft Corporation, Apache 2.0
			See ../../COPYING for clarification regarding multiple authors
*/

#include "util/common-utils.h"
#include "gmm/mle-am-diag-gmm.h"
#include "hmm/transition-model.h"

#include "kaldi-win/src/kaldi_src.h"

int GmmSumAccs(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Sum multiple accumulated stats files for GMM training.\n"
        "Usage: gmm-sum-accs [options] <stats-out> <stats-in1> <stats-in2> ...\n"
        "E.g.: gmm-sum-accs 1.acc 1.1.acc 1.2.acc\n";

    bool binary = true;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "wrong arguments.";
		return -1;
    }

    std::string stats_out_filename = po.GetArg(1);
    kaldi::Vector<double> transition_accs;
    kaldi::AccumAmDiagGmm gmm_accs;

    int num_accs = po.NumArgs() - 1;
    for (int i = 2, max = po.NumArgs(); i <= max; i++) {
      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      kaldi::Input ki(stats_in_filename, &binary_read);
      transition_accs.Read(ki.Stream(), binary_read, true /*add read values*/);
      gmm_accs.Read(ki.Stream(), binary_read, true /*add read values*/);
    }

    // Write out the accs
    {
      kaldi::Output ko(stats_out_filename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
	if (file_log) {
		file_log << "Summed " << num_accs << " stats, total count "
				 << gmm_accs.TotCount() << ", avg like/frame "
				 << (gmm_accs.TotLogLike() / gmm_accs.TotCount()) << "\n";
		file_log << "Total count of stats is " << gmm_accs.TotStatsCount() << "\n";
		file_log << "Written stats to " << stats_out_filename << "\n";
	}
	else {
		KALDI_LOG << "Summed " << num_accs << " stats, total count "
				  << gmm_accs.TotCount() << ", avg like/frame "
				  << (gmm_accs.TotLogLike() / gmm_accs.TotCount());
		KALDI_LOG << "Total count of stats is " << gmm_accs.TotStatsCount();
		KALDI_LOG << "Written stats to " << stats_out_filename;
	}
	return 0;

  } catch(const std::exception &e) {
    KALDI_ERR << e.what() << '\n';
    return -1;
  }
}


