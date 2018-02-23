/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on :  Copyright 2009-2012   Microsoft Corporation, Apache 2.0
								      Johns Hopkins University (Author: Daniel Povey)
									  See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/mle-am-diag-gmm.h"

#include "kaldi-win/src/kaldi_src.h"

//Accumulate stats for GMM training.
int GmmAccStatsAli(int argc, char *argv[], fs::ofstream & file_log) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Accumulate stats for GMM training.\n"
        "Usage:  gmm-acc-stats-ali [options] <model-in> <feature-rspecifier> "
        "<alignments-rspecifier> <stats-out>\n"
        "e.g.:\n gmm-acc-stats-ali 1.mdl scp:train.scp ark:1.ali 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);
    AccumAmDiagGmm gmm_accs;
    gmm_accs.Init(am_gmm, kGmmAll);

    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!alignments_reader.HasKey(key)) {
        KALDI_WARN << "No alignment for utterance " << key;
        num_err++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(key);

        if (alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size " << (alignment.size())
                     << " vs. " << (mat.NumRows());
          num_err++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0;

        for (size_t i = 0; i < alignment.size(); i++) {
          int32 tid = alignment[i],  // transition identifier.
              pdf_id = trans_model.TransitionIdToPdf(tid);
          trans_model.Accumulate(1.0, tid, &transition_accs);
          tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, mat.Row(i), pdf_id, 1.0);
        }
        tot_like += tot_like_this_file;
        tot_t += alignment.size();
        if (num_done % 50 == 0) {
			if(file_log)
				file_log << "Processed " << num_done << " utterances; for utterance "
						 << key << " avg. like is "
						 << (tot_like_this_file / alignment.size())
						 << " over " << alignment.size() << " frames." << "\n";
			else
			  KALDI_LOG << "Processed " << num_done << " utterances; for utterance "
						<< key << " avg. like is "
						<< (tot_like_this_file/alignment.size())
						<< " over " << alignment.size() <<" frames.";
        }
      }
    }
	if (file_log) {
		file_log << "Done " << num_done << " files, " << num_err << " with errors." << "\n";

		file_log << "Overall avg like per frame (Gaussian only) = "
				 << (tot_like / tot_t) << " over " << tot_t << " frames." << "\n";
	}
	else {
		KALDI_LOG << "Done " << num_done << " files, " << num_err << " with errors.";

		KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
				  << (tot_like / tot_t) << " over " << tot_t << " frames.";
	}

    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
	if (file_log)
		file_log << "Written accs." << "\n";
	else
		KALDI_LOG << "Written accs.";
    if (num_done != 0)
      return 0;
    else
      return 1;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}


