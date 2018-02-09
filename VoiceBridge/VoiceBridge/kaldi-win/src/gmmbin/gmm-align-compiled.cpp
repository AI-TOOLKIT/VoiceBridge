/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :  Copyright 2009-2013  Microsoft Corporation, Apache 2.0
								 Johns Hopkins University (Author: Daniel Povey)
			See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc

#include "kaldi-win/src/kaldi_src.h"

int GmmAlignCompiled(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Align features given [GMM-based] models.\n"
        "Usage:   gmm-align-compiled [options] <model-in> <graphs-rspecifier> "
        "<feature-rspecifier> <alignments-wspecifier> [scores-wspecifier]\n"
        "e.g.: \n"
        " gmm-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.ali\n"
        "or:\n"
        " compile-train-graphs tree 1.mdl lex.fst 'ark:sym2int.pl -f 2- words.txt text|' \\\n"
        "   ark:- | gmm-align-compiled 1.mdl ark:- scp:train.scp t, ark:1.ali\n";

    ParseOptions po(usage);
    AlignConfig align_config;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;
    std::string per_frame_acwt_wspecifier;

    align_config.Register(&po);
    po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("self-loop-scale", &self_loop_scale,
                "Scale of self-loop versus non-self-loop log probs [relative to acoustics]");
    po.Register("write-per-frame-acoustic-loglikes", &per_frame_acwt_wspecifier,
                "Wspecifier for table of vectors containing the acoustic log-likelihoods "
                "per frame for each utterance. E.g. ark:foo/per_frame_logprobs.1.ark");
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 5) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "wrong arguments.";
		return -1;
    }

    std::string model_in_filename = po.GetArg(1),
        fst_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        alignment_wspecifier = po.GetArg(4),
        scores_wspecifier = po.GetOptArg(5);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);
    BaseFloatWriter scores_writer(scores_wspecifier);
    BaseFloatVectorWriter per_frame_acwt_writer(per_frame_acwt_wspecifier);

    int num_done = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string utt = fst_reader.Key();
      if (!feature_reader.HasKey(utt)) {
        num_err++;
        KALDI_WARN << "No features for utterance " << utt;
      } else {
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }

        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             &decode_fst);
        }

        DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                               acoustic_scale);

		if (file_log)
			file_log << utt;
		else
        KALDI_LOG << utt;

        AlignUtteranceWrapper(align_config, utt,
                              acoustic_scale, &decode_fst, &gmm_decodable,
                              &alignment_writer, &scores_writer,
                              &num_done, &num_err, &num_retry,
                              &tot_like, &frame_count, &per_frame_acwt_writer);
      }
    }
	if (file_log) {
		file_log << "Overall log-likelihood per frame is " << (tot_like / frame_count) << " over " << frame_count << " frames." << "\n";
		file_log << "Retried " << num_retry << " out of " << (num_done + num_err) << " utterances." << "\n";
		file_log << "Done " << num_done << ", errors on " << num_err << "\n";
	}
	else {
		KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like / frame_count) << " over " << frame_count << " frames.";
		KALDI_LOG << "Retried " << num_retry << " out of " << (num_done + num_err) << " utterances.";
		KALDI_LOG << "Done " << num_done << ", errors on " << num_err;
	}

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
