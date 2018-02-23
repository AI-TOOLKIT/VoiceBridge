/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2013  Microsoft Corporation, Apache 2
					 Johns Hopkins University (Author: Daniel Povey)
See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"

#include "kaldi-win/src/kaldi_src.h"

/** @brief Write an equally spaced alignment (for getting training started).
*/
int AlignEqualCompiled(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =  "Write an equally spaced alignment (for getting training started)"
        "Usage:  align-equal-compiled <graphs-rspecifier> <features-rspecifier> <alignments-wspecifier>\n"
        "e.g.: \n"
        " align-equal-compiled 1.fsts scp:train.scp ark:equal.ali\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      //po.PrintUsage();
      //exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    std::string
        fst_rspecifier = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignment_wspecifier = po.GetArg(3);


    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    int32 done = 0, no_feat = 0, error = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      if (!feature_reader.HasKey(key)) {
        KALDI_WARN << "No features for utterance " << key;
        no_feat++;
      } else {
        const Matrix<BaseFloat> &features = feature_reader.Value(key);
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << key;
          error++;
          continue;
        }
        if (decode_fst.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty decoding graph for " << key;
          error++;
          continue;
        }

        VectorFst<StdArc> path;
        int32 rand_seed = StringHasher()(key); // StringHasher() produces new anonymous
        // object of type StringHasher; we then call operator () on it, with "key".
        if (EqualAlign(decode_fst, features.NumRows(), rand_seed, &path) ) {
          std::vector<int32> aligned_seq, words;
          StdArc::Weight w;
          GetLinearSymbolSequence(path, &aligned_seq, &words, &w);
          KALDI_ASSERT(aligned_seq.size() == features.NumRows());
          alignment_writer.Write(key, aligned_seq);
          done++;
        } else {
          KALDI_WARN << "AlignEqual: did not align utterence " << key;
          error++;
        }
      }
    }

    if (done != 0 && no_feat == 0 && error == 0) {
		if (file_log)
			file_log << "Success: done " << done << " utterances." << "\n";
		else
		KALDI_LOG << "Success: done " << done << " utterances.";
    } else {
      KALDI_WARN << "Computed " << done << " alignments; " << no_feat
                 << " lacked features, " << error
                 << " had other errors.";
    }
    if (done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}


