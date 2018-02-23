/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation, Apache 2.0
			See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-functions.h"
#include "matrix/kaldi-matrix.h"

#include "kaldi-win/src/kaldi_src.h"

int AddDeltas(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Add deltas (typically to raw mfcc or plp features\n"
        "Usage: add-deltas [options] in-rspecifier out-wspecifier\n";
    DeltaFeaturesOptions opts;
    int32 truncate = 0;
    ParseOptions po(usage);
    po.Register("truncate", &truncate, "If nonzero, first truncate features to this dimension.");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      //po.PrintUsage();
      //exit(1);
	  KALDI_ERR << "Wrong arguments AddDeltas.";
	  return -1;
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatMatrixWriter feat_writer(wspecifier);
    SequentialBaseFloatMatrixReader feat_reader(rspecifier);
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats  = feat_reader.Value();

      if (feats.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for key " << key;
        continue;
      }
      Matrix<BaseFloat> new_feats;
      if (truncate != 0) {
		  if (truncate > feats.NumCols()) {
			  KALDI_ERR << "Cannot truncate features as dimension " << feats.NumCols()
				  << " is smaller than truncation dimension.";
			  return -1; //VB
		  }
        SubMatrix<BaseFloat> feats_sub(feats, 0, feats.NumRows(), 0, truncate);
        ComputeDeltas(opts, feats_sub, &new_feats);
      } else
        ComputeDeltas(opts, feats, &new_feats);
      feat_writer.Write(key, new_feats);
    }
    return 0;
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}


