/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation
			See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "feat/feature-functions.h"

#include "kaldi-win/src/kaldi_src.h"

/*Splice features with left and right context (e.g. prior to LDA)*/
int SpliceFeats(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Splice features with left and right context (e.g. prior to LDA)\n"
        "Usage: splice-feats [options] <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: splice-feats scp:feats.scp ark:-\n";
        
    ParseOptions po(usage);
    int32 left_context = 4, right_context = 4;

    po.Register("left-context", &left_context, "Number of frames of left context");
    po.Register("right-context", &right_context, "Number of frames of right context");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments SpliceFeats.";
		return -1;
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    for (; !kaldi_reader.Done(); kaldi_reader.Next()) {
      Matrix<BaseFloat> spliced;
      SpliceFrames(kaldi_reader.Value(),
                   left_context,
                   right_context,
                   &spliced);
      kaldi_writer.Write(kaldi_reader.Key(), spliced);
    }
    return 0;
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}


