/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2012  Microsoft Corporation
		   Apache 2.0
		   See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

#include "kaldi-win/src/kaldi_src.h"

int FeatToDim(int argc, char *argv[]) 
{
  try {
    using namespace kaldi;

    const char *usage =
        "Reads an archive of features.  If second argument is wxfilename, writes\n"
        "the feature dimension of the first feature file; if second argument is\n"
        "wspecifier, writes an archive of the feature dimension, indexed by utterance\n"
        "id.\n"
        "Usage: feat-to-dim [options] <feat-rspecifier> (<dim-wspecifier>|<dim-wxfilename>)\n"
        "e.g.: feat-to-dim scp:feats.scp -\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "wrong arguments.";
		return -1;
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier_or_wxfilename = po.GetArg(2);

    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
      
    if (ClassifyWspecifier(wspecifier_or_wxfilename, NULL, NULL, NULL)
        != kNoWspecifier) {
      Int32Writer dim_writer(wspecifier_or_wxfilename);
      for (; !kaldi_reader.Done(); kaldi_reader.Next())
        dim_writer.Write(kaldi_reader.Key(), kaldi_reader.Value().NumCols());
    } else {
		if (kaldi_reader.Done()) {
			KALDI_ERR << "Could not read any features (empty archive?).";
			return -1; //VB
		}
      Output ko(wspecifier_or_wxfilename, false); // text mode.
      ko.Stream() << kaldi_reader.Value().NumCols() << "\n";
    }
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}


