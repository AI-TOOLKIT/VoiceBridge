/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2014  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
		   See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "transform/cmvn.h"

#include "kaldi-win/src/kaldi_src.h"

int ModifyCmvnStats(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy cepstral mean/variance stats so that some dimensions have 'fake' stats\n"
        "that will skip normalization\n"
        "Usage: modify-cmvn-stats [options] [<fake-dims>] <in-rspecifier> <out-wspecifier>\n"
        "e.g.: modify-cmvn-stats 13:14:15 ark:- ark:-\n"
        "or: modify-cmvn-stats --convert-to-mean-and-var=true ark:- ark:-\n"
        "See also: compute-cmvn-stats\n";

    bool convert_to_mean_and_var = false;

    ParseOptions po(usage);

    po.Register("convert-to-mean-and-var", &convert_to_mean_and_var,
                "If true, convert the stats to a matrix containing the mean "
                "and the centered variance in each dimension");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "wrong arguments.";
		return -1;
    }

    int32 num_done = 0;

    
    std::string skip_dims_str, rspecifier, wspecifier;
    if (po.NumArgs() == 3) {
      skip_dims_str = po.GetArg(1);
      rspecifier = po.GetArg(2);
      wspecifier = po.GetArg(3);
    } else {
      rspecifier = po.GetArg(1);
      wspecifier = po.GetArg(2);
    }      

    std::vector<int32> skip_dims;
    if (!SplitStringToIntegers(skip_dims_str, ":", false, &skip_dims)) {
      KALDI_ERR << "Bad first argument (should be colon-separated list of "
                <<  "integers)";
    }
    
    SequentialDoubleMatrixReader reader(rspecifier);
    DoubleMatrixWriter writer(wspecifier);

    for (; !reader.Done(); reader.Next()) {
      Matrix<double> mat(reader.Value());

      if (mat.NumRows() != 2)
        KALDI_ERR << "Expected input to be CMVN stats (should have two rows)";

      FakeStatsForSomeDims(skip_dims, &mat);
      if (!convert_to_mean_and_var) {
        writer.Write(reader.Key(), mat);
        num_done++;
      } else {
        int32 dim = mat.NumCols() - 1;
        double count = mat(0, dim);
        Matrix<double> modified_mat(2, dim);
        if (count <= 0.0) {
          KALDI_WARN << "Zero or negative count for speaker " << reader.Key()
                     << ", not outputting mean and variance stats.";
          continue;
        }
        for (int32 i = 0; i < dim; i++) {
          double mean = mat(0, i) / count,
              variance = mat(1, i) / count - mean * mean;
          modified_mat(0, i) = mean;
          modified_mat(1, i) = variance;
        }  
        writer.Write(reader.Key(), modified_mat);
        num_done++;
      }
    }
	if (file_log)
		file_log << "Modified " << num_done << " sets of stats." << "\n";
	else
    KALDI_LOG << "Modified " << num_done << " sets of stats.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}


