/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation, Apache 2.0
		   See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "transform/transform-common.h"

#include "kaldi-win/src/kaldi_src.h"

namespace kaldi {

void ApplySoftMaxPerRow(MatrixBase<BaseFloat> *mat) {
  for (int32 i = 0; i < mat->NumRows(); i++) {
    mat->Row(i).ApplySoftMax();
  }
}

}  // namespace kaldi

int CopyMatrix(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy matrices, or archives of matrices (e.g. features or transforms)\n"
        "Also see copy-feats which has other format options\n"
        "\n"
        "Usage: copy-matrix [options] <matrix-in-rspecifier> <matrix-out-wspecifier>\n"
        "  or: copy-matrix [options] <matrix-in-rxfilename> <matrix-out-wxfilename>\n"
        " e.g.: copy-matrix --binary=false 1.mat -\n"
        "   copy-matrix ark:2.trans ark,t:-\n"
        "See also: copy-feats\n";

    bool binary = true;
    bool apply_log = false;
    bool apply_exp = false;
    bool apply_softmax_per_row = false;
    BaseFloat apply_power = 1.0;
    BaseFloat scale = 1.0;

    ParseOptions po(usage);

    po.Register("binary", &binary,
                "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("scale", &scale,
                "This option can be used to scale the matrices being copied.");
    po.Register("apply-log", &apply_log,
                "This option can be used to apply log on the matrices. "
                "Must be avoided if matrix has negative quantities.");
    po.Register("apply-exp", &apply_exp,
                "This option can be used to apply exp on the matrices");
    po.Register("apply-power", &apply_power,
                "This option can be used to apply a power on the matrices");
    po.Register("apply-softmax-per-row", &apply_softmax_per_row,
                "This option can be used to apply softmax per row of the matrices");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
        //po.PrintUsage();
        //exit(1);
		KALDI_ERR << "wrong arguments.";
		return -1;
    }

	if ((apply_log && apply_exp) || (apply_softmax_per_row && apply_exp) ||
		(apply_softmax_per_row && apply_log)) {
		KALDI_ERR << "Only one of apply-log, apply-exp and "
			<< "apply-softmax-per-row can be given";
		return -1; //VB
	}

    std::string matrix_in_fn = po.GetArg(1),
        matrix_out_fn = po.GetArg(2);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(matrix_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(matrix_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

	if (in_is_rspecifier != out_is_wspecifier) {
		KALDI_ERR << "Cannot mix archives with regular files (copying matrices)";
		return -1; //VB
	}

    if (!in_is_rspecifier) {
      Matrix<BaseFloat> mat;
      ReadKaldiObject(matrix_in_fn, &mat);
      if (scale != 1.0) mat.Scale(scale);
      if (apply_log) {
        mat.ApplyFloor(1.0e-20);
        mat.ApplyLog();
      }
      if (apply_exp) mat.ApplyExp();
      if (apply_softmax_per_row) ApplySoftMaxPerRow(&mat);
      if (apply_power != 1.0) mat.ApplyPow(apply_power);
      Output ko(matrix_out_fn, binary);
      mat.Write(ko.Stream(), binary);
	  if(file_log) file_log << "Copied matrix to " << matrix_out_fn << "\n";
	  else KALDI_LOG << "Copied matrix to " << matrix_out_fn;
      return 0;
    } else {
      int num_done = 0;
      BaseFloatMatrixWriter writer(matrix_out_fn);
      SequentialBaseFloatMatrixReader reader(matrix_in_fn);
      for (; !reader.Done(); reader.Next(), num_done++) {
        if (scale != 1.0 || apply_log || apply_exp ||
              apply_power != 1.0 || apply_softmax_per_row) {
          Matrix<BaseFloat> mat(reader.Value());
          if (scale != 1.0) mat.Scale(scale);
          if (apply_log) {
            mat.ApplyFloor(1.0e-20);
            mat.ApplyLog();
          }
          if (apply_exp) mat.ApplyExp();
          if (apply_softmax_per_row) ApplySoftMaxPerRow(&mat);
          if (apply_power != 1.0) mat.ApplyPow(apply_power);
          writer.Write(reader.Key(), mat);
        } else {
          writer.Write(reader.Key(), reader.Value());
        }
      }
	  if (file_log)
		file_log << "Copied " << num_done << " matrices." << "\n";
	  else KALDI_LOG << "Copied " << num_done << " matrices.";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}


