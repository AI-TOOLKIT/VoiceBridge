/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :
	Copyright 2009-2011  Microsoft Corporation
	2013 Johns Hopkins University (author: Daniel Povey)
	Apache 2.0
	See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

#include "kaldi-win/src/kaldi_src.h"


int CopyFeats(int argc, char *argv[], fs::ofstream & file_log)
{
  try {
    using namespace kaldi;
    const char *usage =
        "Copy features [and possibly change format]\n"
        "Usage: copy-feats [options] <feature-rspecifier> <feature-wspecifier>\n"
        "or:   copy-feats [options] <feats-rxfilename> <feats-wxfilename>\n"
        "e.g.: copy-feats ark:- ark,scp:foo.ark,foo.scp\n"
        " or: copy-feats ark:foo.ark ark,t:txt.ark\n"
        "See also: copy-matrix, copy-feats-to-htk, copy-feats-to-sphinx, select-feats,\n"
        "extract-feature-segments, subset-feats, subsample-feats, splice-feats, paste-feats,\n"
        "concat-feats\n";

    ParseOptions po(usage);
    bool binary = true;
    bool htk_in = false;
    bool sphinx_in = false;
    bool compress = false;
    int32 compression_method_in = 1;
    std::string num_frames_wspecifier;
    po.Register("htk-in", &htk_in, "Read input as HTK features");
    po.Register("sphinx-in", &sphinx_in, "Read input as Sphinx features");
    po.Register("binary", &binary, "Binary-mode output (not relevant if writing "
                "to archive)");
    po.Register("compress", &compress, "If true, write output in compressed form"
                "(only currently supported for wxfilename, i.e. archive/script,"
                "output)");
    po.Register("compression-method", &compression_method_in,
                "Only relevant if --compress=true; the method (1 through 7) to "
                "compress the matrix.  Search for CompressionMethod in "
                "src/matrix/compressed-matrix.h.");
    po.Register("write-num-frames", &num_frames_wspecifier,
                "Wspecifier to write length in frames of each utterance. "
                "e.g. 'ark,t:utt2num_frames'.  Only applicable if writing tables, "
                "not when this program is writing individual files.  See also "
                "feat-to-len.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    int32 num_done = 0;

    CompressionMethod compression_method = static_cast<CompressionMethod>(
        compression_method_in);

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      // Copying tables of features.
      std::string rspecifier = po.GetArg(1);
      std::string wspecifier = po.GetArg(2);
      Int32Writer num_frames_writer(num_frames_wspecifier);

      if (!compress) {
        BaseFloatMatrixWriter kaldi_writer(wspecifier);
        if (htk_in) {
          SequentialTableReader<HtkMatrixHolder> htk_reader(rspecifier);
          for (; !htk_reader.Done(); htk_reader.Next(), num_done++) {
            kaldi_writer.Write(htk_reader.Key(), htk_reader.Value().first);
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(htk_reader.Key(),
                                      htk_reader.Value().first.NumRows());
          }
        } else if (sphinx_in) {
          SequentialTableReader<SphinxMatrixHolder<> > sphinx_reader(rspecifier);
          for (; !sphinx_reader.Done(); sphinx_reader.Next(), num_done++) {
            kaldi_writer.Write(sphinx_reader.Key(), sphinx_reader.Value());
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(sphinx_reader.Key(),
                                      sphinx_reader.Value().NumRows());
          }
        } else {
          SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
          for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++) {
            kaldi_writer.Write(kaldi_reader.Key(), kaldi_reader.Value());
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(kaldi_reader.Key(),
                                      kaldi_reader.Value().NumRows());
          }
        }
      } else {
        CompressedMatrixWriter kaldi_writer(wspecifier);
        if (htk_in) {
          SequentialTableReader<HtkMatrixHolder> htk_reader(rspecifier);
          for (; !htk_reader.Done(); htk_reader.Next(), num_done++) {
            kaldi_writer.Write(htk_reader.Key(),
                               CompressedMatrix(htk_reader.Value().first,
                                                compression_method));
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(htk_reader.Key(),
                                      htk_reader.Value().first.NumRows());
          }
        } else if (sphinx_in) {
          SequentialTableReader<SphinxMatrixHolder<> > sphinx_reader(rspecifier);
          for (; !sphinx_reader.Done(); sphinx_reader.Next(), num_done++) {
            kaldi_writer.Write(sphinx_reader.Key(),
                               CompressedMatrix(sphinx_reader.Value(),
                                                compression_method));
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(sphinx_reader.Key(),
                                      sphinx_reader.Value().NumRows());
          }
        } else {
          SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
          for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++) {
            kaldi_writer.Write(kaldi_reader.Key(),
                               CompressedMatrix(kaldi_reader.Value(),
                                                compression_method));
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(kaldi_reader.Key(),
                                      kaldi_reader.Value().NumRows());
          }
        }
      }
	  if (file_log)
		file_log << "Copied " << num_done << " feature matrices." << "\n";
	  else KALDI_LOG << "Copied " << num_done << " feature matrices." << "\n";

      return (num_done != 0 ? 0 : 1);
    } else {
      KALDI_ASSERT(!compress && "Compression not yet supported for single files");
	  if (!num_frames_wspecifier.empty()) {
		  KALDI_ERR << "--write-num-frames option not supported when writing/reading "
			  << "single files.";
		  return -1; //VB
	  }

      std::string feat_rxfilename = po.GetArg(1), feat_wxfilename = po.GetArg(2);

      Matrix<BaseFloat> feat_matrix;
      if (htk_in) {
        Input ki(feat_rxfilename); // Doesn't look for read binary header \0B, because
        // no bool* pointer supplied.
        HtkHeader header; // we discard this info.
        ReadHtk(ki.Stream(), &feat_matrix, &header);
      } else if (sphinx_in) {
        KALDI_ERR << "For single files, sphinx input is not yet supported.";
		return -1; //VB
      } else {
        ReadKaldiObject(feat_rxfilename, &feat_matrix);
      }
      WriteKaldiObject(feat_matrix, feat_wxfilename, binary);
	  if (file_log)
		file_log << "Copied features from " << PrintableRxfilename(feat_rxfilename)
                 << " to " << PrintableWxfilename(feat_wxfilename) << "\n";
	  else KALDI_LOG << "Copied features from " << PrintableRxfilename(feat_rxfilename)
					 << " to " << PrintableWxfilename(feat_wxfilename) << "\n";

	  return 0;
    }
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
