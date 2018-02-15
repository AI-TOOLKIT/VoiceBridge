/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : 2009-2012  Microsoft Corporation
		   Copyright 2013  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
		   See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "transform/cmvn.h"

#include "kaldi-win/src/kaldi_src.h"

namespace kaldi {

bool AccCmvnStatsWrapper(std::string utt,
                         const MatrixBase<BaseFloat> &feats,
                         RandomAccessBaseFloatVectorReader *weights_reader,
                         Matrix<double> *cmvn_stats) {
  if (!weights_reader->IsOpen()) {
    AccCmvnStats(feats, NULL, cmvn_stats);
    return true;
  } else {
    if (!weights_reader->HasKey(utt)) {
      KALDI_WARN << "No weights available for utterance " << utt;
      return false;
    }
    const Vector<BaseFloat> &weights = weights_reader->Value(utt);
    if (weights.Dim() != feats.NumRows()) {
      KALDI_WARN << "Weights for utterance " << utt << " have wrong dimension "
                 << weights.Dim() << " vs. " << feats.NumRows();
      return false;
    }
    AccCmvnStats(feats, &weights, cmvn_stats);
    return true;
  }
}
                  

}

int ComputeCmvnStats(int argc, char *argv[], fs::ofstream & file_log) 
{
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Compute cepstral mean and variance normalization statistics\n"
        "If wspecifier provided: per-utterance by default, or per-speaker if\n"
        "spk2utt option provided; if wxfilename: global\n"
        "Usage: compute-cmvn-stats  [options] <feats-rspecifier> (<stats-wspecifier>|<stats-wxfilename>)\n"
        "e.g.: compute-cmvn-stats --spk2utt=ark:data/train/spk2utt"
        " scp:data/train/feats.scp ark,scp:/foo/bar/cmvn.ark,data/train/cmvn.scp\n"
        "See also: apply-cmvn, modify-cmvn-stats\n";
    
    ParseOptions po(usage);
    std::string spk2utt_rspecifier, weights_rspecifier;
    bool binary = true;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to utterance-list map");
    po.Register("binary", &binary, "write in binary mode (applies only to global CMN/CVN)");
    po.Register("weights", &weights_rspecifier, "rspecifier for a vector of floats "
                "for each utterance, that's a per-frame weight.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    int32 num_done = 0, num_err = 0;
    std::string rspecifier = po.GetArg(1);
    std::string wspecifier_or_wxfilename = po.GetArg(2);

    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);
    
    if (ClassifyWspecifier(wspecifier_or_wxfilename, NULL, NULL, NULL)
        != kNoWspecifier) { // writing to a Table: per-speaker or per-utt CMN/CVN.
      std::string wspecifier = wspecifier_or_wxfilename;

      DoubleMatrixWriter writer(wspecifier);

      if (spk2utt_rspecifier != "") {
        SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
        RandomAccessBaseFloatMatrixReader feat_reader(rspecifier);
        
        for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
          std::string spk = spk2utt_reader.Key();
          const std::vector<std::string> &uttlist = spk2utt_reader.Value();
          bool is_init = false;
          Matrix<double> stats;
          for (size_t i = 0; i < uttlist.size(); i++) {
            std::string utt = uttlist[i];
            if (!feat_reader.HasKey(utt)) {
              KALDI_WARN << "Did not find features for utterance " << utt;
              num_err++;
              continue;
            }
            const Matrix<BaseFloat> &feats = feat_reader.Value(utt);
            if (!is_init) {
              InitCmvnStats(feats.NumCols(), &stats);
              is_init = true;
            }
            if (!AccCmvnStatsWrapper(utt, feats, &weights_reader, &stats)) {
              num_err++;
            } else {
              num_done++;
            }
          }
          if (stats.NumRows() == 0) {
            KALDI_WARN << "No stats accumulated for speaker " << spk;
          } else {
            writer.Write(spk, stats);
          }
        }
      } else {  // per-utterance normalization
        SequentialBaseFloatMatrixReader feat_reader(rspecifier);
        
        for (; !feat_reader.Done(); feat_reader.Next()) {
          std::string utt = feat_reader.Key();
          Matrix<double> stats;
          const Matrix<BaseFloat> &feats = feat_reader.Value();
          InitCmvnStats(feats.NumCols(), &stats);

          if (!AccCmvnStatsWrapper(utt, feats, &weights_reader, &stats)) {
            num_err++;
            continue;
          }
          writer.Write(feat_reader.Key(), stats);          
          num_done++;
        }
      }
    } else { // accumulate global stats
		if (spk2utt_rspecifier != "") {
			KALDI_ERR << "--spk2utt option not compatible with wxfilename as output "
				<< "(did you forget ark:?)";
			return -1; //VB
		}
      std::string wxfilename = wspecifier_or_wxfilename;
      bool is_init = false;
      Matrix<double> stats;
      SequentialBaseFloatMatrixReader feat_reader(rspecifier);
      for (; !feat_reader.Done(); feat_reader.Next()) {
        std::string utt = feat_reader.Key();
        const Matrix<BaseFloat> &feats = feat_reader.Value();
        if (!is_init) {
          InitCmvnStats(feats.NumCols(), &stats);
          is_init = true;
        }
        if (!AccCmvnStatsWrapper(utt, feats, &weights_reader, &stats)) {
          num_err++;
        } else {
          num_done++;
        }
      }
      Matrix<float> stats_float(stats);
      WriteKaldiObject(stats_float, wxfilename, binary);
	  if (file_log)
		  file_log << "Wrote global CMVN stats to " << PrintableWxfilename(wxfilename) << "\n";
	  else
      KALDI_LOG << "Wrote global CMVN stats to " << PrintableWxfilename(wxfilename);
    }
	if (file_log)
		file_log << "Done accumulating CMVN stats for " << num_done << " utterances; " << num_err << " had errors." << "\n";
	else
    KALDI_LOG << "Done accumulating CMVN stats for " << num_done << " utterances; " << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);    
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}


