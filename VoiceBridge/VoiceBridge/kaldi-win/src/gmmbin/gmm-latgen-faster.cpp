/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :   Copyright 2009-2012  Microsoft Corporation, Apache 2.0
					   2013-2014  Johns Hopkins University (author: Daniel Povey)
					   2014  Guoguo Chen
				See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "base/timer.h"
#include "feat/feature-functions.h"  // feature reversal

#include "kaldi-win/src/kaldi_src.h"

int GmmLatgenFaster(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using GMM-based model.\n"
        "Usage: gmm-latgen-faster [options] model-in (fst-in|fsts-rspecifier) features-rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;

    std::string word_syms_filename;
    config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      //po.PrintUsage();
      //exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lattice_wspecifier = po.GetArg(4),
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
	if (!(determinize ? compact_lattice_writer.Open(lattice_wspecifier)
		: lattice_writer.Open(lattice_wspecifier))) {
		KALDI_ERR << "Could not open table for writing lattices: "
			<< lattice_wspecifier;
		return -1; //VB
	}
    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
		if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
			KALDI_ERR << "Could not read symbol table from file "
				<< word_syms_filename;
			return -1; //VB
		}
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_done = 0, num_err = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      timer.Reset();

      {
        LatticeFasterDecoder decoder(*decode_fst, config);

        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          Matrix<BaseFloat> features (feature_reader.Value());
          feature_reader.FreeCurrent();
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_err++;
            continue;
          }

          DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                                 acoustic_scale);

          double like;
          if (DecodeUtteranceLatticeFaster(
                  decoder, gmm_decodable, trans_model, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, &alignment_writer,
                  &words_writer, &compact_lattice_writer, &lattice_writer,
                  &like)) {
            tot_like += like;
            frame_count += features.NumRows();
            num_done++;
          } else num_err++;
        }
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no features available.";
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }

        LatticeFasterDecoder decoder(fst_reader.Value(), config);
        DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                               acoustic_scale);
        double like;
        if (DecodeUtteranceLatticeFaster(
                decoder, gmm_decodable, trans_model, word_syms, utt,
                acoustic_scale, determinize, allow_partial, &alignment_writer,
                &words_writer, &compact_lattice_writer, &lattice_writer,
                &like)) {
          tot_like += like;
          frame_count += features.NumRows();
          num_done++;
        } else num_err++;
      }
    }

    double elapsed = timer.Elapsed();
	if (file_log) {
		file_log << "Time taken " << elapsed
					<< "s: real-time factor assuming 100 frames/sec is "
					<< (elapsed*100.0 / frame_count);
		file_log << "Done " << num_done << " utterances, failed for "
					<< num_err;
		file_log << "Overall log-likelihood per frame is " << (tot_like / frame_count) << " over "
					<< frame_count << " frames.";
	}
	else {
		KALDI_LOG << "Time taken " << elapsed
					<< "s: real-time factor assuming 100 frames/sec is "
					<< (elapsed*100.0 / frame_count);
		KALDI_LOG << "Done " << num_done << " utterances, failed for "
					<< num_err;
		KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like / frame_count) << " over "
					<< frame_count << " frames.";
	}
    delete word_syms;
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}
