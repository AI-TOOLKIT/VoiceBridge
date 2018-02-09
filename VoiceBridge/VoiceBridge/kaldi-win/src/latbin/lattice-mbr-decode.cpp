/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :
Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

See ../../COPYING for clarification regarding multiple authors
*/


#include "util/common-utils.h"
#include "lat/sausages.h"
#include "hmm/posterior.h"

#include "kaldi-win/src/kaldi_src.h"

int LatticeMbrDecode(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Do Minimum Bayes Risk decoding (decoding that aims to minimize the \n"
        "expected word error rate).  Possible outputs include the 1-best path\n"
        "(i.e. the word-sequence, as a sequence of ints per utterance), the\n"
        "computed Bayes Risk for each utterance, and the sausage stats as\n"
        "(for each utterance) std::vector<std::vector<std::pair<int32, float> > >\n"
        "for which we use the same I/O routines as for posteriors (type Posterior).\n"
        "times-wspecifier writes pairs of (start-time, end-time) in frames, for\n"
        "each sausage position, or for each one-best entry if --one-best-times=true.\n"
        "Note: use ark:/dev/null or the empty string for unwanted outputs.\n"
        "Note: times will only be very meaningful if you first use lattice-word-align.\n"
        "If you need ctm-format output, don't use this program but use lattice-to-ctm-conf\n"
        "with --decode-mbr=true.\n"
        "\n"
        "Usage: lattice-mbr-decode [options]  lattice-rspecifier "
        "transcriptions-wspecifier [ bayes-risk-wspecifier "
        "[ sausage-stats-wspecifier [ times-wspecifier] ] ] \n"
        " e.g.: lattice-mbr-decode --acoustic-scale=0.1 ark:1.lats "
        "'ark,t:|int2sym.pl -f 2- words.txt > text' ark:/dev/null ark:1.sau\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat lm_scale = 1.0;
    bool one_best_times = false;

    std::string word_syms_filename;
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for "
                "acoustic likelihoods");
    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "probabilities");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for "
                "words [for debug output]");
    po.Register("one-best-times", &one_best_times, "If true, output times "
                "corresponding to one-best, not whole sausage.");

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 5) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    std::string lats_rspecifier = po.GetArg(1),
        trans_wspecifier = po.GetArg(2),
        bayes_risk_wspecifier = po.GetOptArg(3),
        sausage_stats_wspecifier = po.GetOptArg(4),
        times_wspecifier = po.GetOptArg(5);

    // Read as compact lattice.
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    Int32VectorWriter trans_writer(trans_wspecifier);
    BaseFloatWriter bayes_risk_writer(bayes_risk_wspecifier);
    // Note: type Posterior = vector<vector<pair<int32,BaseFloat> > >
    // happens to be the same as needed for the sausage stats.
    PosteriorWriter sausage_stats_writer(sausage_stats_wspecifier);

    BaseFloatPairVectorWriter times_writer(times_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    int32 n_done = 0, n_words = 0;
    BaseFloat tot_bayes_risk = 0.0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);

      MinimumBayesRisk mbr(clat);

      if (trans_wspecifier != "")
        trans_writer.Write(key, mbr.GetOneBest());
      if (bayes_risk_wspecifier != "")
        bayes_risk_writer.Write(key, mbr.GetBayesRisk());
      if (sausage_stats_wspecifier != "")
        sausage_stats_writer.Write(key, mbr.GetSausageStats());
      if (times_wspecifier != "")
        times_writer.Write(key, one_best_times ? mbr.GetOneBestTimes() :
                           mbr.GetSausageTimes());

      n_done++;
      n_words += mbr.GetOneBest().size();
      tot_bayes_risk += mbr.GetBayesRisk();
    }

	if (file_log) {
		file_log << "Done " << n_done << " lattices.";
		file_log << "Average Bayes Risk per sentence is "
			<< (tot_bayes_risk / n_done) << " and per word, "
			<< (tot_bayes_risk / n_words);
	}
	else {
		KALDI_LOG << "Done " << n_done << " lattices.";
		KALDI_LOG << "Average Bayes Risk per sentence is "
			<< (tot_bayes_risk / n_done) << " and per word, "
			<< (tot_bayes_risk / n_words);
	}
    delete word_syms;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}
