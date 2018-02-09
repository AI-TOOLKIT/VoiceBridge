/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :
 Copyright 2009-2013  Microsoft Corporation
                      Johns Hopkins University (author: Daniel Povey)
See ../../COPYING for clarification regarding multiple authors
*/


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

#include "kaldi-win/src/kaldi_src.h"

int LatticeScale(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Apply scaling to lattice weights\n"
        "Usage: lattice-scale [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-scale --lm-scale=0.0 ark:1.lats ark:scaled.lats\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat inv_acoustic_scale = 1.0;
    BaseFloat lm_scale = 1.0;
    BaseFloat acoustic2lm_scale = 0.0;
    BaseFloat lm2acoustic_scale = 0.0;

    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("inv-acoustic-scale", &inv_acoustic_scale, "An alternative way "
                "of setting the acoustic scale: you can set its inverse.");
    po.Register("lm-scale", &lm_scale, "Scaling factor for graph/lm costs");
    po.Register("acoustic2lm-scale", &acoustic2lm_scale, "Add this times original acoustic costs to LM costs");
    po.Register("lm2acoustic-scale", &lm2acoustic_scale, "Add this times original LM costs to acoustic costs");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);

    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);

    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0;

    KALDI_ASSERT(acoustic_scale == 1.0 || inv_acoustic_scale == 1.0);
    if (inv_acoustic_scale != 1.0)
      acoustic_scale = 1.0 / inv_acoustic_scale;

    std::vector<std::vector<double> > scale(2);
    scale[0].resize(2);
    scale[1].resize(2);
    scale[0][0] = lm_scale;
    scale[0][1] = acoustic2lm_scale;
    scale[1][0] = lm2acoustic_scale;
    scale[1][1] = acoustic_scale;

    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      CompactLattice lat = compact_lattice_reader.Value();
      ScaleLattice(scale, &lat);
      compact_lattice_writer.Write(compact_lattice_reader.Key(), lat);
      n_done++;
    }
	if(file_log)
		file_log << "Done " << n_done << " lattices.";
	else
		KALDI_LOG << "Done " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}
