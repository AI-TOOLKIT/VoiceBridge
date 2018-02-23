/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :  Copyright 2013  Ehsan Variani, Apache 2.0.
					  2013,2016  Johns Hopkins University (Author: Daniel Povey)
			See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

#include "kaldi-win/src/kaldi_src.h"

//NOTE: if TLOG exists the log is going to the twin log (local file log and std::cout); thread safe.
int LatticeDepthPerFrame(int argc, char *argv[], fs::ofstream & file_log) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    using fst::VectorFst;
    using fst::StdArc;
    typedef StdArc::StateId StateId;

    const char *usage =
        "For each lattice, compute a vector of length (num-frames) saying how\n"
        "may arcs cross each frame.  See also lattice-depth\n"
        "Usage: lattice-depth-per-frame <lattice-rspecifier> <depth-wspecifier> [<lattice-wspecifier>]\n"
        "The final <lattice-wspecifier> allows you to write the input lattices out\n"
        "in case you want to do something else with them as part of the same pipe.\n"
        "E.g.: lattice-depth-per-frame ark:- ark,t:-\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    std::string lats_rspecifier = po.GetArg(1);
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    std::string depth_wspecifier = po.GetArg(2);
    Int32VectorWriter lats_depth_writer(depth_wspecifier);

    std::string lattice_wspecifier = po.GetOptArg(3);
    CompactLatticeWriter clat_writer(lattice_wspecifier);

    int64 num_done = 0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      CompactLattice clat = clat_reader.Value();
      std::string key = clat_reader.Key();

      TopSortCompactLatticeIfNeeded(&clat);

      std::vector<int32> depth_per_frame;
      CompactLatticeDepthPerFrame(clat, &depth_per_frame);

      lats_depth_writer.Write(key, depth_per_frame);

      if (!lattice_wspecifier.empty())
        clat_writer.Write(key, clat);

      num_done++;
    }
	if(file_log)
		file_log << "Done " << num_done << " lattices.";
	else
    KALDI_LOG << "Done " << num_done << " lattices.";
    if (num_done != 0) return 0;
    else return 1;
  } catch (const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
