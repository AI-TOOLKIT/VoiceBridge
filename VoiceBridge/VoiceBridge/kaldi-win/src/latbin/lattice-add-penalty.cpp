/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2013     Bagher BabaAli
							  Johns Hopkins University (Author: Daniel Povey)
				See ../../COPYING for clarification regarding multiple authors
*/

#include "lat/lattice-functions.h"

#include "kaldi-win/src/kaldi_src.h"

int LatticeAddPenalty(int argc, char *argv[], fs::ofstream & file_log) {
  using namespace kaldi;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Add word insertion penalty to the lattice.\n"
        "Note: penalties are negative log-probs, base e, and are added to the\n"
        "'language model' part of the cost.\n"
        "\n"
        "Usage: lattice-add-penalty [options] <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-add-penalty --word-ins-penalty=1.0 ark:- ark:-\n";
      
    ParseOptions po(usage);
    
    BaseFloat word_ins_penalty = 0.0;

    po.Register("word-ins-penalty", &word_ins_penalty, "Word insertion penalty");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);
    
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    CompactLatticeWriter clat_writer(lats_wspecifier); // write as compact.

    int64 n_done = 0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      CompactLattice clat(clat_reader.Value());
      AddWordInsPenToCompactLattice(word_ins_penalty, &clat);
      clat_writer.Write(clat_reader.Key(), clat);
      n_done++;
    }
	if(file_log)
		file_log << "Done adding word insertion penalty to " << n_done << " lattices.";
	else
	KALDI_LOG << "Done adding word insertion penalty to " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}
