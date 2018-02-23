/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation, Apache 2
						  2015 Johns Hopkins University (Author: Daniel Povey)
			See ../../COPYING for clarification regarding multiple authors
*/

#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"

/** @brief Add self-loops and transition probabilities to transducer, expanding to transition-ids.
*/
int AddSelfLoops(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Add self-loops and transition probabilities to transducer.  Input transducer\n"
        "has transition-ids on the input side, but only the forward transitions, not the\n"
        "self-loops.  Output transducer has transition-ids on the input side, but with\n"
        "self-loops added.  The --reorder option controls whether the loop is added before\n"
        "the forward transition (if false), or afterward (if true).  The default (true)\n"
        "is recommended as the decoding will in that case be faster.\n"
        "Usage:   add-self-loops [options] transition-gmm/acoustic-model [fst-in] [fst-out]\n"
        "e.g.: \n"
        " add-self-loops --self-loop-scale=0.1 1.mdl HCLGa.fst HCLG.fst\n"
        "or:  add-self-loops --self-loop-scale=0.1 1.mdl <HCLGa.fst >HCLG.fst\n";

    BaseFloat self_loop_scale = 1.0;
    bool reorder = true;
    std::string disambig_in_filename;

    ParseOptions po(usage);
    po.Register("self-loop-scale", &self_loop_scale,
                "Scale for self-loop probabilities relative to LM.");
    po.Register("disambig-syms", &disambig_in_filename,
                "List of disambiguation symbols on input of fst-in [input file]");
    po.Register("reorder", &reorder,
                "If true, reorder symbols for more decoding efficiency");
    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 3) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    std::string model_in_filename = po.GetArg(1);
    std::string fst_in_filename = po.GetOptArg(2);
    if (fst_in_filename == "-") fst_in_filename = "";
    std::string fst_out_filename = po.GetOptArg(3);
    if (fst_out_filename == "-") fst_out_filename = "";
#if _MSC_VER
    if (fst_in_filename == "")
      _setmode(_fileno(stdin),  _O_BINARY);
    if (fst_out_filename == "")
      _setmode(_fileno(stdout),  _O_BINARY);
#endif

    std::vector<int32> disambig_syms_in;
    if (disambig_in_filename != "") {
      if (disambig_in_filename == "-") disambig_in_filename = "";
	  if (!ReadIntegerVectorSimple(disambig_in_filename, &disambig_syms_in)) {
		  KALDI_ERR << "add-self-loops: could not read disambig symbols from "
			  << (disambig_in_filename == "" ?
				  "standard input" : disambig_in_filename);
		  return -1; //VB
	  }
    }

    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);


    fst::VectorFst<fst::StdArc> *fst =
        fst::VectorFst<fst::StdArc>::Read(fst_in_filename);
	if (!fst) {
		KALDI_ERR << "add-self-loops: error reading input FST.";
		return -1; //VB
	}

    // The work gets done here.
    AddSelfLoops(trans_model,
                 disambig_syms_in,
                 self_loop_scale,
                 reorder,
                 fst);

	if (!fst->Write(fst_out_filename)) {
		KALDI_ERR << "add-self-loops: error writing FST to "
			<< (fst_out_filename == "" ?
				"standard output" : fst_out_filename);
		return -1; //VB
	}

    delete fst;
    return 0;
  } catch(const std::exception &e) {
		KALDI_ERR << e.what();
		return -1;
  }
}

