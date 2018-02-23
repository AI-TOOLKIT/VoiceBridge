/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation, Apache 2.0
			See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "util/parse-options.h"
#include "util/text-utils.h"
#include "fst/fstlib.h"
#include "fstext/determinize-star.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"


/*
 A test example:
 ( echo "0 1 1 0"; echo "1 2 0 2"; echo "2 0"; ) | fstcompile | fstrmepslocal | fstprint
# prints:
# 0     1    1    2
# 1
 ( echo "0 1 0 0"; echo "0 0"; echo "1 0" ) | fstcompile | fstrmepslocal | fstprint
# 0
  ( echo "0 1 0 0"; echo "0 0"; echo "1 0" ) | fstcompile | fstrmepslocal | fstprint
  ( echo "0 1 0 0"; echo "0 0"; echo "1 0" ) | fstcompile | fstrmepslocal --use-log=true | fstprint
#  0    -0.693147182

*/

int fstrmepslocal(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		using namespace fst;
		using kaldi::int32;

		const char *usage =
			"Removes some (but not all) epsilons in an algorithm that will always reduce the number of\n"
			"arcs+states.  Option to preserves equivalence in tropical or log semiring, and\n"
			"if in tropical, stochasticit in either log or tropical.\n"
			"\n"
			"Usage:  fstrmepslocal  [in.fst [out.fst] ]\n";

		ParseOptions po(usage);
		bool use_log = false;
		bool stochastic_in_log = true;
		po.Register("use-log", &use_log,
			"Preserve equivalence in log semiring [false->tropical]\n");
		po.Register("stochastic-in-log", &stochastic_in_log,
			"Preserve stochasticity in log semiring [false->tropical]\n");
		po.Read(argc, argv);

		if (po.NumArgs() > 2) {
			//po.PrintUsage();
			//exit(1);
			KALDI_ERR << "wrong arguments.";
			return -1;
		}

		std::string fst_in_filename = po.GetOptArg(1),
			fst_out_filename = po.GetOptArg(2);

		VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

		if (!use_log && stochastic_in_log) {
			RemoveEpsLocalSpecial(fst);
		}
		else if (use_log && !stochastic_in_log) {
			KALDI_ERR << "fstrmsymbols: invalid combination of flags";
			return -1;
		}
		else if (use_log) {
			VectorFst<LogArc> log_fst;
			Cast(*fst, &log_fst);
			delete fst;
			RemoveEpsLocal(&log_fst);
			fst = new VectorFst<StdArc>;
			Cast(log_fst, fst);
		}
		else {
			RemoveEpsLocal(fst);
		}

		WriteFstKaldi(*fst, fst_out_filename);
		delete fst;
		return 0;
	}
	catch (const std::exception &e) {
		KALDI_ERR << e.what();
		return -1;
	}
}

