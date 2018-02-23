/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation, Apache 2
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/determinize-star.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

#include "fst_ext.h"

//Adds self-loops to states of an FST to propagate disambiguation symbols through it. They are added on each final state 
//and each state with non-epsilon output symbols on at least one arc out of the state. Useful in conjunction with predeterminize.
int fstaddselfloops(std::string fst_in_filename,		 // the compiled fst file to apply the loops to
					std::string fst_out_filename,		 // the output file of this function containing the self-loops
					std::string disambig_in_rxfilename,  // ...phones.int
					std::string disambig_out_rxfilename) // ...words.int
{
	try {
		using namespace kaldi;
		using namespace fst;
		using kaldi::int32;

		VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

		std::vector<int32> disambig_in;
		if (!ReadIntegerVectorSimple(disambig_in_rxfilename, &disambig_in)) {
			LOGTW_ERROR << " Could not read disambiguation symbols from " << kaldi::PrintableRxfilename(disambig_in_rxfilename);
			return -1;
		}

		std::vector<int32> disambig_out;
		if (!ReadIntegerVectorSimple(disambig_out_rxfilename, &disambig_out)) {
			LOGTW_ERROR << " Could not read disambiguation symbols from " << kaldi::PrintableRxfilename(disambig_out_rxfilename);
			return -1;
		}

		if (disambig_in.size() != disambig_out.size()) {
			LOGTW_ERROR << " mismatch in size of disambiguation symbols. See " << kaldi::PrintableRxfilename(disambig_in_rxfilename)
				<< " and " << kaldi::PrintableRxfilename(disambig_out_rxfilename);
			return -1;
		}

		AddSelfLoops(fst, disambig_in, disambig_out);

		WriteFstKaldi(*fst, fst_out_filename);

		delete fst;

		return 0;
	}
	catch (const std::exception &e) {
		LOGTW_FATALERROR << e.what();
		return -1;
	}
	return 0;
}

