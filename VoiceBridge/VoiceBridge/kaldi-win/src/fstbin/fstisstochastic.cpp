/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation
*/
// fstbin/fstisstochastic.cc
// Copyright 2009-2011  Microsoft Corporation
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "util/parse-options.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

#include "fst_ext.h"

// e.g. of test:
// echo " 0 0" | fstcompile | fstisstochastic
// should return 0 and print "0 0" [meaning, min and
// max weight are one = exp(0)]
// echo " 0 1" | fstcompile | fstisstochastic
// should  return 1, not stochastic, and print 1 1
// (echo "0 0 0 0 0.693147 "; echo "0 1 0 0 0.693147 "; echo "1 0" ) | fstcompile | fstisstochastic
// should return 0, stochastic; it prints "0 -1.78e-07" for me
// (echo "0 0 0 0 0.693147 "; echo "0 1 0 0 0.693147 "; echo "1 0" ) | fstcompile | fstisstochastic --test-in-log=false
// should return 1, not stochastic in tropical; it prints "0 0.693147" for me
// (echo "0 0 0 0 0 "; echo "0 1 0 0 0 "; echo "1 0" ) | fstcompile | fstisstochastic --test-in-log=false
// should return 0, stochastic in tropical; it prints "0 0" for me
// (echo "0 0 0 0 0.693147 "; echo "0 1 0 0 0.693147 "; echo "1 0" ) | fstcompile | fstisstochastic --test-in-log=false --delta=1
// returns 0 even though not stochastic because we gave it an absurdly large delta.

using namespace kaldi;
using namespace fst;

//Checks whether an FST is stochastic and exits with success if so. Prints out maximum error (in log units).
int fstisstochastic(std::string fst_in_filename,
	//options:
	float delta, //= 0.01 //Maximum error to accept.
	bool test_in_log // = true //Test stochasticity in log semiring.
	)
{
	try {
		using kaldi::int32;
		Fst<StdArc> *fst = ReadFstKaldiGeneric(fst_in_filename);

		bool ans;
		StdArc::Weight min, max;
		if (test_in_log)  ans = IsStochasticFstInLog(*fst, delta, &min, &max);
		else ans = IsStochasticFst(*fst, delta, &min, &max);

		LOGTW_INFO << "min weigth=" << min.Value() << " max weigth=" << max.Value() << '.';
		delete fst;
		if (ans) return 0;  // success;
		else return -1;
	}
	catch (const std::exception &e) {
		LOGTW_ERROR << " " << e.what();
		return -1;
	}

	return 0;
}
