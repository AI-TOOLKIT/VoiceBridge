/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012  Daniel Povey, Apache 2.0
		   See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "util/parse-options.h"
#include "util/text-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/push-special.h"
#include "fstext/kaldi-fst-io.h"

/*
Pushes weights in an FST such that all the states in the FST have arcs and final-probs with weights that
"sum to the same amount (viewed as being in the log semiring). Thus, the \"extra weight\" is distributed 
throughout the FST. Tolerance parameter --delta controls how exact this is, and the speed.
*/
int fstpushspecial(std::string in_name, std::string out_name,  //in.fst, out.fst
	float delta) //Delta cost: after pushing, all states will have a total weight that differs from the average by no more than this. 
{
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    VectorFst<StdArc> *fst = ReadFstKaldi(in_name);

    PushSpecial(fst, delta);

    WriteFstKaldi(*fst, out_name);
    delete fst;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
