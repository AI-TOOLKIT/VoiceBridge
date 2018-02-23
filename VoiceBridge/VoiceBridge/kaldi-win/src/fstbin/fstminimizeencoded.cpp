/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation, Apache 2
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

/* some test  examples:
 ( echo "0 0 0 0"; echo "0 0" ) | fstcompile | fstminimizeencoded | fstprint
 ( echo "0 1 0 0"; echo " 0 2 0 0"; echo "1 0"; echo "2 0"; ) | fstcompile | fstminimizeencoded | fstprint
*/

//Minimizes FST after encoding [similar to fstminimize, but no weight-pushing]
int fstminimizeencoded(std::string in_name, std::string out_name,  //in.fst, out.fst
	float delta) //Delta likelihood used for quantization of weights
{ 
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;
   
    VectorFst<StdArc> *fst = ReadFstKaldi(in_name);
    
    MinimizeEncoded(fst, delta);

    WriteFstKaldi(*fst, out_name);

    delete fst;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
  return 0;
}

