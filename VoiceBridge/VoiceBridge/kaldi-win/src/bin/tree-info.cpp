/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2013  Johns Hopkins University (author: Daniel Povey), Apache 2.0
See ../../COPYING for clarification regarding multiple authors
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
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"

#include "kaldi-win/src/kaldi_src.h"

int TreeInfo(int argc, char *argv[],
	int & numpdfs, int & context_width, int & central_position) //output
{
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Print information about decision tree (mainly the number of pdfs), to stdout\n"
        "Usage:  tree-info <tree-in>\n";
        
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      //po.PrintUsage();
      //exit(1);
	  KALDI_ERR << "Wrong arguments.";
	  return -1;
    }

    std::string tree_in_filename = po.GetArg(1);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_in_filename, &ctx_dep);

	KALDI_LOG << "num-pdfs " << ctx_dep.NumPdfs() << std::endl;
	KALDI_LOG << "context-width " << ctx_dep.ContextWidth() << std::endl;
	KALDI_LOG << "central-position " << ctx_dep.CentralPosition() << std::endl;

	numpdfs = ctx_dep.NumPdfs();
	context_width = ctx_dep.ContextWidth();
	central_position = ctx_dep.CentralPosition();

	return 0;    
  } catch(const std::exception &e) {
	KALDI_ERR << e.what();
	return -1;
  }
}
