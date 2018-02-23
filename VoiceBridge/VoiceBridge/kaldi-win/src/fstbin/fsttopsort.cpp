/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : openfst
*/
// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Topologically sorts an FST.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/topsort.h>

#include "fst_ext.h"

namespace s = fst::script;

int fsttopsort(std::string in_name, std::string out_name) 
{
	std::unique_ptr<MutableFstClass> fst(MutableFstClass::Read(in_name, true));
	if (!fst) return -1;

	bool acyclic = TopSort(fst.get());

	if (!acyclic) LOGTW_WARNING << " " << in_name << ": Input FST is cyclic";

	if (!fst->Write(out_name)) {
		LOGTW_ERROR << " Could not write output file: " << out_name << ".";
		return -1;
	}

	return 0;
}

//TODO:... not tested, experimental code
int fsttopsort(VectorFstClass * pinfst) 
{
	if (!pinfst) return -1;

	bool acyclic = TopSort(pinfst);

	if (!acyclic) LOGTW_WARNING << " Input FST is cyclic.";

	return 0;
}
