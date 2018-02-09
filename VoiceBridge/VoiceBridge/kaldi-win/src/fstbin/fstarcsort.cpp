/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : openfst
*/
// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Sorts arcs of an FST.

#include <cstring>

#include <memory>
#include <string>

#include <fst/script/arcsort.h>
#include <fst/script/getters.h>
#include <fst/log.h>
#include <fst/compat.h>
#include <fst/flags.h>

#include "fst_ext.h"

DEFINE_string(sort_type, "ilabel", "Comparison method, one of: \"ilabel\", \"olabel\"");

DECLARE_string(sort_type);

int fstarcsort(std::string sortType, std::string in_name, std::string out_name)
{
	namespace s = fst::script;
	using fst::script::MutableFstClass;

	if(sortType != "") FLAGS_sort_type = sortType;

	//NOTE: if in_name=="" then it reads from std::cin, which should not happen here because there is no std::cin
	std::unique_ptr<MutableFstClass> fst(MutableFstClass::Read(in_name, true));
	if (!fst) return -1;

	s::ArcSortType sort_type;
	if (!s::GetArcSortType(FLAGS_sort_type, &sort_type)) {
		LOGTW_ERROR << " Unknown or unsupported sort type: " << FLAGS_sort_type << ".";
		return -1;
	}

	s::ArcSort(fst.get(), sort_type);

	if (!fst->Write(out_name)) {
		LOGTW_ERROR << " Could not write output file: " << out_name << ".";
		return -1;
	}

	return 0;
}

