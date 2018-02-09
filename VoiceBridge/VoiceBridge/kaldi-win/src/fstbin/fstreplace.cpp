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
// Performs the dynamic replacement of arcs in one FST with another FST,
// allowing for the definition of FSTs analogous to RTNs.

#include <cstring>

#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/script/getters.h>
#include <fst/script/replace.h>

#include "fst_ext.h"

DEFINE_string(call_arc_labeling, "input", "Which labels to make non-epsilon on the call arc. One of: \"input\" (default), \"output\", \"both\", \"neither\"");
DEFINE_string(return_arc_labeling, "neither", "Which labels to make non-epsilon on the return arc. One of: \"input\", \"output\", \"both\", \"neither\" (default)");
DEFINE_int64(return_label, 0, "Label to put on return arc");
DEFINE_bool(epsilon_on_replace, false, "Call/return arcs are epsilon arcs?");

DECLARE_string(call_arc_labeling);
DECLARE_string(return_arc_labeling);
DECLARE_int64(return_label);
DECLARE_bool(epsilon_on_replace);

void Cleanup(std::vector<fst::script::LabelFstClassPair> *pairs) {
	for (const auto &pair : *pairs) {
		delete pair.second;
	}
	pairs->clear();
}

//Recursively replaces FST arcs with other FST(s).
int fstreplace(
	std::string in_name,			// input root fst file name
	std::string out_name,			// output fst file name
	std::string rootlabel,			// the root label
	FSTLABELMAP fstlabelmap)		// (fst filename, label) pair collection, where label is an integer in string format
{		
	namespace s = fst::script;
	using fst::script::FstClass;
	using fst::script::VectorFstClass;
	using fst::ReplaceLabelType;

	auto *ifst = FstClass::Read(in_name);
	if (!ifst) return -1;

	std::vector<s::LabelFstClassPair> pairs;
	// Note that if the root label is beyond the range of the underlying FST's labels, truncation will occur.
	const auto root = atoll(rootlabel.c_str());
	pairs.emplace_back(root, ifst);

	for (const auto& pair : fstlabelmap)
	{
		ifst = FstClass::Read(pair.first);
		if (!ifst) {
			Cleanup(&pairs);
			return -1;
		}
		// Note that if the root label is beyond the range of the underlying FST's labels, truncation will occur.
		const auto label = atoll(pair.second.c_str());
		pairs.emplace_back(label, ifst);
	}

	ReplaceLabelType call_label_type;
	if (!s::GetReplaceLabelType(FLAGS_call_arc_labeling, FLAGS_epsilon_on_replace, &call_label_type)) 
	{
		LOGTW_ERROR << " Unknown or unsupported call arc replace " << "label type: " << FLAGS_call_arc_labeling;
		return -1;
	}
	ReplaceLabelType return_label_type;
	if (!s::GetReplaceLabelType(FLAGS_return_arc_labeling, FLAGS_epsilon_on_replace, &return_label_type)) 
	{
		LOGTW_ERROR << " Unknown or unsupported return arc replace " << "label type: " << FLAGS_return_arc_labeling;
		return -1;
	}

	s::ReplaceOptions opts(root, call_label_type, return_label_type, FLAGS_return_label);

	VectorFstClass ofst(ifst->ArcType());
	s::Replace(pairs, &ofst, opts);
	Cleanup(&pairs);

	if (!ofst.Write(out_name)) {
		LOGTW_ERROR << " Could not write output file: " << out_name << ".";
		return -1;
	}

	return 0;
}
