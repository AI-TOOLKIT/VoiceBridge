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
// Generates random paths through an FST.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/getters.h>
#include <fst/script/randgen.h>

#ifndef _MSC_VER
#include <unistd.h>
#else
#include <process.h>
#define getpid _getpid
#endif

#include <climits>
#include <ctime>

#include <fst/flags.h>

#include "fst_ext.h"

DEFINE_int32(max_length, INT32_MAX, "Maximum path length");
DEFINE_int32(npath, 1, "Number of paths to generate");
DEFINE_int32(seed, time(nullptr) + getpid(), "Random seed");
DEFINE_string(select, "uniform",
	"Selection type: one of: "
	" \"uniform\", \"log_prob\" (when appropriate),"
	" \"fast_log_prob\" (when appropriate)");
DEFINE_bool(weighted, false,
	"Output tree weighted by path count vs. unweighted paths");
DEFINE_bool(remove_total_weight, false,
	"Remove total weight when output weighted");

DECLARE_int32(max_length);
DECLARE_int32(npath);
DECLARE_int32(seed);
DECLARE_string(select);
DECLARE_bool(weighted);
DECLARE_bool(remove_total_weight);

//TODO:... add all above parameters as input

namespace s = fst::script;

// Generates random paths through an FST.
int fstrandgen(std::string in_name, std::string out_name) 
{
	std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
	if (!ifst) return -1;

	VectorFstClass ofst(ifst->ArcType());

	s::RandArcSelection ras;
	if (!s::GetRandArcSelection(FLAGS_select, &ras)) {
		LOGTW_ERROR << " Unknown or unsupported select type " << FLAGS_select;
		return -1;
	}

	s::RandGen(*ifst, &ofst, FLAGS_seed,
		fst::RandGenOptions<s::RandArcSelection>(
			ras, FLAGS_max_length, FLAGS_npath, FLAGS_weighted,
			FLAGS_remove_total_weight));

	if (!ofst.Write(out_name)) {
		LOGTW_ERROR << " Could not write output file: " << out_name << ".";
		return -1;
	}

	return 0;
}

//TODO:... not tested, experimental code
int fstrandgen(VectorFstClass * pinfst, VectorFstClass * poutfst)
{
	s::RandArcSelection ras;
	if (!s::GetRandArcSelection(FLAGS_select, &ras)) {
		LOGTW_ERROR << " Unknown or unsupported select type " << FLAGS_select;
		return -1;
	}

	VectorFstClass ofst(pinfst->ArcType());

	//TODO:... not sure if the below cast to FstClass is OK
	s::RandGen((FstClass&)pinfst, &ofst, FLAGS_seed,
		fst::RandGenOptions<s::RandArcSelection>(ras, FLAGS_max_length, FLAGS_npath, FLAGS_weighted, FLAGS_remove_total_weight));

	*poutfst = ofst;

	return 0;
}
