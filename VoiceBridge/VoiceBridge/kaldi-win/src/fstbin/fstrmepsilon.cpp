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
// Removes epsilons from an FST.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/getters.h>
#include <fst/script/rmepsilon.h>

#include <fst/fst.h>
#include <fst/weight.h>

#include "fst_ext.h"

DEFINE_bool(connect_rme, true, "Trim output");
DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_int64(nstate, fst::kNoStateId, "State number threshold");
DEFINE_string(queue_type, "auto",
	"Queue type: one of: \"auto\", "
	"\"fifo\", \"lifo\", \"shortest\", \"state\", \"top\"");
DEFINE_string(weight, "", "Weight threshold");
//NOTE: connect had to be renamed because it was already defined in fstcompose
DECLARE_bool(connect_rme);
DECLARE_double(delta);
DECLARE_int64(nstate);
DECLARE_string(queue_type);
DECLARE_string(weight);

//TODO:... add all above parameters as input

namespace s = fst::script;

// Removes epsilons from an FST.
int fstrmepsilon(std::string in_name, std::string out_name) 
{
	std::unique_ptr<MutableFstClass> fst(MutableFstClass::Read(in_name, true));
	if (!fst) return -1;

	const auto weight_threshold =
		FLAGS_weight.empty() ? WeightClass::Zero(fst->WeightType())
		: WeightClass(fst->WeightType(), FLAGS_weight);

	fst::QueueType queue_type;
	if (!s::GetQueueType(FLAGS_queue_type, &queue_type)) {
		LOGTW_ERROR << " Unknown or unsupported queue type: " << FLAGS_queue_type;
		return -1;
	}

	const s::RmEpsilonOptions opts(queue_type, FLAGS_connect_rme, weight_threshold,
		FLAGS_nstate, FLAGS_delta);

	s::RmEpsilon(fst.get(), opts);

	if (!fst->Write(out_name)) {
		LOGTW_ERROR << " Could not write output file: " << out_name << ".";
		return -1;
	}

	return 0;
}

//TODO:... not tested, experimental code
int fstrmepsilon(VectorFstClass * pinfst)
{
	if (!pinfst) return -1;

	const auto weight_threshold =
		FLAGS_weight.empty() ? WeightClass::Zero(pinfst->WeightType())
		: WeightClass(pinfst->WeightType(), FLAGS_weight);

	fst::QueueType queue_type;
	if (!s::GetQueueType(FLAGS_queue_type, &queue_type)) {
		LOGTW_ERROR << " Unknown or unsupported queue type: " << FLAGS_queue_type;
		return -1;
	}

	const s::RmEpsilonOptions opts(queue_type, FLAGS_connect_rme, weight_threshold, FLAGS_nstate, FLAGS_delta);

	s::RmEpsilon(pinfst, opts);

	return 0;
}
