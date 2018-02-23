/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : openfst
*/
/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : openfst
*/

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/compose.h>
#include <fst/script/getters.h>

#include "fst_ext.h"

DEFINE_string(compose_filter, "auto",
	"Composition filter, one of: \"alt_sequence\", \"auto\", "
	"\"match\", \"null\", \"sequence\", \"trivial\"");
DEFINE_bool(connect, true, "Trim output");

DECLARE_string(compose_filter);
DECLARE_bool(connect);

namespace s = fst::script;
using fst::ComposeFilter;
using fst::ComposeOptions;
using fst::script::FstClass;
using fst::script::VectorFstClass;

//Composes two FSTs.
//saves the composed FST's into the file out_name if it is not ""
int fstcompose(std::string in1_name, std::string in2_name, std::string out_name, VectorFstClass * pofst,
	std::string composeFilter, bool bConnect)
{
	std::unique_ptr<FstClass> ifst1(FstClass::Read(in1_name));
	if (!ifst1) return -1;

	std::unique_ptr<FstClass> ifst2(FstClass::Read(in2_name));
	if (!ifst2) return -1;

	if (ifst1->ArcType() != ifst2->ArcType()) {
		LOGTW_ERROR << " Input FSTs must have the same arc type.";
		return -1;
	}

	if (composeFilter != "") FLAGS_compose_filter = composeFilter;
	FLAGS_connect = bConnect;

	VectorFstClass ofst(ifst1->ArcType());

	ComposeFilter compose_filter;
	if (!s::GetComposeFilter(FLAGS_compose_filter, &compose_filter)) {
		LOGTW_ERROR << " Unknown or unsupported compose filter type: " << FLAGS_compose_filter;
		return -1;
	}

	const ComposeOptions opts(FLAGS_connect, compose_filter);

	s::Compose(*ifst1, *ifst2, &ofst, opts);

	pofst = &ofst;

	if (out_name!="" && !ofst.Write(out_name)) {
		LOGTW_ERROR << " Could not write output file: " << out_name << ".";
		return -1;
	}

	return 0;
}
