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
// Prints out various information about an FST such as number of states
// and arcs and property values (see properties.h).

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/script/info.h>

#include "fst_ext.h"

DEFINE_string(arc_filter, "any", "Arc filter: one of: connected states, and (strongly) connected components");
DEFINE_string(info_type, "auto", "Info format: one of: \"auto\", \"long\", \"short\"");
DEFINE_bool(test_properties, true, "Compute property values (if unknown to FST)");
DEFINE_bool(fst_verify, true, "Verify FST sanity");

DECLARE_string(arc_filter);
DECLARE_string(info_type);
DECLARE_bool(test_properties);
DECLARE_bool(fst_verify);

//Prints out information about an FST.
int fstinfo(std::string in_name,
			std::string sArcFilter, std::string sInfoType, 
			bool bTestProperties, bool bFstVerify, fst::FstInfo *info)
{
	namespace s = fst::script;
	using fst::script::FstClass;

	std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
	if (!ifst) return -1;

	FLAGS_test_properties = bTestProperties;
	if(sArcFilter != "") FLAGS_arc_filter = sArcFilter;
	if(sInfoType != "") FLAGS_info_type = sInfoType;
	FLAGS_fst_verify = bFstVerify;

	s::GetFstInfo(*ifst, FLAGS_test_properties, FLAGS_arc_filter, FLAGS_info_type, FLAGS_fst_verify, info);

	return 0;
}

