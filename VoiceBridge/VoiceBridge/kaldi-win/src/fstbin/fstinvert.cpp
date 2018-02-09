/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : openfst
*/

// Inverts a transduction.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/script/invert.h>

#include <fst/script/fst-class.h>
#include <fst/script/invert.h>
#include <fst/script/script-impl.h>

#include "fst_ext.h"

int fstinvert(std::string in_name, std::string out_name) 
{
	namespace s = fst::script;
	using fst::script::MutableFstClass;

	//NOTE: if in_name=="" then it reads from std::cin, which should not happen here because there is no std::cin
	std::unique_ptr<MutableFstClass> fst(MutableFstClass::Read(in_name, true));
	if (!fst) return -1;

	s::Invert(fst.get());

	if (!fst->Write(out_name)) {
		LOGTW_ERROR << " Could not write output file: " << out_name << ".";
		return -1;
	}

	return 0;
}


namespace fst {
	namespace script {

		void Invert(MutableFstClass *fst) {
			Apply<Operation<MutableFstClass>>("Invert", fst->ArcType(), fst);
		}

		REGISTER_FST_OPERATION(Invert, StdArc, MutableFstClass);
		REGISTER_FST_OPERATION(Invert, LogArc, MutableFstClass);
		REGISTER_FST_OPERATION(Invert, Log64Arc, MutableFstClass);

	}  // namespace script
}  // namespace fst
