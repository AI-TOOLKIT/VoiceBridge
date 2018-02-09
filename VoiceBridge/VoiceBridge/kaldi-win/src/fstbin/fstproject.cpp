/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : openfst
*/
// Projects a transduction onto its input or output language.
#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/script/getters.h>
#include <fst/script/project.h>

#include "fst_ext.h"

DEFINE_bool(project_output, false, "Project on output (vs. input)");

DECLARE_bool(project_output);

namespace s = fst::script;

int fstproject(std::string in_name, std::string out_name, bool bProjectOutput) 
{
	std::unique_ptr<MutableFstClass> fst(MutableFstClass::Read(in_name, true));
	if (!fst) return -1;
	
	FLAGS_project_output = bProjectOutput;

	s::Project(fst.get(), s::GetProjectType(FLAGS_project_output));

	if (!fst->Write(out_name)) {
		LOGTW_ERROR << " Could not write output file: " << out_name << ".";
		return -1;
	}

	return 0;
}

//TODO:... not tested
int fstproject(VectorFstClass * pinfst, bool bProjectOutput)
{	
	if (!pinfst) return -1;

	FLAGS_project_output = bProjectOutput;

	s::Project(pinfst, s::GetProjectType(FLAGS_project_output));

	return 0;
}
