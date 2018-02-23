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
#include <fst/script/convert.h>

#include "fst_ext.h"

DEFINE_string(fst_type_convert, "vector", "Output FST type");

DECLARE_string(fst_type_convert);

//Converts an FST to another type.
int fstconvert(std::string in_name, std::string out_name, std::string fsttype)
{
  namespace s = fst::script;
  using fst::script::FstClass;

  std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
  if (!ifst) return -1;

  if (fsttype != "") FLAGS_fst_type_convert = fsttype;

  if (ifst->FstType() != FLAGS_fst_type_convert) {
    std::unique_ptr<FstClass> ofst(s::Convert(*ifst, FLAGS_fst_type_convert));
    if (!ofst) return -1;

	if (!ofst->Write(out_name))
	{
		LOGTW_ERROR << "Failed to convert " + in_name;
		return -1;
	}
  } else {

	if (!ifst->Write(out_name))
	{
		LOGTW_ERROR << "Failed to convert " + in_name;
		return -1;
	}
  }

  return 0;
}
