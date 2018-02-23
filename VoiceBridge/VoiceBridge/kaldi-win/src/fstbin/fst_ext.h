/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : openfst
*/
#pragma once

#include <kaldi-win/stdafx.h>
#include "kaldi-win/utility/Utility.h"
#include <fst/script/info.h>
#include <fst/weight.h>

#include "kaldi-win/utility/Utility.h"

using fst::script::FstClass;
using fst::script::VectorFstClass;
using fst::script::MutableFstClass;
using fst::script::WeightClass;
using fst::SymbolTable;
using fst::SymbolTableTextOptions;
using FSTLABELMAP = std::unordered_map<std::string, std::string>;

int fstarcsort(std::string sortType, std::string in_name, std::string out_name);

int fstcompile(bool bAcceptor, std::string sArc_type, std::string sFst_type,
	std::string sPathSource, std::string sPathDestination,
	std::string sIsymbols, std::string sOsymbols, std::string sSsymbols,
	bool bKeep_isymbols, bool bKeep_osymbols, bool bKeep_state_numbering, bool bAllow_negative_labels);

int fstaddselfloops(std::string fst_in_filename,
	std::string fst_out_filename,
	std::string disambig_in_rxfilename,
	std::string disambig_out_rxfilename);

int fstinvert(std::string in_name, std::string out_name);

int fstinfo(std::string in_name, std::string sArcFilter, std::string sInfoType,	bool bTestProperties, bool bFstVerify, fst::FstInfo *info);

int fstreplace(std::string in_name,	std::string out_name, std::string rootlabel, FSTLABELMAP fstlabelmap);

int fstcompose(std::string in1_name, std::string in2_name, std::string out_name, VectorFstClass * pofst,
	std::string composeFilter = "auto", bool bConnect = true);

int fstproject(std::string in_name, std::string out_name, bool bProjectOutput = false);
int fstproject(VectorFstClass * pinfst, bool bProjectOutput);

int fstrandgen(std::string in_name, std::string out_name);
int fstrandgen(VectorFstClass * pinfst, VectorFstClass * poutfst);

int fstrmepsilon(std::string in_name, std::string out_name);
int fstrmepsilon(VectorFstClass * pinfst);

int fsttopsort(std::string in_name, std::string out_name);
int fsttopsort(VectorFstClass * pinfst);

int fstprint(std::string in_name, std::string out_name);
int fstprint(VectorFstClass * pinfst, std::string out_name);

int fstdeterminizestar(std::string fst_in_str, std::string fst_out_str,
	bool use_log = false,		//Determinize in log semiring.
	float delta = fst::kDelta,  //Delta value used to determine equivalence of weights.
	int max_states = -1);

int fsttablecompose(std::string fst1_in_str, std::string fst2_in_str, std::string fst_out_str,
	std::string match_side = "left",			//Side of composition to do table match, one of: \"left\" or \"right\"
	std::string composeFilter = "sequence",		//Composition filter to use, one of: \"alt_sequence\", \"auto\", \"match\", \"sequence\"
	bool bConnect = true);

int fstisstochastic(std::string fst_in_filename,
	//options:
	float delta = 0.01, //Maximum error to accept.
	bool test_in_log = true); //Test stochasticity in log semiring.

int fstminimizeencoded(std::string in_name, std::string out_name, float delta = fst::kDelta);
int fstpushspecial(std::string in_name, std::string out_name, float delta = fst::kDelta);

int fstcomposecontext(int argc, char *argv[]);

int fstrmsymbols(int argc, char *argv[]);
int fstsymbols(int argc, char **argv);

int fstrmepslocal(int argc, char *argv[]);

int fstconvert(std::string in_name, std::string out_name, std::string fsttype="");



