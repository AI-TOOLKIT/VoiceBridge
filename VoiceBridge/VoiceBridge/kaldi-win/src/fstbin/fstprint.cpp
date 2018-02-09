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
// Prints out binary FSTs in simple text format used by AT&T. The print happens into the given output file!

#include <cstring>

#include <fstream>
#include <memory>
#include <ostream>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/print.h>

#include "fst_ext.h"

DEFINE_bool(acceptor_p, false, "Input in acceptor format?");
DEFINE_string(isymbols_p, "", "Input label symbol table");
DEFINE_string(osymbols_p, "", "Output label symbol table");
DEFINE_string(ssymbols_p, "", "State label symbol table");
DEFINE_bool(numeric_p, false, "Print numeric labels?");
DEFINE_string(save_isymbols_p, "", "Save input symbol table to file");
DEFINE_string(save_osymbols_p, "", "Save output symbol table to file");
DEFINE_bool(show_weight_one_p, false,
	"Print/draw arc weights and final weights equal to semiring One?");
DEFINE_bool(allow_negative_labels_p, false,
	"Allow negative labels (not recommended; may cause conflicts)?");
DEFINE_string(missing_symbol_p, "",
	"Symbol to print when lookup fails (default raises error)");

DECLARE_bool(acceptor_p);
DECLARE_string(isymbols_p);
DECLARE_string(osymbols_p);
DECLARE_string(ssymbols_p);
DECLARE_bool(numeric_p);
DECLARE_string(save_isymbols_p);
DECLARE_string(save_osymbols_p);
DECLARE_bool(show_weight_one_p);
DECLARE_bool(allow_negative_labels_p);
DECLARE_string(missing_symbol_p);

//TODO:... add the above parameters as input

namespace s = fst::script;

// Prints out binary FSTs in simple text format used by AT&T. The print happens into the given output file!
// If the output can not be opened it write to std::cout
int fstprint(std::string in_name, std::string out_name) 
{	
	std::unique_ptr<FstClass> fst(FstClass::Read(in_name));
	if (!fst) return -1;

	string dest = "standard output";
	std::ofstream fstrm;
	if (out_name != "") {
		fstrm.open(out_name);
		if (!fstrm) {
			LOGTW_ERROR << " Open failed, file = " << out_name;
			return -1;
		}
		dest = out_name;
	}
	
	std::ostream &ostrm = fstrm.is_open() ? fstrm : std::cout;
	ostrm.precision(9);

	const SymbolTableTextOptions opts(FLAGS_allow_negative_labels_p);

	std::unique_ptr<const SymbolTable> isyms;
	if (!FLAGS_isymbols_p.empty() && !FLAGS_numeric_p) {
		isyms.reset(SymbolTable::ReadText(FLAGS_isymbols_p, opts));
		if (!isyms) return -1;
	}

	std::unique_ptr<const SymbolTable> osyms;
	if (!FLAGS_osymbols_p.empty() && !FLAGS_numeric_p) {
		osyms.reset(SymbolTable::ReadText(FLAGS_osymbols_p, opts));
		if (!osyms) return -1;
	}

	std::unique_ptr<const SymbolTable> ssyms;
	if (!FLAGS_ssymbols_p.empty() && !FLAGS_numeric_p) {
		ssyms.reset(SymbolTable::ReadText(FLAGS_ssymbols_p));
		if (!ssyms) return -1;
	}

	if (!isyms && !FLAGS_numeric_p && fst->InputSymbols()) {
		isyms.reset(fst->InputSymbols()->Copy());
	}

	if (!osyms && !FLAGS_numeric_p && fst->OutputSymbols()) {
		osyms.reset(fst->OutputSymbols()->Copy());
	}

	s::PrintFst(*fst, ostrm, dest, isyms.get(), osyms.get(), ssyms.get(),
		FLAGS_acceptor_p, FLAGS_show_weight_one_p, FLAGS_missing_symbol_p);

	if (isyms && !FLAGS_save_isymbols_p.empty()) {
		if (!isyms->WriteText(FLAGS_save_isymbols_p)) return -1;
	}

	if (osyms && !FLAGS_save_osymbols_p.empty()) {
		if (!osyms->WriteText(FLAGS_save_osymbols_p)) return -1;
	}

	return 0;
}

//TODO:... not tested, experimental code
int fstprint(VectorFstClass * pinfst, std::string out_name)
{
	if (!pinfst) return -1;

	string dest = "standard output";
	std::ofstream fstrm;
	if (out_name != "") {
		fstrm.open(out_name);
		if (!fstrm) {
			LOGTW_ERROR << " Open failed, file = " << out_name;
			return -1;
		}
		dest = out_name;
	}

	std::ostream &ostrm = fstrm.is_open() ? fstrm : std::cout;
	ostrm.precision(9);

	const SymbolTableTextOptions opts(FLAGS_allow_negative_labels_p);

	std::unique_ptr<const SymbolTable> isyms;
	if (!FLAGS_isymbols_p.empty() && !FLAGS_numeric_p) {
		isyms.reset(SymbolTable::ReadText(FLAGS_isymbols_p, opts));
		if (!isyms) return -1;
	}

	std::unique_ptr<const SymbolTable> osyms;
	if (!FLAGS_osymbols_p.empty() && !FLAGS_numeric_p) {
		osyms.reset(SymbolTable::ReadText(FLAGS_osymbols_p, opts));
		if (!osyms) return -1;
	}

	std::unique_ptr<const SymbolTable> ssyms;
	if (!FLAGS_ssymbols_p.empty() && !FLAGS_numeric_p) {
		ssyms.reset(SymbolTable::ReadText(FLAGS_ssymbols_p));
		if (!ssyms) return -1;
	}

	if (!isyms && !FLAGS_numeric_p && pinfst->InputSymbols()) {
		isyms.reset(pinfst->InputSymbols()->Copy());
	}

	if (!osyms && !FLAGS_numeric_p && pinfst->OutputSymbols()) {
		osyms.reset(pinfst->OutputSymbols()->Copy());
	}

	//TODO:... not sure if the below cast to FstClass is OK
	s::PrintFst((FstClass&)pinfst, ostrm, dest, isyms.get(), osyms.get(), ssyms.get(),
		FLAGS_acceptor_p, FLAGS_show_weight_one_p, FLAGS_missing_symbol_p);

	if (isyms && !FLAGS_save_isymbols_p.empty()) {
		if (!isyms->WriteText(FLAGS_save_isymbols_p)) return -1;
	}

	if (osyms && !FLAGS_save_osymbols_p.empty()) {
		if (!osyms->WriteText(FLAGS_save_osymbols_p)) return -1;
	}

	return 0;
}
