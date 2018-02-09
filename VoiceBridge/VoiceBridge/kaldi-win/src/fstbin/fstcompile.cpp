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
// Creates binary FSTs from simple text format used by AT&T.

#include <cstring>

#include <fstream>
#include <istream>
#include <memory>
#include <string>

#include <fst/log.h>
#include <fst/script/compile.h>
#include <fst/compat.h>
#include <fst/flags.h>

#include "fst_ext.h"

DEFINE_bool(acceptor, false, "Input in acceptor format");
DEFINE_string(arc_type, "standard", "Output arc type");
DEFINE_string(fst_type, "vector", "Output FST type");
DEFINE_string(isymbols, "", "Input label symbol table");
DEFINE_string(osymbols, "", "Output label symbol table");
DEFINE_string(ssymbols, "", "State label symbol table");
DEFINE_bool(keep_isymbols, false, "Store input label symbol table with FST");
DEFINE_bool(keep_osymbols, false, "Store output label symbol table with FST");
DEFINE_bool(keep_state_numbering, false, "Do not renumber input states");
DEFINE_bool(allow_negative_labels, false, "Allow negative labels (not recommended; may cause conflicts)");

DECLARE_bool(acceptor);
DECLARE_string(arc_type);
DECLARE_string(fst_type);
DECLARE_string(isymbols);
DECLARE_string(osymbols);
DECLARE_string(ssymbols);
DECLARE_bool(keep_isymbols);
DECLARE_bool(keep_osymbols);
DECLARE_bool(keep_state_numbering);
DECLARE_bool(allow_negative_labels);

int fstcompile(bool bAcceptor, std::string sArc_type, std::string sFst_type, 
	std::string sPathSource, std::string sPathDestination,
	std::string sIsymbols, std::string sOsymbols, std::string sSsymbols,
	bool bKeep_isymbols, bool bKeep_osymbols, bool bKeep_state_numbering, bool bAllow_negative_labels)
{
	namespace s = fst::script;
	using fst::SymbolTable;
	using fst::SymbolTableTextOptions;
	std::ifstream fstrm;

	FLAGS_acceptor = bAcceptor;
	if (sArc_type != "") FLAGS_arc_type = sArc_type;
	if (sFst_type != "") FLAGS_fst_type = sFst_type;
	FLAGS_keep_isymbols = bKeep_isymbols;
	FLAGS_keep_osymbols = bKeep_osymbols;
	FLAGS_keep_state_numbering = bKeep_state_numbering;
	FLAGS_allow_negative_labels = bAllow_negative_labels;
	FLAGS_isymbols = sIsymbols;
	FLAGS_osymbols = sOsymbols;
	FLAGS_ssymbols = sSsymbols;

	fstrm.open(sPathSource);
	if (!fstrm) {
		LOGTW_ERROR << " can't open input file: " << sPathSource << ".";
		return -1;
	}
	std::istream &istrm = fstrm;

	const SymbolTableTextOptions opts(FLAGS_allow_negative_labels);

	std::unique_ptr<const SymbolTable> isyms;
	if (!FLAGS_isymbols.empty()) {
		isyms.reset(SymbolTable::ReadText(FLAGS_isymbols, opts));
		if (!isyms) return -1;
	}

	std::unique_ptr<const SymbolTable> osyms;
	if (!FLAGS_osymbols.empty()) {
		osyms.reset(SymbolTable::ReadText(FLAGS_osymbols, opts));
		if (!osyms) return -1;
	}

	std::unique_ptr<const SymbolTable> ssyms;
	if (!FLAGS_ssymbols.empty()) {
		ssyms.reset(SymbolTable::ReadText(FLAGS_ssymbols));
		if (!ssyms) return -1;
	}

	s::CompileFst(istrm, sPathSource, sPathDestination, FLAGS_fst_type, FLAGS_arc_type,
		isyms.get(), osyms.get(), ssyms.get(), FLAGS_acceptor,
		FLAGS_keep_isymbols, FLAGS_keep_osymbols,
		FLAGS_keep_state_numbering, FLAGS_allow_negative_labels);

	return 0;
}
