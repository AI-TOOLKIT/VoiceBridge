/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
	Based on : openfst, Apache 2.0
*/
// Performs operations (set, clear, relabel) on the symbols table attached to an
// input FST.

#include <cstring>

#include <memory>
#include <string>
#include <vector>

#include "base/kaldi-common.h"

#include <fst/flags.h>
#include <fst/util.h>
#include <fst/script/fst-class.h>
#include <fst/script/verify.h>

DEFINE_string(isymbols_S, "", "Input label symbol table");
DEFINE_string(osymbols_S, "", "Output label symbol table");
DEFINE_bool(clear_isymbols_S, false, "Clear input symbol table");
DEFINE_bool(clear_osymbols_S, false, "Clear output symbol table");
DEFINE_string(relabel_ipairs_S, "", "Input relabel pairs (numeric)");
DEFINE_string(relabel_opairs_S, "", "Output relabel pairs (numeric)");
DEFINE_string(save_isymbols_S, "", "Save fst file's input symbol table to file");
DEFINE_string(save_osymbols_S, "", "Save fst file's output symbol table to file");
DEFINE_bool(allow_negative_labels_S, false,
	"Allow negative labels (not recommended; may cause conflicts)");
DEFINE_bool(verify_S, false, "Verify fst properities before saving");

DECLARE_string(isymbols_S);
DECLARE_string(osymbols_S);
DECLARE_bool(clear_isymbols_S);
DECLARE_bool(clear_osymbols_S);
DECLARE_string(relabel_ipairs_S);
DECLARE_string(relabel_opairs_S);
DECLARE_string(save_isymbols_S);
DECLARE_string(save_osymbols_S);
DECLARE_bool(allow_negative_labels_S);
DECLARE_bool(verify_S);

int fstsymbols(int argc, char **argv) 
{
	try
	{
		namespace s = fst::script;
		using fst::script::MutableFstClass;
		using fst::ReadLabelPairs;
		using fst::SymbolTable;
		using fst::SymbolTableTextOptions;

		string usage =
			"Performs operations (set, clear, relabel) on the symbol"
			" tables attached to an FST.\n\n  Usage: ";
		usage += argv[0];
		usage += " [in.fst [out.fst]]\n";

		std::set_new_handler(FailedNewHandler);
		SET_FLAGS(usage.c_str(), &argc, &argv, true);
		if (argc > 3) {
			//ShowUsage();
			//return 1;
			KALDI_ERR << "Wrong arguments.";
			return -1;
		}

		const string in_name = argc > 1 && strcmp(argv[1], "-") != 0 ? argv[1] : "";
		const string out_name = argc > 2 ? argv[2] : "";

		std::unique_ptr<MutableFstClass> fst(MutableFstClass::Read(in_name, true));
		if (!fst) return 1;

		if (!FLAGS_save_isymbols_S.empty()) {
			const auto *isyms = fst->InputSymbols();
			if (isyms) {
				isyms->WriteText(FLAGS_save_isymbols_S);
			}
			else {
				KALDI_ERR << argv[0] << ": Saving isymbols but there are no input symbols.";
				return -1; //VB
			}
		}

		if (!FLAGS_save_osymbols_S.empty()) {
			const auto *osyms = fst->OutputSymbols();
			if (osyms) {
				osyms->WriteText(FLAGS_save_osymbols_S);
			}
			else {
				KALDI_ERR << argv[0] << ": Saving osymbols but there are no output symbols.";
				return -1; //VB
			}
		}

		const SymbolTableTextOptions opts(FLAGS_allow_negative_labels_S);

		std::unique_ptr<SymbolTable> isyms;
		if (!FLAGS_isymbols_S.empty()) {
			isyms.reset(SymbolTable::ReadText(FLAGS_isymbols_S, opts));
			fst->SetInputSymbols(isyms.get());
		}
		else if (FLAGS_clear_isymbols_S) {
			fst->SetInputSymbols(nullptr);
		}
		std::unique_ptr<SymbolTable> osyms;
		if (!FLAGS_osymbols_S.empty()) {
			osyms.reset(SymbolTable::ReadText(FLAGS_osymbols_S, opts));
			fst->SetOutputSymbols(osyms.get());
		}
		else if (FLAGS_clear_osymbols_S) {
			fst->SetOutputSymbols(nullptr);
		}

		using Label = int64;
		if (!FLAGS_relabel_ipairs_S.empty()) {
			std::vector<std::pair<Label, Label>> ipairs;
			ReadLabelPairs(FLAGS_relabel_ipairs_S, &ipairs, FLAGS_allow_negative_labels_S);
			std::unique_ptr<SymbolTable> isyms_relabel(
				RelabelSymbolTable(fst->InputSymbols(), ipairs));
			fst->SetInputSymbols(isyms_relabel.get());
		}
		if (!FLAGS_relabel_opairs_S.empty()) {
			std::vector<std::pair<Label, Label>> opairs;
			ReadLabelPairs(FLAGS_relabel_opairs_S, &opairs, FLAGS_allow_negative_labels_S);
			std::unique_ptr<SymbolTable> osyms_relabel(
				RelabelSymbolTable(fst->OutputSymbols(), opairs));
			fst->SetOutputSymbols(osyms_relabel.get());
		}

		if (FLAGS_verify_S && !s::Verify(*fst)) return 1;

		bool ret = fst->Write(out_name);
		if (!ret) return -1;

		return 0;
	}
	catch (const std::exception& e)
	{
		KALDI_ERR << e.what();
		return -1;
	}
}
