/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Gilles Boulianne., Apache 2.0.
*/
// bin/arpa2fst.cc
//
// Copyright 2009-2011  Gilles Boulianne.
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "lm/arpa-lm-compiler.h"
#include "util/kaldi-io.h"
#include "util/parse-options.h"

#include "kaldi-win/utility/Utility.h"

using namespace kaldi;

//Convert an ARPA format language model into an FST
//NOTE: When called without switches, the output G.fst will contain an embedded symbol table. 
//		This is compatible with the way a previous version of arpa2fst worked.
int arpa2fst(std::string arpa_rxfilename, std::string fst_wxfilename,
	// Option flags:
	std::string disambig_symbol, //""		Disambiguator. If provided (e.g. #0), used on input side of backoff links, and <s> and </s> are replaced with epsilons
	std::string read_syms_filename,	//""	e.g. "data/lang/words.txt"
	std::string write_syms_filename, //""	Write generated symbol table to a file
	std::string bos_symbol, // = "<s>"		Beginning of sentence symbol
	std::string eos_symbol, //= "</s>";		End of sentence symbol
	bool keep_symbols, // = false;			Store symbol table with FST. Symbols always saved to FST if symbol tables are neither read or written (otherwise symbols would be lost entirely)
	bool ilabel_sort //= true				Ilabel-sort the output FST
	)
{
	try {
		ParseOptions po("");
		ArpaParseOptions options;
		options.Register(&po);
		po.Register("bos-symbol", &bos_symbol, "Beginning of sentence symbol");
		po.Register("eos-symbol", &eos_symbol, "End of sentence symbol");
		po.Register("disambig-symbol", &disambig_symbol,
			"Disambiguator. If provided (e.g. #0), used on input side of "
			"backoff links, and <s> and </s> are replaced with epsilons");
		po.Register("read-symbol-table", &read_syms_filename, "Use existing symbol table");
		po.Register("write-symbol-table", &write_syms_filename, "Write generated symbol table to a file");
		po.Register("keep-symbols", &keep_symbols,
			"Store symbol table with FST. Symbols always saved to FST if "
			"symbol tables are neither read or written (otherwise symbols "
			"would be lost entirely)");
		po.Register("ilabel-sort", &ilabel_sort, "Ilabel-sort the output FST");
	
		int64 disambig_symbol_id = 0;

		fst::SymbolTable* symbols;
		if (!read_syms_filename.empty()) 
		{
			// Use existing symbols. Required symbols must be in the table.
			kaldi::Input kisym(read_syms_filename);
			symbols = fst::SymbolTable::ReadText(kisym.Stream(), PrintableWxfilename(read_syms_filename));
			if (symbols == NULL) {
				LOGTW_ERROR << " Could not read symbol table from file " << read_syms_filename;
				return -1;
			}

			options.oov_handling = ArpaParseOptions::kSkipNGram;
			if (!disambig_symbol.empty()) {
				disambig_symbol_id = symbols->Find(disambig_symbol);
				if (disambig_symbol_id == fst::SymbolTable::kNoSymbol) {
					LOGTW_ERROR << " Symbol table " << read_syms_filename << " has no symbol for " << disambig_symbol;
					return -1;
				}
			}
		}
		else {
			// Create a new symbol table and populate it from ARPA file.
			symbols = new fst::SymbolTable(PrintableWxfilename(fst_wxfilename));
			options.oov_handling = ArpaParseOptions::kAddToSymbols;
			symbols->AddSymbol("<eps>", 0);
			if (!disambig_symbol.empty()) {
				disambig_symbol_id = symbols->AddSymbol(disambig_symbol);
			}
		}

		// Add or use existing BOS and EOS.
		options.bos_symbol = symbols->AddSymbol(bos_symbol);
		options.eos_symbol = symbols->AddSymbol(eos_symbol);

		// If producing new (not reading existing) symbols and not saving them,
		// need to keep symbols with FST, otherwise they would be lost.
		if (read_syms_filename.empty() && write_syms_filename.empty())
			keep_symbols = true;

		// Actually compile LM.
		KALDI_ASSERT(symbols != NULL);
		ArpaLmCompiler lm_compiler(options, disambig_symbol_id, symbols);
		{
			Input ki(arpa_rxfilename);
			lm_compiler.Read(ki.Stream());
		}

		// Sort the FST in-place if requested by options.
		if (ilabel_sort) {
			fst::ArcSort(lm_compiler.MutableFst(), fst::StdILabelCompare());
		}

		// Write symbols if requested.
		if (!write_syms_filename.empty()) {
			kaldi::Output kosym(write_syms_filename, false);
			symbols->WriteText(kosym.Stream());
		}

		// Write LM FST.
		bool write_binary = true, write_header = false;
		kaldi::Output kofst(fst_wxfilename, write_binary, write_header);
		fst::FstWriteOptions wopts(PrintableWxfilename(fst_wxfilename));
		wopts.write_isymbols = wopts.write_osymbols = keep_symbols;
		lm_compiler.Fst().Write(kofst.Stream(), wopts);

		delete symbols;
	}
	catch (const std::exception &e) {
		LOGTW_ERROR << " " << e.what();
		return -1;
	}

	return 0;
}
