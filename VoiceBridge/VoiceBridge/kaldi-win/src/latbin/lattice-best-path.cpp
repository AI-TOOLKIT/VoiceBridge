/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation, Apache 2.0.
		   See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

#include "kaldi-win/src/kaldi_src.h"

//NOTE: if TLOG exists the log is going to the twin log (local file log and std::cout); thread safe.
int LatticeBestPath(int argc, char *argv[], fs::ofstream & file_log) {
	try {
		using namespace kaldi;
		typedef kaldi::int32 int32;
		typedef kaldi::int64 int64;
		using fst::SymbolTable;
		using fst::VectorFst;
		using fst::StdArc;

		const char *usage =
			"Generate 1-best path through lattices; output as transcriptions and alignments\n"
			"Note: if you want output as FSTs, use lattice-1best; if you want output\n"
			"with acoustic and LM scores, use lattice-1best | nbest-to-linear\n"
			"Usage: lattice-best-path [options]  <lattice-rspecifier> [ <transcriptions-wspecifier> [ <alignments-wspecifier>] ]\n"
			" e.g.: lattice-best-path --acoustic-scale=0.1 ark:1.lats 'ark,t:|int2sym.pl -f 2- words.txt > text' ark:1.ali\n";

		ParseOptions po(usage);
		BaseFloat acoustic_scale = 1.0;
		BaseFloat lm_scale = 1.0;

		std::string word_syms_filename;
		po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
		po.Register("lm-scale", &lm_scale, "Scaling factor for LM probabilities. "
			"Note: the ratio acoustic-scale/lm-scale is all that matters.");
		po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");

		po.Read(argc, argv);

		if (po.NumArgs() < 1 || po.NumArgs() > 3) {
			//po.PrintUsage();
			//exit(1);
			KALDI_ERR << "Wrong arguments.";
			return -1;
		}

		std::string lats_rspecifier = po.GetArg(1),
			transcriptions_wspecifier = po.GetOptArg(2),
			alignments_wspecifier = po.GetOptArg(3);

		SequentialCompactLatticeReader clat_reader(lats_rspecifier);

		Int32VectorWriter transcriptions_writer(transcriptions_wspecifier);

		Int32VectorWriter alignments_writer(alignments_wspecifier);

		fst::SymbolTable *word_syms = NULL;
		if (word_syms_filename != "")
			if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
				KALDI_ERR << "Could not read symbol table from file "
				<< word_syms_filename;


		int32 n_done = 0, n_fail = 0;
		int64 n_frame = 0;
		LatticeWeight tot_weight = LatticeWeight::One();

		for (; !clat_reader.Done(); clat_reader.Next()) {
			std::string key = clat_reader.Key();
			CompactLattice clat = clat_reader.Value();
			clat_reader.FreeCurrent();
			fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
			CompactLattice clat_best_path;
			CompactLatticeShortestPath(clat, &clat_best_path);  // A specialized
			// implementation of shortest-path for CompactLattice.
			Lattice best_path;
			ConvertLattice(clat_best_path, &best_path);
			if (best_path.Start() == fst::kNoStateId) {
				KALDI_WARN << "Best-path failed for key " << key;
				n_fail++;
			}
			else {
				std::vector<int32> alignment;
				std::vector<int32> words;
				LatticeWeight weight;
				GetLinearSymbolSequence(best_path, &alignment, &words, &weight);
				if (file_log) {
					file_log << "For utterance " << key << ", best cost "
						<< weight.Value1() << " + " << weight.Value2() << " = "
						<< (weight.Value1() + weight.Value2())
						<< " over " << alignment.size() << " frames.";
				}
				else {
					KALDI_LOG << "For utterance " << key << ", best cost "
						<< weight.Value1() << " + " << weight.Value2() << " = "
						<< (weight.Value1() + weight.Value2())
						<< " over " << alignment.size() << " frames.";
				}
				if (transcriptions_wspecifier != "")
					transcriptions_writer.Write(key, words);
				if (alignments_wspecifier != "")
					alignments_writer.Write(key, alignment);
				if (word_syms != NULL) {
					std::stringstream ss;
					ss << key << ' ';
					for (size_t i = 0; i < words.size(); i++) {
						std::string s = word_syms->Find(words[i]);
						if (s == "")
							KALDI_WARN << "Word-id " << words[i] << " not in symbol table.";
						ss << s << ' ';
					}
					LOGTW_INFO << ss.str();
				}
				n_done++;
				n_frame += alignment.size();
				tot_weight = Times(tot_weight, weight);
			}
		}

		BaseFloat tot_weight_float = tot_weight.Value1() + tot_weight.Value2();

		if (file_log) {
			file_log << "Overall cost per frame is " << (tot_weight_float / n_frame)
				<< " = " << (tot_weight.Value1() / n_frame) << " [graph]"
				<< " + " << (tot_weight.Value2() / n_frame) << " [acoustic]"
				<< " over " << n_frame << " frames.";
			file_log << "Done " << n_done << " lattices, failed for " << n_fail;
		}
		else {
			KALDI_LOG << "Overall cost per frame is " << (tot_weight_float / n_frame)
				<< " = " << (tot_weight.Value1() / n_frame) << " [graph]"
				<< " + " << (tot_weight.Value2() / n_frame) << " [acoustic]"
				<< " over " << n_frame << " frames.";
			KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
		}
		delete word_syms;
		if (n_done != 0) return 0;
		else return 1;
	}
	catch (const std::exception &e) {
		KALDI_ERR << e.what();
		return -1;
	}
}
