/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2012  Microsoft Corporation
					 2012-2015  Johns Hopkins University (Author: Daniel Povey)
			See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"

#include "kaldi-win/src/kaldi_src.h"

int CompileTrainGraphs(int argc, char *argv[], fs::ofstream & file_log) {
	try {
		using namespace kaldi;
		typedef kaldi::int32 int32;
		using fst::SymbolTable;
		using fst::VectorFst;
		using fst::StdArc;

		const char *usage =
			"Creates training graphs (without transition-probabilities, by default)\n"
			"\n"
			"Usage:   compile-train-graphs [options] <tree-in> <model-in> "
			"<lexicon-fst-in> <transcriptions-rspecifier> <graphs-wspecifier>\n"
			"e.g.: \n"
			" compile-train-graphs tree 1.mdl lex.fst "
			"'ark:sym2int.pl -f 2- words.txt text|' ark:graphs.fsts\n";
		ParseOptions po(usage);

		TrainingGraphCompilerOptions gopts;
		int32 batch_size = 250;
		gopts.transition_scale = 0.0;  // Change the default to 0.0 since we will generally add the
		// transition probs in the alignment phase (since they change eacm time)
		gopts.self_loop_scale = 0.0;  // Ditto for self-loop probs.
		std::string disambig_rxfilename;
		gopts.Register(&po);

		po.Register("batch-size", &batch_size,
			"Number of FSTs to compile at a time (more -> faster but uses "
			"more memory.  E.g. 500");
		po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
			"list of disambiguation symbols in phone symbol table");

		po.Read(argc, argv);

		if (po.NumArgs() != 5) {
			//po.PrintUsage();
			//exit(1);
			KALDI_ERR << "Wrong arguments.";
			return -1;
		}

		std::string tree_rxfilename = po.GetArg(1);
		std::string model_rxfilename = po.GetArg(2);
		std::string lex_rxfilename = po.GetArg(3);
		std::string transcript_rspecifier = po.GetArg(4);
		std::string fsts_wspecifier = po.GetArg(5);

		ContextDependency ctx_dep;  // the tree.
		ReadKaldiObject(tree_rxfilename, &ctx_dep);

		TransitionModel trans_model;
		ReadKaldiObject(model_rxfilename, &trans_model);

		// need VectorFst because we will change it by adding subseq symbol.
		VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_rxfilename);

		std::vector<int32> disambig_syms;
		if (disambig_rxfilename != "")
			if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
				KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
				<< disambig_rxfilename;

		TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms, gopts);

		lex_fst = NULL;  // we gave ownership to gc.

		SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
		TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

		int num_succeed = 0, num_fail = 0;

		if (batch_size == 1) {  // We treat batch_size of 1 as a special case in order
		  // to test more parts of the code.
			for (; !transcript_reader.Done(); transcript_reader.Next()) {
				std::string key = transcript_reader.Key();
				const std::vector<int32> &transcript = transcript_reader.Value();
				VectorFst<StdArc> decode_fst;

				if (!gc.CompileGraphFromText(transcript, &decode_fst)) {
					decode_fst.DeleteStates();  // Just make it empty.
				}
				if (decode_fst.Start() != fst::kNoStateId) {
					num_succeed++;
					fst_writer.Write(key, decode_fst);
				}
				else {
					KALDI_WARN << "Empty decoding graph for utterance "
						<< key;
					num_fail++;
				}
			}
		}
		else {
			std::vector<std::string> keys;
			std::vector<std::vector<int32> > transcripts;
			while (!transcript_reader.Done()) {
				keys.clear();
				transcripts.clear();
				for (; !transcript_reader.Done() &&
					static_cast<int32>(transcripts.size()) < batch_size;
					transcript_reader.Next()) {
					keys.push_back(transcript_reader.Key());
					transcripts.push_back(transcript_reader.Value());
				}
				std::vector<fst::VectorFst<fst::StdArc>* > fsts;
				if (!gc.CompileGraphsFromText(transcripts, &fsts)) {
					KALDI_ERR << "Not expecting CompileGraphs to fail.";
				}
				KALDI_ASSERT(fsts.size() == keys.size());
				for (size_t i = 0; i < fsts.size(); i++) {
					if (fsts[i]->Start() != fst::kNoStateId) {
						num_succeed++;
						fst_writer.Write(keys[i], *(fsts[i]));
					}
					else {
						KALDI_WARN << "Empty decoding graph for utterance "
							<< keys[i];
						num_fail++;
					}
				}
				DeletePointers(&fsts);
			}
		}
		if (file_log)
			file_log << "compile-train-graphs: succeeded for " << num_succeed << " graphs, failed for " << num_fail << "\n";
		else
			KALDI_LOG << "compile-train-graphs: succeeded for " << num_succeed << " graphs, failed for " << num_fail;
		return (num_succeed != 0 ? 0 : 1);
	}
	catch (const std::exception &e) {
		KALDI_ERR << e.what();
		return -1;
	}
}
