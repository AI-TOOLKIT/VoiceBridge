/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on :
*/
// Copyright 2013  Daniel Povey (Johns Hopkins University)
// See ../../COPYING for clarification regarding multiple authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//  http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/lattice-functions.h"
#include "lat/push-lattice.h"
#include "lat/minimize-lattice.h"

#include "kaldi-win/src/kaldi_src.h"

/*
	LatticeDeterminizePruned : Determinize lattices, keeping only the best path (sequence of acoustic states) for each input-symbol sequence.
*/
int LatticeDeterminizePruned(int argc, char *argv[], fs::ofstream & file_log) {
	try {
		using namespace kaldi;
		typedef kaldi::int32 int32;

		const char *usage =
			"Determinize lattices, keeping only the best path (sequence of acoustic states)\n"
			"for each input-symbol sequence.  This version does pruning as part of the\n"
			"determinization algorithm, which is more efficient and prevents blowup.\n"
			"See http://kaldi-asr.org/doc/lattices.html for more information on lattices.\n"
			"\n"
			"Usage: lattice-determinize-pruned [options] lattice-rspecifier lattice-wspecifier\n"
			" e.g.: lattice-determinize-pruned --acoustic-scale=0.1 --beam=6.0 ark:in.lats ark:det.lats\n";

		ParseOptions po(usage);
		BaseFloat acoustic_scale = 1.0;
		BaseFloat beam = 10.0;
		bool minimize = false;
		fst::DeterminizeLatticePrunedOptions opts; // Options used in DeterminizeLatticePruned--
		// this options class does not have its own Register function as it's viewed as
		// being more part of "fst world", so we register its elements independently.
		opts.max_mem = 50000000;
		opts.max_loop = 0; // was 500000;

		po.Register("acoustic-scale", &acoustic_scale,
			"Scaling factor for acoustic likelihoods");
		po.Register("beam", &beam, "Pruning beam [applied after acoustic scaling].");
		po.Register("minimize", &minimize,
			"If true, push and minimize after determinization");
		opts.Register(&po);
		po.Read(argc, argv);

		if (po.NumArgs() != 2) {
			//po.PrintUsage();
			//exit(1);
			KALDI_ERR << "Wrong arguments.";
			return -1;
		}

		std::string lats_rspecifier = po.GetArg(1),
			lats_wspecifier = po.GetArg(2);


		// Read as regular lattice-- this is the form the determinization code
		// accepts.
		SequentialLatticeReader lat_reader(lats_rspecifier);

		// Write as compact lattice.
		CompactLatticeWriter compact_lat_writer(lats_wspecifier);

		int32 n_done = 0, n_warn = 0;

		// depth stats (for diagnostics).
		double sum_depth_in = 0.0,
			sum_depth_out = 0.0, sum_t = 0.0;

		if (acoustic_scale == 0.0)
			KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";

		for (; !lat_reader.Done(); lat_reader.Next()) {
			std::string key = lat_reader.Key();
			Lattice lat = lat_reader.Value();

			KALDI_VLOG(2) << "Processing lattice " << key;

			Invert(&lat); // so word labels are on the input side.
			lat_reader.FreeCurrent();
			fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);
			if (!TopSort(&lat)) {
				KALDI_WARN << "Could not topologically sort lattice: this probably means it"
					" has bad properties e.g. epsilon cycles.  Your LM or lexicon might "
					"be broken, e.g. LM with epsilon cycles or lexicon with empty words.";
			}
			fst::ArcSort(&lat, fst::ILabelCompare<LatticeArc>());
			CompactLattice det_clat;
			if (!DeterminizeLatticePruned(lat, beam, &det_clat, opts)) {
				KALDI_WARN << "For key " << key << ", determinization did not succeed"
					"(partial output will be pruned tighter than the specified beam.)";
				n_warn++;
			}
			fst::Connect(&det_clat);
			if (det_clat.NumStates() == 0) {
				KALDI_WARN << "For key " << key << ", determinized and trimmed lattice "
					"was empty.";
				n_warn++;
			}
			if (minimize) {
				PushCompactLatticeStrings(&det_clat);
				PushCompactLatticeWeights(&det_clat);
				MinimizeCompactLattice(&det_clat);
			}

			int32 t;
			TopSortCompactLatticeIfNeeded(&det_clat);
			double depth = CompactLatticeDepth(det_clat, &t);
			sum_depth_in += lat.NumStates();
			sum_depth_out += depth * t;
			sum_t += t;

			fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &det_clat);
			compact_lat_writer.Write(key, det_clat);
			n_done++;
		}

		if (sum_t != 0.0) {
			if(file_log)
				file_log << "Average input-lattice depth (measured at at state level) is "
				<< (sum_depth_in / sum_t) << ", output depth is "
				<< (sum_depth_out / sum_t) << ", over " << sum_t << " frames "
				<< " (average num-frames = " << (sum_t / n_done) << ")." << "\n";
			else
			KALDI_LOG << "Average input-lattice depth (measured at at state level) is "
				<< (sum_depth_in / sum_t) << ", output depth is "
				<< (sum_depth_out / sum_t) << ", over " << sum_t << " frames "
				<< " (average num-frames = " << (sum_t / n_done) << ").";
		}
		if (file_log)
			file_log << "Done " << n_done << " lattices, determinization finished "
			<< "earlier than specified by the beam (or output was empty) on "
			<< n_warn << " of these." << "\n";
		else
			KALDI_LOG << "Done " << n_done << " lattices, determinization finished "
			<< "earlier than specified by the beam (or output was empty) on "
			<< n_warn << " of these.";
		return (n_done != 0 ? 0 : 1);
	}
	catch (const std::exception &e) {
		KALDI_ERR << e.what();
		return -1;
	}
}
