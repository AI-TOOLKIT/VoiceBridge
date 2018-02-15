/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on : Copyright 2013        Pegah Ghahremani
			   2013-2014   Johns Hopkins University (author: Daniel Povey)
			   Apache 2.0
			   See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/pitch-functions.h"
#include "feat/wave-reader.h"

#include "kaldi-win/src/kaldi_src.h"

int ComputeKaldiPitchFeats(int argc, char *argv[], fs::ofstream & file_log) {
	try {
		using namespace kaldi;
		const char *usage =
			"Apply Kaldi pitch extractor, starting from wav input.  Output is 2-dimensional\n"
			"features consisting of (NCCF, pitch in Hz), where NCCF is between -1 and 1, and\n"
			"higher for voiced frames.  You will typically pipe this into\n"
			"process-kaldi-pitch-feats.\n"
			"Usage: compute-kaldi-pitch-feats [options...] <wav-rspecifier> <feats-wspecifier>\n"
			"e.g.\n"
			"compute-kaldi-pitch-feats --sample-frequency=8000 scp:wav.scp ark:- \n"
			"\n"
			"See also: process-kaldi-pitch-feats, compute-and-process-kaldi-pitch-feats\n";

		ParseOptions po(usage);
		PitchExtractionOptions pitch_opts;
		int32 channel = -1; // Note: this isn't configurable because it's not a very
							// good idea to control it this way: better to extract the
							// on the command line (in the .scp file) using sox or
							// similar.

		pitch_opts.Register(&po);

		po.Read(argc, argv);

		if (po.NumArgs() != 2) {
			//po.PrintUsage();
			//exit(1);
			KALDI_ERR << "Wrong arguments.";
			return -1;
		}

		std::string wav_rspecifier = po.GetArg(1),
			feat_wspecifier = po.GetArg(2);

		SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
		BaseFloatMatrixWriter feat_writer(feat_wspecifier);

		int32 num_done = 0, num_err = 0;
		for (; !wav_reader.Done(); wav_reader.Next()) {
			std::string utt = wav_reader.Key();
			const WaveData &wave_data = wav_reader.Value();

			int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
			{
				KALDI_ASSERT(num_chan > 0);
				// reading code if no channels.
				if (channel == -1) {
					this_chan = 0;
					if (num_chan != 1)
						KALDI_WARN << "Channel not specified but you have data with "
						<< num_chan << " channels; defaulting to zero";
				}
				else {
					if (this_chan >= num_chan) {
						KALDI_WARN << "File with id " << utt << " has "
							<< num_chan << " channels but you specified channel "
							<< channel << ", producing no output.";
						continue;
					}
				}
			}

			if (pitch_opts.samp_freq != wave_data.SampFreq()) {
				KALDI_ERR << "Sample frequency mismatch: you specified "
					<< pitch_opts.samp_freq << " but data has "
					<< wave_data.SampFreq() << " (use --sample-frequency "
					<< "option).  Utterance is " << utt;
				return -1;
			}

			SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
			Matrix<BaseFloat> features;
			try {
				ComputeKaldiPitch(pitch_opts, waveform, &features);
			}
			catch (...) {
				KALDI_WARN << "Failed to compute pitch for utterance " << utt;
				num_err++;
				continue;
			}

			feat_writer.Write(utt, features);
			if (num_done % 50 == 0 && num_done != 0) {
				if (file_log)
					file_log << "Processed " << num_done << " utterances" << "\n";
				else
					KALDI_VLOG(2) << "Processed " << num_done << " utterances";
			}
			num_done++;
		}
		if (file_log)
			file_log << "Done " << num_done << " utterances, " << num_err << " with errors." << "\n";
		else
			KALDI_LOG << "Done " << num_done << " utterances, " << num_err << " with errors.";
		return (num_done != 0 ? 0 : 1);
	}
	catch (const std::exception &e) {
		KALDI_ERR << e.what();
		return -1;
	}
}

