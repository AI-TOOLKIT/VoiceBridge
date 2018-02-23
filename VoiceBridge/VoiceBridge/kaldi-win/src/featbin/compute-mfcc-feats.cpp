/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : 
	Copyright 2009-2012  Microsoft Corporation
    Johns Hopkins University (author: Daniel Povey)
	Apache 2.0
	See ../../COPYING for clarification regarding multiple authors
*/
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"

#include "kaldi-win/src/kaldi_src.h"

int ComputeMFCCFeats(int argc, char *argv[], fs::ofstream & file_log)
{
	try {
		using namespace kaldi;
		const char *usage =
			"Create MFCC feature files.\n"
			"Usage:  compute-mfcc-feats [options...] <wav-rspecifier> <feats-wspecifier>\n";

		// construct all the global objects
		ParseOptions po(usage);
		MfccOptions mfcc_opts;
		bool subtract_mean = false;
		BaseFloat vtln_warp = 1.0;
		std::string vtln_map_rspecifier;
		std::string utt2spk_rspecifier;
		int32 channel = -1;
		BaseFloat min_duration = 0.0;
		// Define defaults for gobal options
		std::string output_format = "kaldi";

		// Register the MFCC option struct
		mfcc_opts.Register(&po);

		// Register the options
		po.Register("output-format", &output_format, "Format of the output "
			"files [kaldi, htk]");
		po.Register("subtract-mean", &subtract_mean, "Subtract mean of each "
			"feature file [CMS]; not recommended to do it this way. ");
		po.Register("vtln-warp", &vtln_warp, "Vtln warp factor (only applicable "
			"if vtln-map not specified)");
		po.Register("vtln-map", &vtln_map_rspecifier, "Map from utterance or "
			"speaker-id to vtln warp factor (rspecifier)");
		po.Register("utt2spk", &utt2spk_rspecifier, "Utterance to speaker-id map "
			"rspecifier (if doing VTLN and you have warps per speaker)");
		po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, "
			"0 -> left, 1 -> right)");
		po.Register("min-duration", &min_duration, "Minimum duration of segments "
			"to process (in seconds).");

		po.Read(argc, argv);

		if (po.NumArgs() != 2) {
			//po.PrintUsage();
			//exit(1);
			KALDI_ERR << "wrong arguments.";
			return -1;
		}

		std::string wav_rspecifier = po.GetArg(1);

		std::string output_wspecifier = po.GetArg(2);

		Mfcc mfcc(mfcc_opts);

		SequentialTableReader<WaveHolder> reader(wav_rspecifier);
		BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.
		TableWriter<HtkMatrixHolder> htk_writer;

		if (utt2spk_rspecifier != "")
			KALDI_ASSERT(vtln_map_rspecifier != "" && "the utt2spk option is only "
				"needed if the vtln-map option is used.");
		RandomAccessBaseFloatReaderMapped vtln_map_reader(vtln_map_rspecifier,
			utt2spk_rspecifier);

		if (output_format == "kaldi") {
			if (!kaldi_writer.Open(output_wspecifier)) {
				KALDI_ERR << "Could not initialize output with wspecifier " << output_wspecifier;
				return -1; //VB
			}
		}
		else if (output_format == "htk") {
			if (!htk_writer.Open(output_wspecifier)) {
				KALDI_ERR << "Could not initialize output with wspecifier " << output_wspecifier;
				return -1; //VB
			}
		}
		else {
			KALDI_ERR << "Invalid output_format string " << output_format;
			return -1; //VB
		}

		int32 num_utts = 0, num_success = 0;
		for (; !reader.Done(); reader.Next()) {
			num_utts++;
			std::string utt = reader.Key();
			const WaveData &wave_data = reader.Value();
			if (wave_data.Duration() < min_duration) {
				KALDI_WARN << "File: " << utt << " is too short ("
					<< wave_data.Duration() << " sec): producing no output.";
				continue;
			}
			int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
			{  // This block works out the channel (0=left, 1=right...)
				KALDI_ASSERT(num_chan > 0);  // should have been caught in
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
			BaseFloat vtln_warp_local;  // Work out VTLN warp factor.
			if (vtln_map_rspecifier != "") {
				if (!vtln_map_reader.HasKey(utt)) {
					KALDI_WARN << "No vtln-map entry for utterance-id (or speaker-id) "
						<< utt;
					continue;
				}
				vtln_warp_local = vtln_map_reader.Value(utt);
			}
			else {
				vtln_warp_local = vtln_warp;
			}

			SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
			Matrix<BaseFloat> features;
			try {
				mfcc.ComputeFeatures(waveform, wave_data.SampFreq(), vtln_warp_local, &features);
			}
			catch (...) {
				KALDI_WARN << "Failed to compute features for utterance "
					<< utt;
				continue;
			}
			if (subtract_mean) {
				Vector<BaseFloat> mean(features.NumCols());
				mean.AddRowSumMat(1.0, features);
				mean.Scale(1.0 / features.NumRows());
				for (int32 i = 0; i < features.NumRows(); i++)
					features.Row(i).AddVec(-1.0, mean);
			}
			if (output_format == "kaldi") {
				kaldi_writer.Write(utt, features);
			}
			else {
				std::pair<Matrix<BaseFloat>, HtkHeader> p;
				p.first.Resize(features.NumRows(), features.NumCols());
				p.first.CopyFromMat(features);
				HtkHeader header = {
					features.NumRows(),
					100000,  // 10ms shift
					static_cast<int16>(sizeof(float)*(features.NumCols())),
					static_cast<uint16>(006 | // MFCC
					(mfcc_opts.use_energy ? 0100 : 020000)) // energy; otherwise c0
				};
				p.second = header;
				htk_writer.Write(utt, p);
			}
			if (num_utts % 10 == 0) {
				if (file_log)
					file_log << "Processed " << num_utts << " utterances" << "\n";
				else KALDI_LOG << "Processed " << num_utts << " utterances" << "\n";
			}
			if (file_log)
				file_log << "Processed features for key " << utt << "\n";
			else KALDI_VLOG(2) << "Processed features for key " << utt << "\n";
			num_success++;
		}
		if (file_log)
			file_log << "Done " << num_success << " out of " << num_utts << " utterances." << "\n";
		else KALDI_LOG << "Done " << num_success << " out of " << num_utts << " utterances." << "\n";

		return (num_success != 0 ? 0 : 1);
	}
	catch (const std::exception &e) {
		KALDI_ERR << e.what();
		return -1;
	}
}

