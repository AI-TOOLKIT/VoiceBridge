/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on :  
*/
// Copyright 2009-2011  Microsoft Corporation
//                2014  Guoguo Chen
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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

#include "kaldi-win/src/kaldi_src.h"

/*
	GmmPostToGpost :  Convert state-level posteriors to Gaussian-level posteriors
*/
int GmmPostToGpost(int argc, char *argv[], fs::ofstream & file_log) {
	using namespace kaldi;
	try {
		const char *usage =
			"Convert state-level posteriors to Gaussian-level posteriors\n"
			"Usage:  gmm-post-to-gpost [options] <model-in> <feature-rspecifier> <posteriors-rspecifier> "
			"<gpost-wspecifier>\n"
			"e.g.: \n"
			" gmm-post-to-gpost 1.mdl scp:train.scp ark:1.post ark:1.gpost\n";

		ParseOptions po(usage);
		bool binary = true;
		BaseFloat rand_prune = 0.0;
		po.Register("binary", &binary, "Write output in binary mode");
		po.Register("rand-prune", &rand_prune, "Randomized pruning of posteriors less than this");
		po.Read(argc, argv);

		if (po.NumArgs() != 4) {
			//po.PrintUsage();
			//exit(1);
			KALDI_ERR << "Wrong arguments.";
			return -1;
		}

		std::string model_filename = po.GetArg(1),
			feature_rspecifier = po.GetArg(2),
			posteriors_rspecifier = po.GetArg(3),
			gpost_wspecifier = po.GetArg(4);

		using namespace kaldi;
		typedef kaldi::int32 int32;

		AmDiagGmm am_gmm;
		TransitionModel trans_model;
		{
			bool binary;
			Input ki(model_filename, &binary);
			trans_model.Read(ki.Stream(), binary);
			am_gmm.Read(ki.Stream(), binary);
		}

		double tot_like = 0.0;
		double tot_t = 0.0;

		SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

		GaussPostWriter gpost_writer(gpost_wspecifier);

		int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
		for (; !feature_reader.Done(); feature_reader.Next()) {
			std::string key = feature_reader.Key();
			if (!posteriors_reader.HasKey(key)) {
				num_no_posterior++;
			}
			else {
				const Matrix<BaseFloat> &mat = feature_reader.Value();
				const Posterior &posterior = posteriors_reader.Value(key);
				GaussPost gpost(posterior.size());

				if (posterior.size() != mat.NumRows()) {
					KALDI_WARN << "Posterior vector has wrong size " << (posterior.size()) << " vs. " << (mat.NumRows());
					num_other_error++;
					continue;
				}

				num_done++;
				BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

				Posterior pdf_posterior;
				ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
				for (size_t i = 0; i < posterior.size(); i++) {
					gpost[i].reserve(pdf_posterior[i].size());
					for (size_t j = 0; j < pdf_posterior[i].size(); j++) {
						int32 pdf_id = pdf_posterior[i][j].first;
						BaseFloat weight = pdf_posterior[i][j].second;
						const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
						Vector<BaseFloat> this_post_vec;
						BaseFloat like =
							gmm.ComponentPosteriors(mat.Row(i), &this_post_vec);
						this_post_vec.Scale(weight);
						if (rand_prune > 0.0)
							for (int32 k = 0; k < this_post_vec.Dim(); k++)
								this_post_vec(k) = RandPrune(this_post_vec(k),
									rand_prune);
						if (!this_post_vec.IsZero())
							gpost[i].push_back(std::make_pair(pdf_id, this_post_vec));
						tot_like_this_file += like * weight;
						tot_weight += weight;
					}
				}
				if (file_log)
					file_log << "Average like for this file is "
					<< (tot_like_this_file / tot_weight) << " over "
					<< tot_weight << " frames." << "\n";
				else 
					KALDI_VLOG(1) << "Average like for this file is "
					<< (tot_like_this_file / tot_weight) << " over "
					<< tot_weight << " frames.";

				tot_like += tot_like_this_file;
				tot_t += tot_weight;
				gpost_writer.Write(key, gpost);
			}
		}

		if (file_log) {
			file_log << "Done " << num_done << " files, " << num_no_posterior
				<< " with no posteriors, " << num_other_error
				<< " with other errors." << "\n";

			file_log << "Overall avg like per frame (Gaussian only) = "
				<< (tot_like / tot_t) << " over " << tot_t << " frames." << "\n";

			file_log << "Done converting post to gpost" << "\n";
		}
		else {
			KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
				<< " with no posteriors, " << num_other_error
				<< " with other errors.";

			KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
				<< (tot_like / tot_t) << " over " << tot_t << " frames.";

			KALDI_LOG << "Done converting post to gpost";
		}

		if (num_done != 0) return 0;
		else return 1;
	}
	catch (const std::exception &e) {
		KALDI_ERR << e.what();
		return -1;
	}
}


