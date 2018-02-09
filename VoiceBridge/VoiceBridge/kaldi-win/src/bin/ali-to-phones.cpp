/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on : Copyright 2009-2011  Microsoft Corporation, Apache 2
						 2015       IMSL, PKU-HKUST (author: Wei Shi)
						 See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

#include "kaldi-win/src/kaldi_src.h"

int AliToPhones(int argc, char *argv[], fs::ofstream & file_log) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert model-level alignments to phone-sequences (in integer, "
        "not text, form)\n"
        "Usage:  ali-to-phones  [options] <model> <alignments-rspecifier> "
        "<phone-transcript-wspecifier|ctm-wxfilename>\n"
        "e.g.: \n"
        " ali-to-phones 1.mdl ark:1.ali ark:-\n"
        "or:\n"
        " ali-to-phones --ctm-output 1.mdl ark:1.ali 1.ctm\n"
        "See also: show-alignments lattice-align-phones\n";
    ParseOptions po(usage);
    bool per_frame = false;
    bool write_lengths = false;
    bool ctm_output = false;
    BaseFloat frame_shift = 0.01;
    po.Register("ctm-output", &ctm_output,
                "If true, output the alignments in ctm format "
                "(the confidences will be set to 1)");
    po.Register("frame-shift", &frame_shift,
                "frame shift used to control the times of the ctm output");
    po.Register("per-frame", &per_frame,
                "If true, write out the frame-level phone alignment "
                "(else phone sequence)");
    po.Register("write-lengths", &write_lengths,
                "If true, write the #frames for each phone (different format)");


    po.Read(argc, argv);

    KALDI_ASSERT(!(per_frame && write_lengths) && "Incompatible options.");

    if (po.NumArgs() != 3) {
      //po.PrintUsage();
      //exit(1);
		KALDI_ERR << "wrong arguments.";
		return -1;
    }

    std::string model_filename = po.GetArg(1),
        alignments_rspecifier = po.GetArg(2);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    SequentialInt32VectorReader reader(alignments_rspecifier);
    std::string empty;
    Int32VectorWriter phones_writer(ctm_output ? empty :
                                    (write_lengths ? empty : po.GetArg(3)));
    Int32PairVectorWriter pair_writer(ctm_output ? empty :
                                      (write_lengths ? po.GetArg(3) : empty));

    std::string ctm_wxfilename(ctm_output ? po.GetArg(3) : empty);
    Output ctm_writer(ctm_wxfilename, false);
    if (ctm_output) {
      ctm_writer.Stream() << std::fixed;
      ctm_writer.Stream().precision(frame_shift >= 0.01 ? 2 : 3);
    }

    int32 n_done = 0;

    for (; !reader.Done(); reader.Next()) {
      std::string key = reader.Key();
      const std::vector<int32> &alignment = reader.Value();

      std::vector<std::vector<int32> > split;
      SplitToPhones(trans_model, alignment, &split);

      if (ctm_output) {
        BaseFloat phone_start = 0.0;
        for (size_t i = 0; i < split.size(); i++) {
          KALDI_ASSERT(!split[i].empty());
          int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
          int32 num_repeats = split[i].size();
          ctm_writer.Stream() << key << " 1 " << phone_start << " "
                      << (frame_shift * num_repeats) << " " << phone << std::endl;
          phone_start += frame_shift * num_repeats;
        }
      } else if (!write_lengths) {
        std::vector<int32> phones;
        for (size_t i = 0; i < split.size(); i++) {
          KALDI_ASSERT(!split[i].empty());
          int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
          int32 num_repeats = split[i].size();
          //KALDI_ASSERT(num_repeats!=0);
          if (per_frame)
            for(int32 j = 0; j < num_repeats; j++)
              phones.push_back(phone);
          else
            phones.push_back(phone);
        }
        phones_writer.Write(key, phones);
      } else {
        std::vector<std::pair<int32, int32> > pairs;
        for (size_t i = 0; i < split.size(); i++) {
          KALDI_ASSERT(split[i].size() > 0);
          int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
          int32 num_repeats = split[i].size();
          //KALDI_ASSERT(num_repeats!=0);
          pairs.push_back(std::make_pair(phone, num_repeats));
        }
        pair_writer.Write(key, pairs);
      }
      n_done++;
    }
	if(file_log)
		file_log << "Done " << n_done << " utterances.";
	else
    KALDI_LOG << "Done " << n_done << " utterances.";
	return 0;
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}


