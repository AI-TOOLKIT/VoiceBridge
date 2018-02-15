/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :
 Copyright 2009-2011  Microsoft Corporation
                2014  Johns Hopkins University (authors: Jan Trmal, Daniel Povey)
                2015  Brno Universiry of technology (author: Karel Vesely)

See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/parse-options.h"
#include "tree/context-dep.h"
#include "util/edit-distance.h"

#include "kaldi-win/src/kaldi_src.h"

/*
	//NOTE ZSO @+16012018: added output file as an option!

	Compute WER by comparing different transcriptions. Takes two transcription files, in integer or text format,
	and outputs overall WER statistics to standard output or to a file if specified.
*/
int ComputeWer(int argc, char *argv[], fs::ofstream & file_log) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Compute WER by comparing different transcriptions\n"
        "Takes two transcription files, in integer or text format,\n"
        "and outputs overall WER statistics to standard output.\n"
        "\n"
        "Usage: compute-wer [options] <ref-rspecifier> <hyp-rspecifier>\n"
        "E.g.: compute-wer --text --mode=present ark:data/train/text ark:hyp_text\n"
        "See also: align-text,\n"
        "Example scoring script: egs/wsj/s5/steps/score_kaldi.sh\n";

    ParseOptions po(usage);

    std::string mode = "strict";
    po.Register("mode", &mode,
                "Scoring mode: \"present\"|\"all\"|\"strict\":\n"
                "  \"present\" means score those we have transcriptions for\n"
                "  \"all\" means treat absent transcriptions as empty\n"
                "  \"strict\" means die if all in ref not also in hyp");
    
    bool dummy = false;
    po.Register("text", &dummy, "Deprecated option! Keeping for compatibility reasons.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) { //@+zso
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "Wrong arguments.";
		return -1;
    }

    std::string ref_rspecifier = po.GetArg(1);
    std::string hyp_rspecifier = po.GetArg(2);
	std::string wspecifier = po.GetArg(3); //@+zso 16012018 write results to output file

    if (mode != "strict" && mode != "present" && mode != "all") {
      KALDI_ERR << "--mode option invalid: expected \"present\"|\"all\"|\"strict\", got "
                << mode;
	  return -1; //VB
    }

    int32 num_words = 0, word_errs = 0, num_sent = 0, sent_errs = 0,
        num_ins = 0, num_del = 0, num_sub = 0, num_absent_sents = 0;

    // Both text and integers are loaded as vector of strings,
    SequentialTokenVectorReader ref_reader(ref_rspecifier);
    RandomAccessTokenVectorReader hyp_reader(hyp_rspecifier);
    
    // Main loop, accumulate WER stats,
    for (; !ref_reader.Done(); ref_reader.Next()) {
      std::string key = ref_reader.Key();
      const std::vector<std::string> &ref_sent = ref_reader.Value();
      std::vector<std::string> hyp_sent;
      if (!hyp_reader.HasKey(key)) {
		  if (mode == "strict") {
			  KALDI_ERR << "No hypothesis for key " << key << " and strict "
				  "mode specifier.";
			  return -1; //VB
		  }
        num_absent_sents++;
        if (mode == "present")  // do not score this one.
          continue;
      } else {
        hyp_sent = hyp_reader.Value(key);
      }
      num_words += ref_sent.size();
      int32 ins, del, sub;
      word_errs += LevenshteinEditDistance(ref_sent, hyp_sent, &ins, &del, &sub);
      num_ins += ins;
      num_del += del;
      num_sub += sub;

      num_sent++;
      sent_errs += (ref_sent != hyp_sent);
    }

    // Compute WER, SER,
    BaseFloat percent_wer = 100.0 * static_cast<BaseFloat>(word_errs)
        / static_cast<BaseFloat>(num_words);
    BaseFloat percent_ser = 100.0 * static_cast<BaseFloat>(sent_errs)
        / static_cast<BaseFloat>(num_sent);

    // Print the ouptut to KALDI_LOG or to the output file if specified
	/*@-zso
    std::cout.precision(2);
    std::cerr.precision(2);
    std::cout << "%WER " << std::fixed << percent_wer << " [ " << word_errs
              << " / " << num_words << ", " << num_ins << " ins, "
              << num_del << " del, " << num_sub << " sub ]"
              << (num_absent_sents != 0 ? " [PARTIAL]" : "") << '\n';
    std::cout << "%SER " << std::fixed << percent_ser <<  " [ "
               << sent_errs << " / " << num_sent << " ]\n";
    std::cout << "Scored " << num_sent << " sentences, "
              << num_absent_sents << " not present in hyp.\n";
	*/
	//@+zso
	KALDI_LOG << "%WER " << std::fixed << std::setprecision(2) << percent_wer << " [ " << word_errs
		<< " / " << num_words << ", " << num_ins << " ins, "
		<< num_del << " del, " << num_sub << " sub ]"
		<< (num_absent_sents != 0 ? " [PARTIAL]" : "") << '\n';
	KALDI_LOG << "%SER " << std::fixed << std::setprecision(2) << percent_ser << " [ "
		<< sent_errs << " / " << num_sent << " ]\n";
	KALDI_LOG << "Scored " << num_sent << " sentences, "
		<< num_absent_sents << " not present in hyp.\n";
	//NOTE ZSO: special file format for easy reading of the parameters
	if (wspecifier != "") {
		fs::ofstream file_ws(wspecifier, fs::ofstream::binary | fs::ofstream::out);
		if (!file_ws) {
			KALDI_ERR << "Output file is not accessible " << wspecifier << ".";
			return -1;
		}
		file_ws << std::fixed << std::setprecision(2) 
			<< percent_wer << " " 
			<< word_errs << " " 
			<< num_words << " " 
			<< num_ins << " " 
			<< num_del << " " 
			<< num_sub << " " 
			<< percent_ser << " " 
			<< sent_errs << " " 
			<< num_sent << '\n';
		file_ws.flush(); file_ws.close();
	}

    return 0;
  } catch(const std::exception &e) {
	  KALDI_ERR << e.what();
    return -1;
  }
}
