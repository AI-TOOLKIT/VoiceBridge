/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
Based on: see below
*/
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2008, Massachusetts Institute of Technology              //
// Copyright (c) 2010-2013, Giulio Paci <giuliopaci@gmail.com>            //
// All rights reserved.                                                   //
//                                                                        //
// Redistribution and use in source and binary forms, with or without     //
// modification, are permitted provided that the following conditions are //
// met:                                                                   //
//                                                                        //
//     * Redistributions of source code must retain the above copyright   //
//       notice, this list of conditions and the following disclaimer.    //
//                                                                        //
//     * Redistributions in binary form must reproduce the above          //
//       copyright notice, this list of conditions and the following      //
//       disclaimer in the documentation and/or other materials provided  //
//       with the distribution.                                           //
//                                                                        //
//     * Neither the name of the Massachusetts Institute of Technology    //
//       nor the names of its contributors may be used to endorse or      //
//       promote products derived from this software without specific     //
//       prior written permission.                                        //
//                                                                        //
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS    //
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT      //
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  //
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT   //
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,  //
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT       //
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  //
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY  //
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT    //
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE  //
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   //
////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <cstdio>
#include "util/CommandOptions.h"
//#include "util/Logger.h"
#include "util/ZFile.h"
#include "Types.h"
#include "Lattice.h"
#include "PerplexityOptimizer.h"
#include "WordErrorRateOptimizer.h"

#include "mitlm.h"

#ifdef F77_DUMMY_MAIN
#  ifdef __cplusplus
extern "C"
#  endif
int F77_DUMMY_MAIN () { return 1; }
#endif

using std::vector;
using std::string;

////////////////////////////////////////////////////////////////////////////////

//const char *headerDesc =
//"Usage: evaluate-ngram [Options]\n"
//"\n"
//"Evaluates the performance of an n-gram language model.  It also supports\n"
//"various n-gram language model conversions, including changes in order,\n"
//"vocabulary, and file format.\n"
//"\n"
//"Filename argument can be an ASCII file, a compressed file (ending in .Z or .gz),\n"
//"or '-' to indicate stdin/stdout.\n";
//
//const char *footerDesc_tmpl =
//"---------------------------------------------------------------\n"
//"| %-59s |\n"
//"| Copyright (C) 2009 Bo-June (Paul) Hsu                       |\n"
//"| MIT Computer Science and Artificial Intelligence Laboratory |\n"
//"---------------------------------------------------------------\n";

////////////////////////////////////////////////////////////////////////////////

VOICEBRIDGE_API int EvaluateNgram(int argc, char* argv[]) {
	/*
    // Parse command line options.
    char *footerDesc = new char[strlen(footerDesc_tmpl)+strlen(PACKAGE_STRING)+1+59];
    sprintf(footerDesc, footerDesc_tmpl, PACKAGE_STRING);
    mitlm::CommandOptions opts(headerDesc, footerDesc);
    delete [] footerDesc;
	*/
	//@+zso
	std::string sheaderDesc(""), sfooterDesc("");
	mitlm::CommandOptions opts(sheaderDesc.c_str(), sfooterDesc.c_str());

    opts.AddOption("h,help", "Print this message.");
    opts.AddOption("verbose", "Set verbosity level.", "1", "int");
    opts.AddOption("o,order", "Set the n-gram order of the estimated LM.", "3", "int");
    opts.AddOption("v,vocab", "Fix the vocab to only words from the specified file.", NULL, "file");
    opts.AddOption("l,lm", "Load specified LM.", NULL, "file");
    opts.AddOption("cl,compile-lattices", "[SLS] Compile lattices into a binary format.", NULL, "file");
    opts.AddOption("wb,write-binary", "Write LM/counts files in binary format.", "false", "boolean");
    opts.AddOption("wv,write-vocab", "Write LM vocab to file.", NULL, "file");
    opts.AddOption("wl,write-lm", "Write ARPA backoff LM to file.", NULL, "file");
    opts.AddOption("ep,eval-perp", "Compute test set perplexity.", NULL, "files");
    opts.AddOption("ew,eval-wer", "Compute test set lattice word error rate.", NULL, "files");
    opts.AddOption("em,eval-margin", "Compute test set lattice margin.", NULL, "files");
    if (!opts.ParseArguments(argc, (const char **)argv) ||
        opts["help"] != NULL) {
        //std::cout << std::endl;
        //opts.PrintHelp();
        //return 1;
		return -1;
    }

    // Process basic command line arguments.
    size_t order = atoi(opts["order"]);
    bool writeBinary = mitlm::AsBoolean(opts["write-binary"]);
    //@-zso mitlm::Logger::SetVerbosity(atoi(opts["verbose"]));
    if (!opts["lm"]) {
        //mitlm::Logger::Error(0, "Language model must be specified using -lm.");
		LOGTW_ERROR << "Language model must be specified using -lm.";
        //exit(1);
		return -1;
    }

    // Load language model.
    mitlm::ArpaNgramLM lm(order);
    if (opts["vocab"]) {
        //mitlm::Logger::Log(1, "Loading vocab %s...\n", opts["vocab"]);
		LOGTW_INFO << "Loading vocab " << opts["vocab"] << "...";
	mitlm::ZFile vocabZFile(opts["vocab"]);
        lm.LoadVocab(vocabZFile);
    }
    //mitlm::Logger::Log(1, "Loading LM %s...\n", opts["lm"]);
	LOGTW_INFO << "Loading LM " << opts["lm"] << "...";
    mitlm::ZFile lmZFile(opts["lm"], "r");
    lm.LoadLM(lmZFile);

    // Compile lattices.
    if (opts["compile-lattices"]) {
        //mitlm::Logger::Log(0, "Compiling lattices %s:\n", opts["compile-lattices"]);
		LOGTW_INFO << "Compiling lattices " << opts["compile-lattices"] << "...";
        mitlm::ZFile latticesZFile(opts["compile-lattices"]);
	mitlm::WordErrorRateOptimizer eval(lm, order);
        eval.LoadLattices(latticesZFile);
        string outFile(opts["compile-lattices"]);
        outFile += ".bin";
        mitlm::ZFile outZFile(outFile.c_str(), "w");
        eval.SaveLattices(outZFile);
    }

    // Evaluate LM.
    mitlm::ParamVector params(lm.defParams());
    if (opts["eval-perp"]) {
        //mitlm::Logger::Log(0, "Perplexity Evaluations:\n");
		LOGTW_INFO << "Perplexity Evaluations:";
        vector<string> evalFiles;
        mitlm::trim_split(evalFiles, opts["eval-perp"], ',');
        for (size_t i = 0; i < evalFiles.size(); i++) {
            //mitlm::Logger::Log(1, "Loading eval set %s...\n", evalFiles[i].c_str());
			LOGTW_INFO << "Loading eval set " << evalFiles[i].c_str() << "...";
            mitlm::ZFile evalZFile(evalFiles[i].c_str());
	    mitlm::PerplexityOptimizer eval(lm, order);
            eval.LoadCorpus(evalZFile);

            //mitlm::Logger::Log(0, "\t%s\t%.3f\n", evalFiles[i].c_str(), eval.ComputePerplexity(params));
			LOGTW_INFO << "\t" << evalFiles[i].c_str() << "\t" << eval.ComputePerplexity(params);
        }
    }
    if (opts["eval-margin"]) {
        //mitlm::Logger::Log(0, "Margin Evaluations:\n");
		LOGTW_INFO << "Margin Evaluations:";
        vector<string> evalFiles;
        mitlm::trim_split(evalFiles, opts["eval-margin"], ',');
        for (size_t i = 0; i < evalFiles.size(); i++) {
            //mitlm::Logger::Log(1, "Loading eval lattices %s...\n", evalFiles[i].c_str());
			LOGTW_INFO << "Loading eval lattices " << evalFiles[i].c_str() << "...";
            mitlm::ZFile evalZFile(evalFiles[i].c_str());
	    mitlm::WordErrorRateOptimizer eval(lm, order);
            eval.LoadLattices(evalZFile);

            //mitlm::Logger::Log(0, "\t%s\t%.3f\n", evalFiles[i].c_str(), eval.ComputeMargin(params));
			LOGTW_INFO << "\t" << evalFiles[i].c_str() << "\t" << eval.ComputeMargin(params);
        }
    }
    if (opts["eval-wer"]) {
        //mitlm::Logger::Log(0, "WER Evaluations:\n");
		LOGTW_INFO << "WER Evaluations:";
        vector<string> evalFiles;
        mitlm::trim_split(evalFiles, opts["eval-wer"], ',');
        for (size_t i = 0; i < evalFiles.size(); i++) {
            //mitlm::Logger::Log(1, "Loading eval lattices %s...\n", evalFiles[i].c_str());
			LOGTW_INFO << "Loading eval lattices " << evalFiles[i].c_str() << "...";
            mitlm::ZFile evalZFile(evalFiles[i].c_str());
	    mitlm::WordErrorRateOptimizer eval(lm, order);
            eval.LoadLattices(evalZFile);

            //mitlm::Logger::Log(0, "\t%s\t%.2f%%\n", evalFiles[i].c_str(), eval.ComputeWER(params));
			LOGTW_INFO << "\t" << evalFiles[i].c_str() << "\t" << eval.ComputeWER(params);
        }
    }

    // Save results.
    if (opts["write-vocab"]) {
        //mitlm::Logger::Log(1, "Saving vocabulary to %s...\n", opts["write-vocab"]);
		LOGTW_INFO << "Saving vocabulary to " << opts["write-vocab"] << "...";

        mitlm::ZFile vocabZFile(opts["write-vocab"], "w");
        lm.SaveVocab(vocabZFile);
    }
    if (opts["write-lm"]) {
        //mitlm::Logger::Log(1, "Saving LM to %s...\n", opts["write-lm"]);
		LOGTW_INFO << "Saving LM to " << opts["write-lm"] << "...";

        mitlm::ZFile lmZFile(opts["write-lm"], "w");
        lm.SaveLM(lmZFile, writeBinary);
    }

    return 0;
}
