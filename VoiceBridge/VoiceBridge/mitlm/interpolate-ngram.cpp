/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
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
#include "util/ZFile.h"
//#include "util/Logger.h"
#include "Types.h"
#include "Smoothing.h"
#include "NgramLM.h"
#include "InterpolatedNgramLM.h"
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
//"Usage: interpolate-ngram [Options]\n"
//"\n"
//"Interpolates multiple n-gram models by computing appropriate interpolation\n"
//"weights from optional features and constructing a statically interpolated\n"
//"n-gram model.  Parameters can be optionally tuned to optimize development set\n"
//"performance.\n"
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

VOICEBRIDGE_API int InterpolateNgram(int argc, char* argv[]) {
	/*@-zso
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
    opts.AddOption("u,unk", "Replace all out of vocab words with <unk>.", "false", "boolean");
    opts.AddOption("l,lm", "Interpolate specified LM files.", NULL, "file");
    opts.AddOption("t,text", "Interpolate models trained from text files.", NULL, "files");
    opts.AddOption("c,counts", "Interpolate models trained from counts files.", NULL, "files");
    opts.AddOption("s,smoothing", "Specify smoothing algorithms.", "ModKN", "ML, FixKN, FixModKN, FixKN#, KN, ModKN, KN#");
    opts.AddOption("wf,weight-features", "Specify n-gram weighting features.", NULL, "features-template");
    opts.AddOption("i,interpolation", "Specify interpolation mode.", "LI", "LI, CM, GLI");
    opts.AddOption("if,interpolation-features", "Specify interpolation features.", NULL, "features-template");
    opts.AddOption("tpo,tie-param-order", "Tie parameters across n-gram order.", "true", "boolean");
    opts.AddOption("tpl,tie-param-lm", "Tie parameters across LM components.", "false", "boolean");
    opts.AddOption("p,params", "Set initial model params.", NULL, "file");
    opts.AddOption("oa,opt-alg", "Specify optimization algorithm.", "LBFGS", "Powell, LBFGS, LBFGSB");
    opts.AddOption("op,opt-perp", "Tune params to minimize dev set perplexity.", NULL, "file");
    opts.AddOption("ow,opt-wer", "Tune params to minimize lattice word error rate.", NULL, "file");
    opts.AddOption("om,opt-margin", "Tune params to minimize lattice margin.", NULL, "file");
    opts.AddOption("wb,write-binary", "Write LM/counts files in binary format.", "false", "boolean");
    opts.AddOption("wp,write-params", "Write tuned model params to file.", NULL, "file");
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

    // Read language models.
    vector<mitlm::SharedPtr<mitlm::NgramLMBase> > lms;
    vector<string> corpusFiles;
    if (opts["text"] || opts["counts"]) {
        if (opts["text"] && opts["counts"]) {
            //mitlm::Logger::Error(1, "Cannot specify both -text and -counts.\n");
			LOGTW_ERROR << "Cannot specify both -text and -counts.";
            //exit(1);
			return -1;
        }

        vector<string> smoothings;
        vector<string> features;
        bool           fromText = opts["text"];
        if (fromText)
            mitlm::trim_split(corpusFiles, opts["text"], ',');
        else
            mitlm::trim_split(corpusFiles, opts["counts"], ',');
        mitlm::trim_split(smoothings, opts["smoothing"], ';');
        mitlm::trim_split(features, opts["weight-features"], ';');

        if ((smoothings.size() != 1 && smoothings.size() != corpusFiles.size()) ||
            (features.size() > 1 && features.size() != corpusFiles.size())) {
            //mitlm::Logger::Error(1, "Inconsistent number of LM components.\n");
			LOGTW_ERROR << "Inconsistent number of LM components.";

            //exit(1);
			return -1;
        }

        for (size_t i = 0; i < corpusFiles.size(); i++) {
            mitlm::NgramLM *pLM = new mitlm::NgramLM(order);
            pLM->Initialize(opts["vocab"], mitlm::AsBoolean(opts["unk"]),
                            fromText ? corpusFiles[i].c_str() : NULL, 
                            fromText ? NULL : corpusFiles[i].c_str(), 
                            mitlm::GetItem(smoothings, i), mitlm::GetItem(features, i));
            lms.push_back((mitlm::SharedPtr<mitlm::NgramLMBase>)pLM);
        }
    }

    // Read component language model input files.
    if (opts["lm"]) {
        vector<string> lmFiles;
        mitlm::trim_split(lmFiles, opts["lm"], ',');
        for (size_t l = 0; l < lmFiles.size(); l++) {
            //mitlm::Logger::Log(1, "Loading component LM %s...\n", lmFiles[l].c_str());
			LOGTW_INFO << "Loading component LM " << lmFiles[l].c_str() << "...";

            mitlm::ArpaNgramLM *pLM = new mitlm::ArpaNgramLM(order);
            if (opts["vocab"]) {
                mitlm::ZFile vocabZFile(opts["vocab"]);
                pLM->LoadVocab(vocabZFile);
                if (mitlm::AsBoolean(opts["unk"])) {
                    //mitlm::Logger::Error(1, "-unk with -lm is not implemented yet.\n");
					LOGTW_ERROR << "-unk with -lm is not implemented yet.";

                    //exit(1);
					return -1;
                }
            }
            mitlm::ZFile lmZFile(lmFiles[l].c_str(), "r");
            pLM->LoadLM(lmZFile);
            lms.push_back((mitlm::SharedPtr<mitlm::NgramLMBase>)pLM);
            corpusFiles.push_back(lmFiles[l]);
        }
    }

    if(lms.size() < 1) {
            //mitlm::Logger::Error(1, "No language model loaded.\n");
			LOGTW_ERROR << "No language model loaded.";

            //exit(1);
			return -1;
    }

    // Interpolate language models.
    //mitlm::Logger::Log(1, "Interpolating component LMs...\n");
	LOGTW_INFO << "Interpolating component LMs...";

    if (mitlm::AsBoolean(opts["tie-param-order"]))
        //mitlm::Logger::Log(1, "Tying parameters across n-gram order...\n");
		LOGTW_INFO << "Tying parameters across n-gram order...";
    if (mitlm::AsBoolean(opts["tie-param-lm"]))
        //mitlm::Logger::Log(1, "Tying parameters across LM components...\n");
		LOGTW_INFO << "Tying parameters across LM components...";
    mitlm::InterpolatedNgramLM ilm(order, mitlm::AsBoolean(opts["tie-param-order"]), mitlm::AsBoolean(opts["tie-param-lm"]));
    ilm.LoadLMs(lms);
    
    // Process features.
    vector<vector<string> > lmFeatures;
    if (opts["interpolation-features"]) {
        vector<string> features;
        mitlm::trim_split(features, opts["interpolation-features"], ';');
        if (features.size() != 1 && features.size() != lms.size()) {
            //mitlm::Logger::Error(1, "# components specified in -interpolation-features does not match number of LMs.\n");
			LOGTW_ERROR << "# components specified in -interpolation-features does not match number of LMs.";
            //exit(1);
			return -1;
        }
        lmFeatures.resize(features.size());
        for (size_t l = 0; l < features.size(); ++l)
            mitlm::trim_split(lmFeatures[l], features[l].c_str(), ',');
    }

    // Process interpolation.
    const char *interpolation = opts["interpolation"];
    //mitlm::Logger::Log(1, "Interpolation Method = %s\n", interpolation);
	LOGTW_INFO << "Interpolation Method = " << interpolation;

    if (strcmp(interpolation, "LI") == 0) {
        // Verify no features are specified.
        if (lmFeatures.size() > 0) {
            //mitlm::Logger::Error(1, "Linear interpolation uses no features.\n");
			LOGTW_ERROR << "Linear interpolation uses no features.";
            //exit(1);
			return -1;
        }
    } else {
        mitlm::Interpolation mode;
        if (strcmp(interpolation, "CM") == 0) {
            if (lmFeatures.size() > 0) {
                for (size_t l = 0; l < lmFeatures.size(); ++l) {
                    if (lmFeatures[l].size() > 1) {
                        //mitlm::Logger::Error(1, "Too many features specified.\n");
						LOGTW_ERROR << "Too many features specified.";
                        //exit(1);
						return -1;
                    }
                }
            } else {
                // Default CM features to log:sumhist:*.effcounts.
                lmFeatures.resize(1);
                lmFeatures[0].push_back("log:sumhist:%s.effcounts");
            }
            mode = mitlm::CountMerging;
        } else if (strcmp(interpolation, "GLI") == 0) {
            mode = mitlm::GeneralizedLinearInterpolation;
        } else {
            //mitlm::Logger::Error(1, "Unsupported interpolation mode %s.\n", interpolation);
			LOGTW_ERROR << "Unsupported interpolation mode " << interpolation;

            //exit(1);
			return -1;
        }

        // Load features.
        vector<vector<mitlm::FeatureVectors> > featureList(lms.size());
        for (size_t l = 0; l < lms.size(); ++l) {
            vector<string> &features = lmFeatures[lmFeatures.size()==1 ? 0 : l];
            featureList[l].resize(features.size());
            for (size_t f = 0; f < features.size(); ++f) {
                char feature[1024];
                sprintf(feature, features[f].c_str(), 
                        mitlm::GetBasename(corpusFiles[l]).c_str());
                //mitlm::Logger::Log(1, "Loading feature for %s from %s...\n", corpusFiles[l].c_str(), feature);
				LOGTW_INFO << "Loading feature for " << corpusFiles[l].c_str() << " from " << feature << "...";

                ilm.model().LoadComputedFeatures(featureList[l][f], feature, order);
            }
        }
        ilm.SetInterpolation(mode, featureList);
    }

    // Estimate LM.
    mitlm::ParamVector params(ilm.defParams());
    if (opts["params"]) {
        //mitlm::Logger::Log(1, "Loading parameters from %s...\n", opts["params"]);
		LOGTW_INFO << "Loading parameters from " << opts["params"] << "...";

        mitlm::ZFile f(opts["params"], "r");
        VerifyHeader(f, "Param");
        ReadVector(f, params);
        if (params.length() != ilm.defParams().length()) {
            //mitlm::Logger::Error(1, "Number of parameters mismatched.\n");
			LOGTW_ERROR << "Number of parameters mismatched.";
            //exit(1);
			return -1;
        }
    }

    mitlm::Optimization optAlg = mitlm::ToOptimization(opts["opt-alg"]);
    if (optAlg == mitlm::UnknownOptimization) {
        //mitlm::Logger::Error(1, "Unknown optimization algorithms '%s'.\n", opts["opt-alg"]);
		LOGTW_ERROR << "Unknown optimization algorithms '" << opts["opt-alg"] << "'.";

        //exit(1);
		return -1;
    }
        
    if (opts["opt-perp"]) {
        if (params.length() == 0) {
            //mitlm::Logger::Warn(1, "No parameters to optimize.\n");
			LOGTW_WARNING << "No parameters to optimize.";
        } else {
            //mitlm::Logger::Log(1, "Loading development set %s...\n", opts["opt-perp"]);
			LOGTW_INFO << "Loading development set " << opts["opt-perp"] << "...";

            mitlm::ZFile devZFile(opts["opt-perp"]);
            mitlm::PerplexityOptimizer dev(ilm, order);
            dev.LoadCorpus(devZFile);

            //mitlm::Logger::Log(1, "Optimizing %lu parameters...\n", params.length());
			LOGTW_INFO << "Optimizing " << params.length() << " parameters..." << "...";

            double optEntropy = dev.Optimize(params, optAlg);
            //mitlm::Logger::Log(2, " Best perplexity = %f\n", exp(optEntropy));
			LOGTW_INFO << " Best perplexity = " << exp(optEntropy);
        }
    }
    if (opts["opt-margin"]) {
        if (params.length() == 0) {
            //mitlm::Logger::Warn(1, "No parameters to optimize.\n");
			LOGTW_WARNING << "No parameters to optimize.";
        } else {
            //mitlm::Logger::Log(1, "Loading development lattices %s...\n", opts["opt-margin"]);
			LOGTW_INFO << "Loading development lattices " << opts["opt-margin"] << "...";

            mitlm::ZFile devZFile(opts["opt-margin"]);
            mitlm::WordErrorRateOptimizer dev(ilm, order);
            dev.LoadLattices(devZFile);

            //mitlm::Logger::Log(1, "Optimizing %lu parameters...\n", params.length());
			LOGTW_INFO << "Optimizing " << params.length() << " parameters..." << "...";

            double optMargin = dev.OptimizeMargin(params, optAlg);
            //mitlm::Logger::Log(2, " Best margin = %f\n", optMargin);
			LOGTW_INFO << " Best margin = " << optMargin;
        }
    }
    if (opts["opt-wer"]) {
        if (params.length() == 0) {
            //mitlm::Logger::Warn(1, "No parameters to optimize.\n");
			LOGTW_WARNING << "No parameters to optimize.";
        } else {
            //mitlm::Logger::Log(1, "Loading development lattices %s...\n", opts["opt-wer"]);
			LOGTW_INFO << "Loading development lattices " << opts["opt-wer"] << "...";
            mitlm::ZFile devZFile(opts["opt-wer"]);
            mitlm::WordErrorRateOptimizer dev(ilm, order);
            dev.LoadLattices(devZFile);

            //mitlm::Logger::Log(1, "Optimizing %lu parameters...\n", params.length());
			LOGTW_INFO << "Optimizing " << params.length() << " parameters..." << "...";
            double optWER = dev.OptimizeWER(params, optAlg);
            //mitlm::Logger::Log(2, " Best WER = %f%%\n", optWER);
			LOGTW_INFO << " Best WER = " << optWER;
        }
    }

    // Estimate full model.
    if (opts["write-lm"] || opts["eval-perp"] || 
        opts["eval-margin"] || opts["eval-wer"]) {
        //mitlm::Logger::Log(1, "Estimating full n-gram model...\n");
		LOGTW_INFO << "Estimating full n-gram model...";
        ilm.Estimate(params);
    }

    // Save results.
    if (opts["write-params"]) {
        //mitlm::Logger::Log(1, "Saving parameters to %s...\n", opts["write-params"]);
		LOGTW_INFO << "Saving parameters to " << opts["write-params"] << "...";
        mitlm::ZFile f(opts["write-params"], "w");
        WriteHeader(f, "Param");
        WriteVector(f, params);
    }
    if (opts["write-vocab"]) {
        //mitlm::Logger::Log(1, "Saving vocabulary to %s...\n", opts["write-vocab"]);
		LOGTW_INFO << "Saving vocabulary to " << opts["write-vocab"] << "...";
        mitlm::ZFile vocabZFile(opts["write-vocab"], "w");
        ilm.SaveVocab(vocabZFile);
    }
    if (opts["write-lm"]) {
        //mitlm::Logger::Log(1, "Saving LM to %s...\n", opts["write-lm"]);
		LOGTW_INFO << "Saving LM to " << opts["write-lm"]  << "...";
        mitlm::ZFile lmZFile(opts["write-lm"], "w");
        ilm.SaveLM(lmZFile, writeBinary);
    }

    // Evaluate LM.
    if (opts["eval-perp"]) {
        //mitlm::Logger::Log(0, "Perplexity Evaluations:\n");
		LOGTW_INFO << "Perplexity Evaluations:";
        vector<string> evalFiles;
        mitlm::trim_split(evalFiles, opts["eval-perp"], ',');
        for (size_t i = 0; i < evalFiles.size(); i++) {
            //mitlm::Logger::Log(1, "Loading eval set %s...\n", evalFiles[i].c_str());
			LOGTW_INFO << "Loading eval set " << evalFiles[i].c_str() << "...";
            mitlm::ZFile evalZFile(evalFiles[i].c_str());
            mitlm::PerplexityOptimizer eval(ilm, order);
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
            mitlm::WordErrorRateOptimizer eval(ilm, order);
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
            mitlm::WordErrorRateOptimizer eval(ilm, order);
            eval.LoadLattices(evalZFile);

            //mitlm::Logger::Log(0, "\t%s\t%.2f%%\n", evalFiles[i].c_str(), eval.ComputeWER(params));
			LOGTW_INFO << "\t" << evalFiles[i].c_str() << "\t" << eval.ComputeWER(params);
        }
    }

    return 0;
}
