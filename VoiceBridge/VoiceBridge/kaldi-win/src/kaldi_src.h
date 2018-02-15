#pragma once
/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/

#pragma once
#include <kaldi-win/stdafx.h>
#include "kaldi-win/utility/Utility.h"
#include "kaldi-win/src/fstbin/fst_ext.h"

//featbin
int ComputeMFCCFeats(int argc, char *argv[], fs::ofstream & file_log);
int CopyFeats(int argc, char *argv[], fs::ofstream & file_log);
int PasteFeats(int argc, char *argv[], fs::ofstream & file_log);
int ExtractSegments(int argc, char *argv[], fs::ofstream & file_log);
int FeatToDim(int argc, char *argv[]);

int ComputeCmvnStats(int argc, char *argv[], fs::ofstream & file_log);
int ComputeCmvnStatsTwoChannel(int argc, char *argv[], fs::ofstream & file_log);
int ModifyCmvnStats(int argc, char *argv[], fs::ofstream & file_log);

int ApplyCmvn(int argc, char *argv[], fs::ofstream & file_log);
int AddDeltas(int argc, char *argv[]);
int SubsetFeats(int argc, char *argv[], fs::ofstream & file_log);
int SpliceFeats(int argc, char *argv[]);
int TransformFeats(int argc, char *argv[], fs::ofstream & file_log);

int ComposeTransforms(int argc, char *argv[], fs::ofstream & file_log);

int ProcessKaldiPitchFeats(int argc, char *argv[], fs::ofstream & file_log);
int ComputeKaldiPitchFeats(int argc, char *argv[], fs::ofstream & file_log);

//gmmbin
int GmmInfo(int argc, char *argv[]);
int GmmInitMono(int argc, char *argv[]);
int GmmEst(int argc, char *argv[], fs::ofstream & file_log);
int GmmAlignCompiled(int argc, char *argv[], fs::ofstream & file_log);
int GmmAccStatsAli(int argc, char *argv[], fs::ofstream & file_log);
int GmmBoostSilence(int argc, char *argv[], fs::ofstream & file_log);
int GmmSumAccs(int argc, char *argv[], fs::ofstream & file_log);
int GmmLatgenFaster(int argc, char *argv[], fs::ofstream & file_log);
int GmmMixup(int argc, char *argv[], fs::ofstream & file_log);
int GmmInitModel(int argc, char *argv[], fs::ofstream & file_log);
int GmmAccMllt(int argc, char *argv[], fs::ofstream & file_log);
int GmmTransformMeans(int argc, char *argv[], fs::ofstream & file_log);
int GmmEstFmllr(int argc, char *argv[], fs::ofstream & file_log);
int GmmAccStatsTwofeats(int argc, char *argv[], fs::ofstream & file_log);
int GmmPostToGpost(int argc, char *argv[], fs::ofstream & file_log);
int GmmEstFmllrGpost(int argc, char *argv[], fs::ofstream & file_log);
int GmmRescoreLattice(int argc, char *argv[], fs::ofstream & file_log);

//bin
int AlignEqualCompiled(int argc, char *argv[], fs::ofstream & file_log);
int CompileTrainGraphs(int argc, char *argv[], fs::ofstream & file_log);
int CopyMatrix(int argc, char *argv[], fs::ofstream & file_log);
int AliToPhones(int argc, char *argv[], fs::ofstream & file_log);
int ComputeWerBootci(int argc, char *argv[], fs::ofstream & file_log);
int ComputeWer(int argc, char *argv[], fs::ofstream & file_log);
int TreeInfo(int argc, char *argv[],
	int & numpdfs, int & context_width, int & central_position); //output
int MakeHTransducer(int argc, char *argv[]);
int AddSelfLoops(int argc, char *argv[]);
int AmInfo(std::string model_in_filename,
	int & nofphones,
	int & nofpdfs,
	int & noftransitionids,
	int & noftransitionstates
);
int AlignText(int argc, char *argv[]);
int AccTreeStats(int argc, char *argv[], fs::ofstream & file_log);
int SumTreeStats(int argc, char *argv[], fs::ofstream & file_log);
int BuildTree(int argc, char *argv[], fs::ofstream & file_log);
int ClusterPhones(int argc, char *argv[], fs::ofstream & file_log);
int CompileQuestions(int argc, char *argv[], fs::ofstream & file_log);
int ConvertAli(int argc, char *argv[], fs::ofstream & file_log);
int AccLda(int argc, char *argv[], fs::ofstream & file_log);
int EstLda(int argc, char *argv[], fs::ofstream & file_log);
int WeightSilencePost(int argc, char *argv[], fs::ofstream & file_log);
int AliToPost(int argc, char *argv[], fs::ofstream & file_log);
int EstMllt(int argc, char *argv[], fs::ofstream & file_log);

//latbin
int LatticeDepthPerFrame(int argc, char *argv[], fs::ofstream & file_log);
int LatticeBestPath(int argc, char *argv[], fs::ofstream & file_log);
int LatticeAddPenalty(int argc, char *argv[], fs::ofstream & file_log);
int LatticeMbrDecode(int argc, char *argv[], fs::ofstream & file_log);
int LatticePrune(int argc, char *argv[], fs::ofstream & file_log);
int LatticeScale(int argc, char *argv[], fs::ofstream & file_log);
int LatticeToPost(int argc, char *argv[], fs::ofstream & file_log);
int LatticeDeterminizePruned(int argc, char *argv[], fs::ofstream & file_log);
int Lattice1best(int argc, char *argv[], fs::ofstream & file_log);
int LatticeAlignWords(int argc, char *argv[], fs::ofstream & file_log);
int LatticeAlignWordsLexicon(int argc, char *argv[], fs::ofstream & file_log);
int LinearToNbest(int argc, char *argv[], fs::ofstream & file_log);
int NbestToProns(int argc, char *argv[], fs::ofstream & file_log);
