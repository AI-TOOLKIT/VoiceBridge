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

//@-zso removed new line from log

#include <string>
#include <vector>
#include <algorithm>
#include "util/FastIO.h"
#include "util/CommandOptions.h"
#include "Types.h"
#include "NgramModel.h"
#include "NgramLM.h"
#include "Smoothing.h"

using std::vector;

////////////////////////////////////////////////////////////////////////////////

namespace mitlm {

NgramLMBase::NgramLMBase(size_t order)
    : _pModel(new NgramModel(order)), _order(order),
      _probVectors(order + 1), _bowVectors(order + 1) {
}

void
NgramLMBase::LoadVocab(ZFile &vocabFile) {
    _pModel->LoadVocab(vocabFile);
}

void
NgramLMBase::SaveVocab(ZFile &vocabFile, bool asBinary) const {
    _pModel->SaveVocab(vocabFile, asBinary);
}

void
NgramLMBase::SaveLM(ZFile &lmFile, bool asBinary) const {
    if (asBinary) {
        WriteUInt64(lmFile, MITLMv1);
        Serialize(lmFile);
    } else
        _pModel->SaveLM(_probVectors, _bowVectors, lmFile);
}

void
NgramLMBase::Serialize(FILE *outFile) const {
    WriteHeader(outFile, "NgramLM");
    _pModel->Serialize(outFile);
    for (size_t o = 0; o <= order(); ++o)
        WriteVector(outFile, _probVectors[o]);
    for (size_t o = 0; o < order(); ++o)
        WriteVector(outFile, _bowVectors[o]);
}

void
NgramLMBase::Deserialize(FILE *inFile) {
    VerifyHeader(inFile, "NgramLM");
    _pModel->Deserialize(inFile);
    SetOrder(_pModel->size() - 1);
    for (size_t o = 0; o <= order(); ++o)
        ReadVector(inFile, _probVectors[o]);
    for (size_t o = 0; o < order(); ++o)
        ReadVector(inFile, _bowVectors[o]);
}

void
NgramLMBase::SetOrder(size_t order) {
    _pModel->SetOrder(order);
    _order = order;
    _probVectors.resize(order + 1);
    _bowVectors.resize(order);
}

Mask *
NgramLMBase::GetMask(vector<BitVector> &probMaskVectors,
                     vector<BitVector> &bowMaskVectors) const {
    return NULL;
}

bool
NgramLMBase::Estimate(const ParamVector &params, Mask *pMask) {
    return true;
}

void
NgramLMBase::SetModel(const SharedPtr<NgramModel> &m,
                      const VocabVector &vocabMap,
                      const vector<IndexVector> &ngramMap) {
    for (size_t o = 1; o <= _order; ++o) {
        size_t len = m->sizes(o);
        NgramModel::ApplySort(ngramMap[o], _probVectors[o], len, (Prob)0);
        if (o < _order)
            NgramModel::ApplySort(ngramMap[o], _bowVectors[o], len, (Prob)1);
    }
    _pModel = m;

    // Fill in missing probabilities with backoff values.
    for (size_t o = 1; o <= _order; ++o) {
        IndexVector        hists(this->hists(o));
        const IndexVector &backoffs(this->backoffs(o));
        const ProbVector & boProbs(_probVectors[o-1]);
        const ProbVector & bows(_bowVectors[o-1]);
        ProbVector &       probs(_probVectors[o]);
        MaskAssign(probs == 0, boProbs[backoffs] * bows[hists], probs);
        assert(!anyTrue(isnan(probs)));
        assert(!anyTrue(isnan(bows)));
    }
}

////////////////////////////////////////////////////////////////////////////////

void
ArpaNgramLM::LoadLM(ZFile &lmFile) {
    if (ReadUInt64(lmFile) == MITLMv1) {
        Deserialize(lmFile);
    } else {
        lmFile.ReOpen();
        _pModel->LoadLM(_probVectors, _bowVectors, lmFile);
    }
}

////////////////////////////////////////////////////////////////////////////////

void
NgramLM::Initialize(const char *vocab, bool useUnknown,
                    const char *text, const char *counts,
                    const char *smoothingDesc, const char *featureDesc) {
    // Read vocabulary.
    if (useUnknown) {
        Logger::Log(1, "Replace unknown words with <unk>...");
        UseUnknown();
    }
    if (vocab) {
        Logger::Log(1, "Loading vocab %s...", vocab);
        ZFile vocabZFile(vocab);
        LoadVocab(vocabZFile);
    }
    
    // Read language model input files.
    string corpusFile;
    if (text) {
        vector<string> textFiles;
        trim_split(textFiles, text, ',');
        for (size_t i = 0; i < textFiles.size(); i++) {
            Logger::Log(1, "Loading corpus %s...", textFiles[i].c_str());
            ZFile corpusZFile(ZFile(textFiles[i].c_str()));
            LoadCorpus(corpusZFile);
            if (corpusFile.length() == 0) corpusFile = textFiles[i].c_str();
        }
    }
    if (counts) {
        vector<string> countsFiles;
        trim_split(countsFiles, counts, ',');
        for (size_t i = 0; i < countsFiles.size(); i++) {
            Logger::Log(1, "Loading counts %s...", countsFiles[i].c_str());
            ZFile countsZFile(ZFile(countsFiles[i].c_str()));
            LoadCounts(countsZFile);
            if (corpusFile.length() == 0) corpusFile = countsFiles[i].c_str();
        }
    }

    // Process n-gram weighting features.
    if (featureDesc) {
        vector<string> feats;
        trim_split(feats, featureDesc, ',');
        vector<vector<DoubleVector> > featureList(feats.size());
        for (size_t f = 0; f < feats.size(); ++f) {
            char feature[1024];
            sprintf(feature, feats[f].c_str(), GetBasename(corpusFile).c_str());
            Logger::Log(1, "Loading weight features %s...", feature);
            model().LoadComputedFeatures(featureList[f], feature);
        }
        SetWeighting(featureList);
    }
    
    // Set smoothing algorithms.
    vector<string> smoothings;
    trim_split(smoothings, smoothingDesc, ',');
    if (smoothings.size() != 1 && smoothings.size() != order()) {
        Logger::Error(1, "Inconsistent number of smoothing algorithms.");
	throw std::runtime_error("Inconsistent number of smoothing algorithms.");
    }
    vector<SharedPtr<Smoothing> > smoothingAlgs(order() + 1);
    for (size_t o = 1; o <= order(); o++) {
        const char *smoothing = smoothings.size() == 1 ?
            smoothings[0].c_str() : smoothings[o-1].c_str();
        Logger::Log(1, "Smoothing[%i] = %s", o, smoothing);
        smoothingAlgs[o] = Smoothing::Create(smoothing);
        if (smoothingAlgs[o].get() == NULL) {
            Logger::Error(1, "Unknown smoothing %s.", smoothing);
	    throw std::runtime_error("Unknown smoothing algorithm.");
        }
    }
    Logger::Log(1, "Set smoothing algorithms...");
    SetSmoothingAlgs(smoothingAlgs);
}

void
NgramLM::LoadCorpus(ZFile &corpusFile, bool reset) {
    _pModel->LoadCorpus(_countVectors, corpusFile, reset);
}

void
NgramLM::LoadCounts(ZFile &countsFile, bool reset) {
    if (ReadUInt64(countsFile) == MITLMv1) {
        if (!reset)
            throw std::runtime_error("Not implemented yet.");
        VerifyHeader(countsFile, "NgramCounts");
        _pModel->Deserialize(countsFile);
        SetOrder(_pModel->size() - 1);
        for (size_t o = 0; o <= order(); ++o)
            ReadVector(countsFile, _countVectors[o]);
    } else {
        countsFile.ReOpen();
        _pModel->LoadCounts(_countVectors, countsFile, reset);
    }
}

void
NgramLM::SaveCounts(ZFile &countsFile, bool asBinary) const {
    if (asBinary) {
        WriteUInt64(countsFile, MITLMv1);
        WriteHeader(countsFile, "NgramCounts");
        _pModel->Serialize(countsFile);
        for (size_t o = 0; o <= order(); ++o)
            WriteVector(countsFile, _countVectors[o]);
    } else {
        _pModel->SaveCounts(_countVectors, countsFile);
    }
}

void
NgramLM::SaveEffCounts(ZFile &countsFile, bool asBinary) const {
    vector<CountVector> effCountVectors(_order + 1);
    for (size_t o = 1; o <= _order; ++o) {
        effCountVectors[o].reset(sizes(o), 0);
        const Smoothing *smoothing = (const Smoothing *)_smoothings[o];
        effCountVectors[o].attach(smoothing->effCounts());
    }
    if (asBinary) {
        WriteUInt64(countsFile, MITLMv1);
        WriteHeader(countsFile, "NgramCounts");
        _pModel->Serialize(countsFile);
        for (size_t o = 0; o <= order(); ++o)
            WriteVector(countsFile, effCountVectors[o]);
    } else {
        _pModel->SaveCounts(effCountVectors, countsFile);
    }
}

void
NgramLM::SetSmoothingAlgs(const vector<SharedPtr<Smoothing> > &smoothings) {
    assert(smoothings.size() == _order + 1);
    _smoothings = smoothings;
    for (size_t o = 1; o <= _order; ++o) {
        assert(_smoothings[o]);
        _smoothings[o]->Initialize(this, o);
    }

    // Allocate and initialize variables.
    for (size_t o = 0; o < _order; ++o) {
        size_t len = _pModel->sizes(o);
        _probVectors[o].reset(len);
        _bowVectors[o].reset(len);
    }
    _probVectors[_order].reset(_pModel->sizes(_order));

    // Compute 0th order probability.
    if (vocab().IsFixedVocab())
        _probVectors[0][0] = Prob(1.0 / sizes(1));
    else
        _probVectors[0][0] = Prob(1.0 / sum(_countVectors[1] > 0));

    // Compute default parameters.
    _paramStarts.reset(_order + 2);
    VectorBuilder<Param> builder;
    for (size_t o = 1; o <= _order; ++o) {
        _paramStarts[o] = (int)builder.length(); //@+zso (int)
        builder.append(_smoothings[o]->defParams());
    }
    _paramStarts[_order + 1] = (int)builder.length(); //@+zso (int)
    _defParams = builder;
}

void
NgramLM::SetWeighting(const vector<FeatureVectors> &featureList) {
    // NOTE: Remap featureList[f][o] to _featureList[o][f].
    if (featureList.size() > 0) _featureList.resize(featureList[0].size());
    for (size_t o = 0; o < _featureList.size(); ++o) {
        _featureList[o].resize(featureList.size());
        for (size_t f = 0; f < featureList.size(); ++f) {
            assert(featureList[f].size() == _featureList.size());
            _featureList[o][f].attach(featureList[f][o]);
        }
    }
}

void
NgramLM::SetOrder(size_t order) {
    NgramLMBase::SetOrder(order);
    _countVectors.resize(order + 1);
    _featureList.resize(order + 1);
}

Mask *
NgramLM::GetMask(vector<BitVector> &probMaskVectors,
                 vector<BitVector> &bowMaskVectors) const {
    // Copy prob and bow masks.
    NgramLMMask *pMask = new NgramLMMask();
    pMask->ProbMaskVectors = probMaskVectors;
    pMask->BowMaskVectors  = bowMaskVectors;

    // Let each smoothing algorithm update filter (in reverse).
    pMask->SmoothingMasks.resize(_order + 1);
    for (size_t o = _order; o > 0; o--)
        _smoothings[o]->UpdateMask(*pMask);
    return pMask;
}

bool
NgramLM::Estimate(const ParamVector &params, Mask *pMask) {
    NgramLMMask *pNgramLMMask = (NgramLMMask *)pMask;
    for (size_t o = 1; o <= _order; o++) {
        Range r(_paramStarts[o], _paramStarts[o+1]);
        if (!_smoothings[o]->Estimate(params[r], pNgramLMMask,
                                      _probVectors[o], _bowVectors[o-1]))
            return false;
    }
    return true;
}

void
NgramLM::SetModel(const SharedPtr<NgramModel> &m,
                  const VocabVector &vocabMap,
                  const vector<IndexVector> &ngramMap) {
    _pModel = m;
    for (size_t o = 1; o <= _order; ++o) {
        size_t len = m->sizes(o);
        NgramModel::ApplySort(ngramMap[o], _countVectors[o], len, 0);
        for (size_t f = 0; f < _featureList[o].size(); ++f)
            NgramModel::ApplySort(ngramMap[o], _featureList[o][f], len, 0.0);
    }
    SetSmoothingAlgs(_smoothings);
}

}
