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
// Copyright (c) 2013, Giulio Paci <giuliopaci@gmail.com>                 //
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

#include <cassert>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include "util/FastIO.h"
#include "util/constants.h"
#include "Vocab.h"

namespace mitlm {

////////////////////////////////////////////////////////////////////////////////

struct VocabIndexCompare
{
    const Vocab &_vocab;
    VocabIndexCompare(const Vocab &vocab) : _vocab(vocab) { }
    bool operator()(int i, int j) { return strcmp(_vocab[i], _vocab[j]) < 0; }
};

////////////////////////////////////////////////////////////////////////////////

const VocabIndex Vocab::Invalid         = (VocabIndex)-1;
const VocabIndex Vocab::EndOfSentence   = (VocabIndex)0;

////////////////////////////////////////////////////////////////////////////////

// Create Vocab with specified capacity.
Vocab::Vocab(size_t capacity) : _length(0), _fixedVocab(false),
                                _unkIndex(Invalid) {
    Reserve(capacity);
    Add("</s>");
}

void
Vocab::SetFixedVocab(bool fixedVocab) {
    _fixedVocab = fixedVocab;
}

void
Vocab::UseUnknown() {
    assert(!_fixedVocab);  // Call UseUnknown() before SetReadOnly().
    if (_unkIndex == Invalid) {
        _unkIndex = Add("<unk>");
        assert(_unkIndex == 1);
    }
}

// Return associated index of the word, or Invalid if not found.
// In case of collision, apply quadratic probing.
VocabIndex
Vocab::Find(const char *word, size_t len) const {
    if (len == 3 && strcmp(word, "<s>") == 0)
        return EndOfSentence;

    size_t     skip = 0;
    VocabIndex pos  = (int)(StringHash(word, len) & _hashMask); //@+zso (int)
    VocabIndex index;
    while ((index = _indices[pos]) != Invalid &&
           !(wordlen(index)==len && strncmp(operator[](index), word, len)==0)) {
        pos = (int)((pos + ++skip) & _hashMask); //@+zso (int)
    }
    return (index == Invalid) ? _unkIndex : index;
}

// Add word to the vocab and return the associated index.
// If word already exists, return the existing index.
VocabIndex
Vocab::Add(const char *word, size_t len) {
    if (len == 3 && strcmp(word, "<s>") == 0)
        return EndOfSentence;

    VocabIndex *pIndex = _FindIndex(word, len);
    if (*pIndex == Invalid && !_fixedVocab) {
        // Increase index table size as needed.
        if (size() >= _offsetLens.length()) {
            Reserve(std::max((size_t)1<<16, _offsetLens.length()*2));
            pIndex = _FindIndex(word, len);
        }
        *pIndex = (int)_length;  //@+zso (int)
        _offsetLens[_length++] = OffsetLen((mitlm::uint)_buffer.size(), (mitlm::uint)len); //@+zso (mitlm::uint)
        _buffer.append(word, len + 1);        // Include terminating NULL.
    }
    return (*pIndex == Invalid) ? _unkIndex : *pIndex;
}

void
Vocab::Reserve(size_t capacity) {
    // Reserve index table and value vector with specified capacity.
    if (capacity != _offsetLens.length()) {
        _Reindex(nextPowerOf2((unsigned long)(capacity + capacity/4))); //@+zso (unsigned long)
        _offsetLens.resize(capacity);
    }
}

// Sort the vocabulary and output the mapping from original to new index.
bool
Vocab::Sort(VocabVector &sortMap) {
    // Sort indices using vocab index comparison function.
    // - Skip the first two words: </s> (and optionally <unk>).
    int               numFixedWords = (_unkIndex == Invalid) ? 1 : 2;
    VocabIndexCompare compare(*this);
    VocabVector       sortIndices = Range(size());
    if (!sortIndices[Range(numFixedWords, size())].sort(compare)) {
        sortMap = Range(size());
        return false;
    }

    // Build new string buffer for the sorted words.
    // Change offsets to refer to new string buffer.
    // Build sort mapping that maps old to new indices.
    std::string     newBuffer;
    OffsetLenVector newOffsetLens(size());
    newBuffer.reserve(_buffer.size());
    sortMap.reset(size());
    for (VocabIndex i = 0; i < (VocabIndex)size(); ++i) {
        const OffsetLen &offsetLen = _offsetLens[sortIndices[i]];
        newOffsetLens[i] = OffsetLen((mitlm::uint)newBuffer.length(), offsetLen.Len);  //@+zso (mitlm::uint)
        newBuffer.append(&_buffer[offsetLen.Offset], offsetLen.Len + 1);
        sortMap[sortIndices[i]] = i;
    }
    _buffer.swap(newBuffer);
    _offsetLens.swap(newOffsetLens);

    // Rebuild index map by applying sortMap.
    MaskAssign(_indices != Invalid, sortMap[_indices], _indices);

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// Loads vocabulary from file where each word appears on a non-# line.
void
Vocab::LoadVocab(ZFile &vocabFile) {
    uint64_t v = !MITLMv1;
    try {
        v = ReadUInt64(vocabFile);
    } catch(std::runtime_error e) {
    }
    if (v == MITLMv1) {
        Deserialize(vocabFile);
    } else {
        vocabFile.ReOpen();
        char   line[mitlm::kMaxLineLength];
        int len = 0;
        while (!feof(vocabFile)) {
            getline(vocabFile, line, mitlm::kMaxLineLength, &len);
            if (len > 0 && line[0] != '#')
                Add(line, len);
        }
    }
}

// Saves vocabulary to file with each word on its own line.
void
Vocab::SaveVocab(ZFile &vocabFile, bool asBinary) const {
    if (asBinary) {
        WriteUInt64(vocabFile, MITLMv1);
        Serialize(vocabFile);
    } else {
        for (const OffsetLen *p = _offsetLens.begin();
             p !=_offsetLens.begin() + _length; ++p) {
            fputs(&_buffer[p->Offset], vocabFile);
            fputc('\n', vocabFile);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void
Vocab::Serialize(FILE *outFile) const {
    WriteHeader(outFile, "Vocab");
    WriteString(outFile, _buffer);
}

void
Vocab::Deserialize(FILE *inFile) {
    VerifyHeader(inFile, "Vocab");
    ReadString(inFile, _buffer);

    // Count the number of words.
    _length = 0;
    for (size_t i = 0; i < _buffer.capacity(); ++i)
        if (_buffer[i] == '\0') ++_length;
    _offsetLens.resize(_length);

    // Rebuild _offsetLens and _indices.
	mitlm::uint offset = 0; //@+zso mitlm::uint
    _length = 0;
    for (mitlm::uint i = 0; i < (mitlm::uint)_buffer.capacity(); ++i) { //@+zso (mitlm::uint) x2
        if (_buffer[i] == '\0') {
            _offsetLens[_length++] = OffsetLen(offset, i - offset); 
            offset = i + 1;
        }
    }
    _Reindex(nextPowerOf2((unsigned long)(_length + _length/4))); //@+zso (unsigned long)
}

////////////////////////////////////////////////////////////////////////////////

// Return the iterator to the position of the word.
// If word is not found, return the position to insert the word.
// In case of collision, apply quadratic probing.
// NOTE: This function assumes the index table is not full.
VocabIndex *
Vocab::_FindIndex(const char *word, size_t len) {
    size_t     skip = 0;
    VocabIndex pos  = (int)StringHash(word, len) & _hashMask; //@+zso (int)
    VocabIndex index;
    while ((index = _indices[pos]) != Invalid &&
           !(wordlen(index)==len && strncmp(operator[](index), word, len)==0)) {
        pos = (int)((pos + ++skip) & _hashMask); //@+zso (int)
    }
    return &_indices[pos];
}

// Resize index table to the specified capacity.
void
Vocab::_Reindex(size_t indexSize) {
    assert(indexSize > size() && isPowerOf2((unsigned long)indexSize)); //@+zso (unsigned long)

    _indices.reset(indexSize, Invalid);
    _hashMask = indexSize - 1;
    OffsetLen *p = _offsetLens.begin();
    for (VocabIndex i = 0; i < (VocabIndex)size(); ++i, ++p) {
        size_t     skip = 0;
        VocabIndex pos  = (int)(StringHash(&_buffer[p->Offset], p->Len) & _hashMask); //@+zso (int)
        while (_indices[pos] != Invalid)
            pos = (int)((pos + ++skip) & _hashMask); //@+zso (int)
        _indices[pos] = i;
    }
}

}
