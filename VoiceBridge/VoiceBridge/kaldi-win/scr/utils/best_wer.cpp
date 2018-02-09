/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/
#include "kaldi-win\scr\kaldi_scr.h"

/*
	Selects the best WER; 
	_wer : a vector of file paths to Wer computation results from which to choose the best one.
	output goes to best_wip and best_lmwt
	input format 1 line per file (output from ComputeWer()):
	0	<< percent_wer << " "
	1	<< word_errs << " "
	2	<< num_words << " "
	3	<< num_ins << " "
	4	<< num_del << " "
	5	<< num_sub << " "
	6	<< percent_ser << " "
	7	<< sent_errs << " "
	8	<< num_sent << '\n';
*/
int BestWer(std::vector<fs::path> _wer, std::string & best_wip, int & best_lmwt)
{
	StringTable _tb;
	for (fs::path p : _wer)	{
		StringTable _t;
		int ret = ReadStringTable(p.string(), _t);
		double pser = StringToNumber<double>(_t[0][6], -1.0);
		double pwer = StringToNumber<double>(_t[0][0], -1.0);
		if (ret < 0 || _t[0].size() < 9 || pser < 0 || pwer < 0) {
			LOGTW_ERROR << "Could not read WER results from " << p.string();
			return -1;
		}
		std::vector<std::string> _v;
		_v.push_back(_t[0][6]);
		_v.push_back(_t[0][0]);
		_v.push_back(p.string());
		_tb.push_back(_v);
	}

	if (SortStringTable(_tb, 0, 1, "number", "number", false) < 0) {
		LOGTW_ERROR << "Could not determine best WER.";
		return -1;
	}

	//get the wip and lmwt from the file path; the first in the map is the best (lowest %)
	std::string p(_tb[0][2]);
	std::vector<std::string> _fp;
	boost::algorithm::trim(p);
	strtk::parse(p, "_", _fp, strtk::split_options::compress_delimiters);	
	if (_fp.size() < 3) {
		LOGTW_ERROR << "Could not determine best WER.";
		return -1;
	}

	//last is the WIP and the one before is the LMWT
	best_wip = _fp[_fp.size()-1];
	best_lmwt = StringToNumber<int>(_fp[_fp.size() - 2], -1);

	if (best_lmwt < 0) {
		LOGTW_ERROR << "Could not determine best WER.";
		return -1;
	}

	LOGTW_INFO << "Found the best WER/SER combination: WER=" << _tb[0][1] << "%, SER=" << _tb[0][0] << "%.";

	return 0;
}

