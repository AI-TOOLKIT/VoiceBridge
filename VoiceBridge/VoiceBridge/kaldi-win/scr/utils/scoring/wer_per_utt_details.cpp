/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2015 Johns Hopkins University (Author: Yenda Trmal <jtrmal@gmail.com>), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"

static std::string rjustify(std::string str, int maxlen) {
	return str.insert(0, maxlen, ' ');
}
static std::string ljustify(std::string str, int maxlen) {
	return str.append(maxlen, ' ');
}

static std::string cjustify(std::string str, int maxlen) {
	int right_spaces = int((maxlen - str.length()) / 2);
	int left_spaces = maxlen - str.length() - right_spaces;
	rjustify(str, right_spaces);
	ljustify(str, left_spaces);
	return str;
}

/*
The program works as a filter -- reads the output from align-text program, parses it and outputs the requested entries 
on the output. The filter can be used (for example) to generate detailed statistics from scoring.
Input:
	UTT-A word-A word-A; <eps> word-A; word-B word-B; word-C <eps>; word-D word-D; word-E word-X;
	
	e.g.:
	"  key1 a a ; b <eps> ; c c \n"
	"  key2 d f ; e e \n"

Output:
	UTT-A ref  word-A   <eps>  word-B  word-C  word-D  word-E
	UTT-A hyp  word-A  word-A  word-B   <eps>  word-D  word-X
	UTT-A op      C       I       C       D       C       S
	UTT-A #csid 3 1 1 1
*/
int WerPerUttDetails(
	fs::path in,
	fs::path out,
	std::string special_symbol,
	std::string separator,
	bool output_hyp,
	bool output_ref,
	bool output_ops,
	bool output_csid
)
{
	StringTable t_in;
	if (ReadStringTable(in.string(), t_in, " " + separator) < 0) return -1;

	//prepare output file
	fs::ofstream f_out(out, fs::ofstream::binary | fs::ofstream::out);
	if (!f_out) {
		LOGTW_ERROR << "Could not write to file " << out.string();
		return -1;
	}

	using UMAPSI = std::unordered_map<std::string, int>;
	int c = 0;
	for (StringTable::const_iterator it(t_in.begin()), it_end(t_in.end()); it != it_end; ++it) {
		UMAPSI OPCOUNTS;
		OPCOUNTS.emplace("I", 0);
		OPCOUNTS.emplace("D", 0);
		OPCOUNTS.emplace("S", 0);
		OPCOUNTS.emplace("C", 0);
		string_vec HYP, REF, OP;
		c = 0;
		//check that there are uneven number of columns (1 ID and several pairs)
		if ((*it).size() % 2 == 0) {
			LOGTW_ERROR << "Wrong line in input file (WerPerUttDetails).";
			LOGTW_DEBUG << "More details: ";
			return -1;
		}
		std::string utt_id((*it)[0]);
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) {
			if (c == 0) { c++;  continue; } //NOTE: skip the first ID field
			std::string ref(*itc1);
			std::string hyp(*(itc1 + 1));

			//make sure that it is UTF8
			ref = ConvertToUTF8(ref);
			hyp = ConvertToUTF8(ref);

			REF.push_back(ref);
			HYP.push_back(hyp);
			if (hyp == special_symbol) {
				OP.push_back("D");
				OPCOUNTS["D"] += 1;
			} else if(ref == special_symbol) {
				OP.push_back("I");
				OPCOUNTS["I"] += 1;
			} else if(ref != hyp) {
				OP.push_back("S");
				OPCOUNTS["S"] += 1;
			}
			else {
				OP.push_back("C");
				OPCOUNTS["C"] += 1;
			}
			c++; ++itc1; //skip to next pair
		}///itc1

		if (OP.size() != HYP.size()) {
			LOGTW_ERROR << "Number of edit ops is not equal to the length of the text for utterance " << utt_id;
		}

		string_vec hyp_str;
		string_vec ref_str;
		string_vec op_str;
		for (int i = 0; i < OP.size(); i++) {
			int maxlen = __max(REF[i].length(), HYP[i].length());
			maxlen = __max(maxlen, OP[i].length());

			ref_str.push_back(cjustify(REF[i], maxlen));
			hyp_str.push_back(cjustify(HYP[i], maxlen));
			op_str.push_back(cjustify(OP[i], maxlen));
		}

		if (output_ref) LOGTW_INFO << utt_id << " ref  " << join_vector(ref_str, "  ");
		if (output_hyp) LOGTW_INFO << utt_id << " hyp  " << join_vector(hyp_str, "  ");
		if (output_ops) LOGTW_INFO << utt_id << " op   " << join_vector(op_str, "  ");
		if (output_csid) 
			LOGTW_INFO << utt_id << " #csid" << " " << OPCOUNTS["C"] << " " << OPCOUNTS["S"] << " " << OPCOUNTS["I"] << " " << OPCOUNTS["D"];

		//save to file
		if (output_ref) f_out << utt_id << " ref  " << join_vector(ref_str, "  ") << "\n";
		if (output_hyp) f_out << utt_id << " hyp  " << join_vector(hyp_str, "  ") << "\n";
		if (output_ops) f_out << utt_id << " op   " << join_vector(op_str, "  ") << "\n";
		if (output_csid)
			f_out << utt_id << " #csid" << " " << OPCOUNTS["C"] << " " << OPCOUNTS["S"] << " " << OPCOUNTS["I"] << " " << OPCOUNTS["D"] << "\n";

	}

	f_out.flush(); f_out.close();

	return 0;
}

