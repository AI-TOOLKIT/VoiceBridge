/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/
#include "kaldi-win\scr\kaldi_scr.h"

// Generate a topology file. This allows control of the number of states in the non - silence HMMs, and in the silence HMMs.

int GenerateTopology(int num_nonsil_states, int num_sil_states, StringTable nonsil_phones, StringTable sil_phones, fs::path path_topo_output)
{
	if ( !(num_nonsil_states >= 1 && num_nonsil_states <= 100) )
	{
		LOGTW_ERROR << " Unexpected number of nonsilence-model states: " << num_nonsil_states << ".";
		return -1;
	}
	if ( !( (num_sil_states == 1 || num_sil_states >= 3) && num_sil_states <= 100) )
	{
		LOGTW_ERROR << " Unexpected number of silence-model states: " <<  num_sil_states << ".";
		return -1;
	}

	//open output file
	//NOTE: ofstream write adds an '\r' in front of the '\n' automatically and some code in Kaldi crashes. To prevent this
	//		add std::ios::binary option to each ofstream to make sure that '\r' is not added!
	fs::ofstream file_output_txt(path_topo_output, std::ios::binary);
	if (!file_output_txt) {
		LOGTW_ERROR << " can't open output file: " << path_topo_output.string() << ".";
		return -1;
	}
	//NOTE: we make sure that even if there is space between the elements eveythigng comes to one line
	std::string _nonsil_phones, _sil_phones;
	int c = 0;
	for (int i = 0; i < nonsil_phones.size(); i++) {
		for (int j = 0; j < nonsil_phones[i].size(); j++) {
			if (c > 0) _nonsil_phones.append(" ");
			_nonsil_phones.append(nonsil_phones[i][j]);
			c++;
		}
	}
	c = 0;
	for (int i = 0; i < sil_phones.size(); i++) {
		for (int j = 0; j < sil_phones[i].size(); j++) {
			if (c > 0) _sil_phones.append(" ");
			_sil_phones.append(sil_phones[i][j]);
			c++;
		}
	}
	static const boost::regex rexp(":");
	_nonsil_phones = boost::regex_replace(_nonsil_phones, rexp, " ");
	_sil_phones = boost::regex_replace(_sil_phones, rexp, " ");

	file_output_txt << "<Topology>\n";
	file_output_txt << "<TopologyEntry>\n";
	file_output_txt << "<ForPhones>\n";
	file_output_txt << _nonsil_phones << "\n";
	file_output_txt << "</ForPhones>\n";
	for (int state = 0; state < num_nonsil_states; state++) {
		int statep1 = state + 1;
		file_output_txt << "<State> " << state << " <PdfClass> " << state << " <Transition> " << state << " 0.75 <Transition> " << statep1 << " 0.25 </State>\n";
	}
	file_output_txt << "<State> " << num_nonsil_states << " </State>\n"; // non - emitting final state.
	file_output_txt << "</TopologyEntry>\n";
	// Now silence phones.They have a different topology-- apart from the first and
	// last states, it's fully connected, as long as you have >= 3 states.

	if (num_sil_states > 1) {
		double transp = 1.0 / (num_sil_states - 1);
		file_output_txt << "<TopologyEntry>\n";
		file_output_txt << "<ForPhones>\n";
		file_output_txt << _sil_phones << "\n";
		file_output_txt << "</ForPhones>\n";
		file_output_txt << "<State> 0 <PdfClass> 0 ";
		for (int nextstate = 0; nextstate < num_sil_states - 1; nextstate++) {
			// Transitions to all but last emitting state.
			file_output_txt << "<Transition> " << nextstate << " " << transp << " ";
		}
		file_output_txt << "</State>\n";
		for (int state = 1; state < num_sil_states - 1; state++) {
			// the central states all have transitions to themselves and to the last emitting state.
				file_output_txt << "<State> " << state << " <PdfClass> " << state << " ";
			for (int nextstate = 1; nextstate < num_sil_states; nextstate++) {
				file_output_txt << "<Transition> " << nextstate << " " << transp << " ";
			}
			file_output_txt << "</State>\n";
		}
		// Final emitting state(non - skippable).
		int state = num_sil_states - 1;
		file_output_txt << "<State> " << state << " <PdfClass> " << state << " <Transition> " << state << " 0.75 <Transition> " << num_sil_states << " " << 0.25 << " </State>\n";
		// Final nonemitting state :
		file_output_txt << "<State> " << num_sil_states << " </State>\n";
		file_output_txt << "</TopologyEntry>\n";
	}
	else {
		file_output_txt << "<TopologyEntry>\n";
		file_output_txt << "<ForPhones>\n";
		file_output_txt << _sil_phones << "\n";
		file_output_txt << "</ForPhones>\n";
		file_output_txt << "<State> 0 <PdfClass> 0 ";
		file_output_txt << "<Transition> 0 0.75 ";
		file_output_txt << "<Transition> 1 0.25 ";
		file_output_txt << "</State>\n";
		file_output_txt << "<State> " << num_sil_states << " </State>\n"; // non - emitting final state.
		file_output_txt << "</TopologyEntry>\n";
	}

	file_output_txt << "</Topology>\n";

	file_output_txt.flush(); file_output_txt.close();
	return 0;
}
