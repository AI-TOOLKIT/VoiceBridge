/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2011 Microsoft Corporation, Apache 2.0.
*/
#include "kaldi-win\scr\kaldi_scr.h"

/*
This function splits up any kind of .scp or archive-type file. If there is no utt2spk option it will work on any text file and
will split it up with an approximately equal number of lines in each but. With the utt2spk option it will work on anything 
that has the utterance-id as the first entry on each line; the utt2spk file is of the form "utterance speaker" (on each line).
It splits it into equal size chunks as far as it can.  If you use the utt2spk option it will make sure these chunks 
coincide with speaker boundaries. In this case, if there are more chunks than speakers (and in some other circumstances), 
some of the resulting chunks will be empty and it will print an error message and exit with nonzero status.
Note: that you can use this script to split the utt2spk file itself,
*/

using MAPUNORDSS = std::unordered_map<std::string, std::string>;
using MAPUNORDSI = std::unordered_map<std::string, int>;
using MAPUNORDSST = std::unordered_map<std::string, StringTable>;
using MAPUNORDSSV = std::unordered_map<std::string, std::vector<std::string>>;

//NOTE: output_segments must be at least 1 file path!
//NOTE: if num_jobs and job_id are given it will only output the file with the job_id
int SplitScp(fs::path inscp, std::vector<fs::path> output_segments, int num_jobs, int job_id, fs::path utt2spk_file)
{
	if (output_segments.size()<1) {
		LOGTW_ERROR << " Output segments are not defined.";
		return -1;
	}

	if ( (num_jobs > 0 && job_id < 0) || job_id >= num_jobs) {
		LOGTW_ERROR << " invalid num-jobs= " << num_jobs << " and job-id= " << job_id << ".";
		return -1;
	}

	std::vector<fs::path> OUTPUTS;

	if (num_jobs == 0) { // without the job option
		OUTPUTS = output_segments;
	}
	else {
		for (int j = 0; j < num_jobs; j++) {
			if (j == job_id) {
				if (output_segments.size() > 0) { 
					OUTPUTS.push_back(output_segments[0]);
				}
				else 
				{
					LOGTW_ERROR << " you must define an output file.";
					return -1;
				}
			}
			else {
				OUTPUTS.push_back("NUL"); //NOTE: this will write to a non existing file on Windows (contents lost)
			}
		}
	}

	//open input scp file
	StringTable table_inscp;
	if (ReadStringTable(inscp.string(), table_inscp) < 0) {
		LOGTW_ERROR << " fail to open: " << inscp.string() << ".";
		return -1;
	}

	int numscps = OUTPUTS.size();  // number of output files.
	int numlines = table_inscp.size();
	if (numlines == 0) {
		LOGTW_ERROR << " [SplitScp] empty input scp file " << inscp.string() << ".";
		return -1;
	}

	if (utt2spk_file != "") 
	{ //split an utt2spk_file

		//TODO:... this part is not tested

		StringTable table_utt2spk;
		if (ReadStringTable(utt2spk_file.string(), table_utt2spk) < 0) {
			LOGTW_ERROR << " fail to open: " << utt2spk_file.string() << ".";
			return -1;
		}
		MAPUNORDSS utt2spk;
		for (StringTable::const_iterator it(table_utt2spk.begin()), it_end(table_utt2spk.end()); it != it_end; ++it) {
			if ((*it).size() != 2) {
				LOGTW_ERROR << " utt2spk has wrong format. 2 columns are expected in " << utt2spk_file.string() << ".";
				return -1;
			}
			utt2spk.emplace((*it)[0], (*it)[1]);
		}		
		std::vector<std::string> spkrs;
		MAPUNORDSI spk_count;
		MAPUNORDSST spk_data;
		for (StringTable::const_iterator it(table_inscp.begin()), it_end(table_inscp.end()); it != it_end; ++it) {
			if ((*it).size() < 1 || (*it)[0]=="") continue; //skip empty lines
			MAPUNORDSS::const_iterator iu = utt2spk.find((*it)[0]);
			if (iu == utt2spk.end()) {
				LOGTW_ERROR << " No such utterance " << (*it)[0] << " in utt2spk file " << utt2spk_file.string() << ".";
				return -1;
			}
			std::string s(iu->second);
			MAPUNORDSI::const_iterator ic = spk_count.find(s);
			if (ic == spk_count.end()) 
			{//not found
				spkrs.push_back(s);
				spk_count.emplace(s, 0);
				StringTable table; //empty table
				spk_data.emplace(s, table);
			}
			spk_count[s]++;
			(spk_data[s]).push_back( (*it) ); //NOTE: for each key we store a StringTable (rows of data)
		}

		// Now split as equally as possible. First allocate spks to files by allocating an approximately equal number of speakers.
		int numspks = spkrs.size();  // number of speakers.		
		if (numspks < numscps) {
			LOGTW_ERROR << " Refusing to split data because number of speakers " << numspks << " is less "
					  << "than the number of output .scp files " << numscps << ".";
			return -1;
		}
		std::vector<std::vector<std::string>> scparray;
		std::vector<int> scpcount;
		for (int scpidx = 0; scpidx < numscps; scpidx++) {
			std::vector<std::string> _sv; //empty string vector
			scparray.push_back(_sv);
			scpcount.push_back(0);
		}

		for (int spkidx = 0; spkidx < numspks; spkidx++) {
			int scpidx = int((spkidx*numscps) / numspks);
			std::string spk = spkrs[spkidx];
			scparray[scpidx].push_back(spk);
			scpcount[scpidx] += spk_count[spk];
		}

		/* Now will try to reassign beginning + ending speakers to different scp's and see if it gets more balanced. Suppose objf we're 
		   minimizing is sum_i (num utts in scp[i] - average)^2. We can show that if considering changing just 2 scp's, we minimize
		   this by minimizing the squared difference in sizes.  This is equivalent to minimizing the absolute difference in sizes.  This
		   shows this method is bound to converge.
		*/
		bool changed = true;
		while (changed) {
			changed = false;
			for (int scpidx = 0; scpidx < numscps; scpidx++) 
			{ // First try to reassign ending spk of this scp.
				if (scpidx < numscps - 1) {
					int sz = scparray[scpidx].size();
					if (sz > 0) {
						std::string spk = scparray[scpidx][sz - 1];
						int count = spk_count[ spk ];
						int nutt1 = scpcount[scpidx];
						int nutt2 = scpcount[scpidx + 1];
						if (std::abs((nutt2 + count) - (nutt1 - count)) < std::abs(nutt2 - nutt1))
						{// Would decrease size - diff by reassigning spk...
							scpcount[scpidx + 1] += count;
							scpcount[scpidx] -= count;							
							scparray[scpidx].pop_back();
							scparray[scpidx+1].insert(scparray[scpidx+1].begin(), spk);
							changed = true;
						}
					}
				}		
				if (scpidx > 0 && scparray[scpidx].size() > 0) {
					std::string spk = scparray[scpidx][0];
					int count = spk_count[spk];
					int nutt1 = scpcount[scpidx - 1];
					int nutt2 = scpcount[scpidx];
					if (std::abs((nutt2 - count) - (nutt1 + count)) < std::abs(nutt2 - nutt1))
					{// Would decrease size - diff by reassigning spk...
						scpcount[scpidx - 1] += count;
						scpcount[scpidx] -= count;
						scparray[scpidx].erase(scparray[scpidx].begin());
						scparray[scpidx - 1].push_back(spk);
						changed = true;
					}
				} //if
			} //for
		} //while
		
		// Now print out the files...
		for(int scpidx = 0; scpidx < numscps; scpidx++) {
			fs::path scpfn = OUTPUTS[scpidx];
			fs::ofstream file_(scpfn, std::ios::binary);
			if (!file_) {
				LOGTW_ERROR << "Can't open output file: " << scpfn.string() << ".";
				return -1;
			}
			int count = 0;
			if(scparray[scpidx].size() == 0) {
				LOGTW_ERROR << " [SplitScp] empty .scp file (too many splits and too few speakers?).";
				return -1;
			} else {
				for each (std::string spk in scparray[scpidx] ) {
					for (StringTable::const_iterator it(spk_data[spk].begin()), it_end(spk_data[spk].end()); it != it_end; ++it) {
						int c = 0;
						for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) 
						{
							if(c > 0) file_ << " ";
							file_ << *itc1;
							c++;
						}
						file_ << "\n";
					}
					//
					count += spk_count[spk];
				}
				if(count != scpcount[scpidx]) 
				{ 
					LOGTW_ERROR << " [SplitScp] Count mismatch (coding error).";
					return -1;
				}
			}
			file_.flush(); file_.close();
		}
	}
	else {
		// This block is the "normal" case where there is no utt2spk option and we just break into equal size chunks.
		int linesperscp = int( numlines / numscps); // the "whole part"..
		if (!(linesperscp >= 1)) {
			LOGTW_ERROR << " [SplitScp] You are splitting into too many pieces! [reduce number of jobs (nj)].";
			return -1;
		}
		int remainder = numlines - (linesperscp * numscps);
		if (!(remainder >= 0 && remainder < numlines)) {
			LOGTW_ERROR << " [SplitScp] bad remainder " << remainder << ".";
			return -1;
		}
		// [just doing int() rounds down].
		int n = 0;
		for(int scpidx = 0; scpidx < OUTPUTS.size(); scpidx++) {
			fs::path scpfile = OUTPUTS[scpidx];
			fs::ofstream file_(scpfile, std::ios::binary);
			if (!file_) {
				LOGTW_ERROR << "Can't open output file: " << scpfile.string() << ".";
				return -1;
			}
			for(int k = 0; k < linesperscp + (scpidx < remainder ? 1 : 0); k++) {
				int c = 0;
				for (string_vec::const_iterator itc1(table_inscp[n].begin()), itc1_end(table_inscp[n].end()); itc1 != itc1_end; ++itc1)
				{
					if (c > 0) file_ << " ";
					file_ << *itc1;
					c++;
				}
				file_ << "\n";	
				n++;
			}
			file_.flush(); file_.close();
		}
		if (!(n == numlines)) {
			LOGTW_ERROR << " [SplitScp] code error, n != numlines.";
			return -1;
		}
	} //else

	return 0;
}
