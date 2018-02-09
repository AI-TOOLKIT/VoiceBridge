/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :	Copyright 2010-2012   Microsoft Corporation, Apache 2
            2012-2016   Johns Hopkins University (author: Daniel Povey)
            2015   Xiaohui Zhang
*/

#include "kaldi-win\scr\kaldi_scr.h"

/*
This function takes multiple lists of utterance-ids or any file whose first field of each line is an utterance-id, 
as filters, and filters an scp file (or any file whose "n-th" field is an utterance id), printing
out only those lines whose "n-th" field is in filter. The index of the "n-th" field is 1, by default, but can be changed 
by using the -f <n> switch
Similar to FilterScp, but it uses multiple filters and output multiple filtered files.

field : zero based index of the fields/columns
jobstart, jobend : 1 based index
Expects jobname ="JOBID" which will be replaced by the real ID
*/
int FilterScps(int jobstart, int jobend, fs::path idlist, fs::path in_scp, fs::path out_scp, bool no_warn, int field)
{
	std::string jobname("JOBID");
	if (jobstart > jobend) {
		LOGTW_ERROR << " Internal error, invalid jobid range.";
		return -1;
	}
	if (!ContainsString(idlist.string(), jobname) && jobend > jobstart)	{
		LOGTW_ERROR << " Internal error, you are trying to use multiple filter files as filter patterns but you are providing just one filter file.";
		return -1;
	}
	if (!ContainsString(out_scp.string(), jobname) && jobend > jobstart) {
		LOGTW_ERROR << " Internal error, you are trying to use multiple filter files as filter patterns but you are providing just one output file.";
		return -1;
	}

	/*
	id2jobs hashes from the id (e.g. utterance-id) to an array of the relevant job-ids (which are integers). 
	In any normal use-case, this array will contain exactly one job-id for any given id, but we want to be agnostic about this.
	*/
	using MAPSIV = std::unordered_map<std::string, std::vector<int>>;
	MAPSIV id2jobs;
	/*
	job2output hashes from the job-id, to an anonymous array containing a sequence of output lines.
	*/
	using MAPISV = std::unordered_map<int, std::vector<std::string>>;
	MAPISV job2output;
	//
	bool warn_uncovered = false;
	bool warn_multiply_covered = false;
	for (int jobid = jobstart; jobid <= jobend; jobid++) 
	{
		std::string idlist_n = idlist.string();
		ReplaceStringInPlace(idlist_n, jobname, std::to_string(jobid));
		StringTable t_idlistn;
		if (ReadStringTable(idlist_n, t_idlistn) < 0) return -1;
		for (StringTable::const_iterator it(t_idlistn.begin()), it_end(t_idlistn.end()); it != it_end; ++it)
		{
			if ((*it).size() < 1) {
				LOGTW_ERROR << " Invalid line in " << idlist_n << ".";
				return -1;
			}
			std::string id((*it)[0]);
			MAPSIV::iterator itm = id2jobs.find(id);
			if (itm == id2jobs.end()) {
				std::vector<int> _id;
				_id.push_back(jobid);
				id2jobs.emplace(id, _id);
			}
			else {
				itm->second.push_back(jobid);
			}
		}

		//job2output
		std::vector<std::string> _l;
		job2output.emplace(jobid, _l);
	}

	int line = 1;
	StringTable table_in_scp;
	if (ReadStringTable(in_scp.string(), table_in_scp) < 0) return -1;
	for (StringTable::const_iterator it(table_in_scp.begin()), it_end(table_in_scp.end()); it != it_end; ++it)
	{
		if (!((*it).size() >= field+1)) {
			LOGTW_ERROR << " Invalid input scp data in " << in_scp.string() << " at line " << line << ".";
			return -1;
		}

		std::string id((*it)[field]);
		MAPSIV::iterator itm = id2jobs.find(id);
		if (itm == id2jobs.end()) {
			warn_uncovered = true;
		}
		else {
			std::vector<int> jobs = itm->second;
			if(jobs.size()>1) warn_multiply_covered = true;
			for each(int job_id in jobs) {
				MAPISV::iterator itmj = job2output.find(job_id);
				if (itmj == job2output.end()) {
					LOGTW_ERROR << " Unexpected internal error.";
					return -1;
				}
				else {
					std::string sline;
					int c = 0;
					for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) 
					{
						if (c > 0) sline.append(" ");
						sline.append(*itc1);
						c++;
					}
					itmj->second.push_back(sline);
				}
			}///for
		}///else
		line++;
	}

	for (int jobid = jobstart; jobid <= jobend; jobid++)
	{
		std::string outfile_n = out_scp.string();
		ReplaceStringInPlace(outfile_n, jobname, std::to_string(jobid));
		fs::ofstream file_out_scp(outfile_n, std::ios::binary);
		if (!file_out_scp) {
			LOGTW_ERROR << " Can't open output file: " << outfile_n << ".";
			return -1;
		}
		bool printed = false;
		MAPISV::iterator itmj = job2output.find(jobid);
		if (itmj == job2output.end()) {
			LOGTW_WARNING << " output to " << outfile_n << " is empty.";
		}
		else {
			for each(std::string sline in itmj->second)
			{
				file_out_scp << sline << "\n";
				printed = true;
			}
			if (!printed) {
				LOGTW_WARNING << " output to " << outfile_n << " is empty.";
			}
		}
	}

	if (warn_uncovered && !no_warn) {
		LOGTW_WARNING << " some input lines did not get output.";
	}
	if (warn_multiply_covered && !no_warn) {
		LOGTW_WARNING << " some input lines were output to multiple files (OK if splitting per utt).";
	}

	return 0;
}

