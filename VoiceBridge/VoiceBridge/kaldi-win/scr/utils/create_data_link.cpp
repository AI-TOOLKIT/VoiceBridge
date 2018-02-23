/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : 
Copyright 2013  Guoguo Chen
          2014  Johns Hopkins University (author: Daniel Povey)
		  Apache 2.0.

	 NOTE: not using symbolic links because it needs administrator privilege in Windows!
		   Using a map instead where fullpath is the unique key.

	 On Unix systems this function distributes data onto different file systems by making symbolic
	 links. It is supposed to use together with utils/create_split_dir.pl, which
	 creates a "storage" directory that links to different file systems.
	 If a sub-directory egs/storage does not exist, it does nothing. If it exists,
	 then it selects pseudo-randomly a number from those available in egs/storage/*
	 creates a link such as egs/egs.3.4.ark -> storage/4/egs.3.4.ark
*/
#include "kaldi-win\scr\kaldi_scr.h"

int GetGCD(int a, int b) {
	while (a != b) {
		if (a > b) {
			a = a - b;
		}
		else {
			b = b - a;
		}
	}
	return a;
}

//_fullpaths : paths to an ark file name
int CreateDataLink(std::vector<fs::path> _fullpaths, MSYMLINK & msymlinkstorage)
{
	if (_fullpaths.size() < 1) return 0;
	// Check if the storage has been created. If so, do nothing.
	fs::path storage_dir(_fullpaths[0].parent_path() / "storage");
	if (!fs::exists(storage_dir)) {
		//LOGTW_INFO << " 'storage' directory does not exist in " << example_fullpath.parent_path().string() << ". Parallel processing will not be applied.";
		return 0;
	}

	// First, get a list of the available storage directories, and check if they are properly created.
	std::string srex = "^[0-9]*$";
	std::vector<fs::path> storage_dirs = GetDirectories(storage_dir, srex);
	int num_storage = (int)storage_dirs.size();
	//check if the correct subdirs exist
	for (int x = 1; x <= num_storage; x++) {
		if (!fs::exists(storage_dir / std::to_string(x))) {
			LOGTW_ERROR << " " << (storage_dir / std::to_string(x)).string() << " directory does not exist.";
			return -1;
		}
	}

	//get coprimes
	std::vector<int> coprimes;
	//NOTE: possible BUG is corrected; n < num_storage was replaced by n <= num_storage TODO:... test
	for (int n = 1; n <= num_storage; n++) {
		if (GetGCD(n, num_storage) == 1) {
			coprimes.push_back(n);
		}
	}
	std::string sbasedir(_fullpaths[0].parent_path().string());
	for each (fs::path fullpath in _fullpaths) {
		if (fullpath.parent_path().string() != sbasedir) {
			LOGTW_ERROR << " Mismatch in directory names of arguments: " << sbasedir << " versus " << fullpath.parent_path().string() << ".";
			return -1;
		}

		//Finally, work out the directory index where we should put the data to.
		std::string basename(fullpath.filename().string());
		static const boost::regex rexp("[^0-9]+");
		std::string sfilename_numbers = boost::regex_replace(basename, rexp, " ");
		boost::algorithm::trim(sfilename_numbers);
		std::vector<int> filename_numbers;
		strtk::parse(sfilename_numbers, " ", filename_numbers, strtk::split_options::compress_delimiters);
		int total = 0;
		int index = 0;
		for each(int x in filename_numbers) {
			if (index >= (int)coprimes.size()) {
				index = 0;
			}
			total += x * coprimes[index];
			index++;
		}

		int dir_index = total % num_storage + 1;

		//Make the link for later use
		//NOTE: not using symbolic links because it needs administrator privilege in Windows!
		//		using a map instead where fullpath is the unique key
		if (fs::exists(fullpath)) {
			//unlink
			MSYMLINK::const_iterator ci = msymlinkstorage.find(fullpath.string());
			if (ci != msymlinkstorage.end()) msymlinkstorage.erase(ci);
		}
		//make the link
		msymlinkstorage.emplace(fullpath.string(), (storage_dir / std::to_string(dir_index) / basename).string());
	}
	 
	return 0;
}

