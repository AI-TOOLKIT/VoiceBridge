/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2016  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"

//NOTE: using an automatically ordered map! It also contains only unique keys
using INT2TXT_MAP = std::map<int, std::string>;
using MAPSS = std::map<std::string, std::string>;
using MAPSI = std::map<std::string, int>;
using LD_MAP = std::map<int, std::map<int, int>>;
using LD_MAPF = std::multimap<std::map<int, int>, int>;
using LD_MAP1 = std::map<int, int>; //one level deeper in LD_MAP

/*
	If depth_to_count is a map from depth-in-frames to count, return the depth-in-frames that equals the (fraction * 100)'th
	 percentile of the distribution.
*/
static int GetPercentile(const LD_MAP1 & depth_to_count, double fraction)
{ //NOTE: depth_to_count is automatically sorted!
	int this_total_frames = 0;
	for (const auto& pair2 : depth_to_count)
		this_total_frames += pair2.second;

	if (this_total_frames == 0)
		return 0;
	else {
		int count_cutoff = (int)(fraction * this_total_frames);
		int cur_count_total = 0;
		for (const auto& pair2 : depth_to_count)
		{
			int depth = pair2.first;
			int count = pair2.second;
			assert(pair2.second >= 0);
			cur_count_total += count;
			if (cur_count_total >= count_cutoff) {
				return depth;
			}
		}
		assert(false); //we shouldn't reach here.
		return 0;
	}
}


static double GetMean(const LD_MAP1 & depth_to_count)
{
	int this_total_frames = 0, this_total_depth = 0;
	for (const auto& pair2 : depth_to_count)
		this_total_frames += pair2.second;
	if (this_total_frames == 0)
		return 0;
	for (const auto& pair2 : depth_to_count)
		this_total_depth += (pair2.first * pair2.second);
	return 1.0 * this_total_depth / this_total_frames;
}

/*
	This function reads stats created in analyze_lats to print information about lattice depths broken down per phone.
*/
int AnalyzeLatticeDepthStats(
	std::vector<fs::path> & _latticestats,	//input lattice stats files
	fs::path lang,							//lang directory	
	float frequencyCutoffPercentage			//default = 0.5, Cutoff, expressed as a percentage (between 0 and 100), 
											//of frequency at which we print stats for a phone. 
	)
{
	INT2TXT_MAP phone_int2text;

	try {
		StringTable t_phones; //word, number
		if (ReadStringTable((lang / "phones.txt").string(), t_phones) < 0) return -1;
		for (StringTable::const_iterator it(t_phones.begin()), it_end(t_phones.end()); it != it_end; ++it)
			phone_int2text.emplace(std::stoi((*it)[1]), (*it)[0]);

		//this is a special case... for begin - and end - of - sentence stats, we group all nonsilence phones together:
		phone_int2text[0] = "nonsilence";

		//populate the set 'nonsilence', which will contain the integer phone-ids of nonsilence phones(and disambig phones, which won't matter).
		std::set<int> nonsilence;
		for (const auto& pair : phone_int2text)
			nonsilence.insert(pair.first);
		nonsilence.erase(0);

		//Open lang / phones / silence.csl-- while there are many ways of obtaining the silence / nonsilence phones, 
		//we read this because it's present in graph directories as well as lang directories.
		StringTable t_silence; //
		if (ReadStringTable((lang / "phones" / "silence.csl").string(), t_silence, " :") < 0) return -1;
		for (int i = 0; i < t_silence[0].size(); i++)
			nonsilence.erase(std::stoi(t_silence[0][i]));

		/*
		phone_depth_counts is a dict of dicts. For each integer phone-id 'phone', phone_depth_counts[phone] is a map 
		from depth to count (of frames on which that was the 1-best phone in the alignment, and the lattice depth
		had that value).  So we'd access it as count = phone_depth_counts[phone][depth].
		*/
		LD_MAP phone_depth_counts;
		int total_frames = 0;

		for(fs::path p : _latticestats)
		{			
			StringTable t_; //each line has: phone, depth, count
			if (ReadStringTable(p.string(), t_) < 0) return -1;
			for (StringTable::const_iterator it(t_.begin()), it_end(t_.end()); it != it_end; ++it)
			{
				int nc = (*it).size();
				if (nc != 3) {
					LOGTW_ERROR << ".";
					return -1;
				}
				int phone(std::stoi((*it)[0])), depth(std::stoi((*it)[1])), count(std::stoi((*it)[2]));
				LD_MAP1 m1;
				m1.emplace(depth, 0); //NOTE: count=0
				LD_MAP::iterator itm = phone_depth_counts.find(phone);
				if (itm != phone_depth_counts.end())
				{
					LD_MAP1 * pm1 = &itm->second;
					LD_MAP1::iterator itm1 = pm1->find(depth);
					if (itm1 == pm1->end()) {
						pm1->emplace(depth, 0); //NOTE: count=0
					}
				}
				else {
					phone_depth_counts.emplace(phone, m1);
				}

				phone_depth_counts[phone][depth] += count;				
				total_frames += count;
				//
				if (nonsilence.find(phone) != nonsilence.end()) {
					
					//add nonsilence_phone; first check if the index already exists, if not then allocate memory
					int nonsilence_phone = 0;
					LD_MAP::iterator itm0 = phone_depth_counts.find(nonsilence_phone);
					if (itm0 != phone_depth_counts.end())
					{ //nonsilence_phone exists check if depth exists
						LD_MAP1 * pm1 = &itm0->second;
						LD_MAP1::iterator itm1 = pm1->find(depth);
						if (itm1 == pm1->end()) {
							pm1->emplace(depth, 0); //NOTE: count=0
						}
					}
					else {
						LD_MAP1 m1; m1.emplace(depth, 0); //NOTE: count=0
						phone_depth_counts.emplace(nonsilence_phone, m1);
					}
					phone_depth_counts[nonsilence_phone][depth] += count;
				}
				//allocate memory
				LD_MAP1 m1u;
				m1u.emplace(depth, 0); //NOTE: count=0
				int universal_phone = -1;
				phone_depth_counts.emplace(universal_phone, m1u);
				//count
				phone_depth_counts[universal_phone][depth] += count;
			}
		}

		if (total_frames == 0) {
			LOGTW_ERROR << "No input found.";
			return -1;
		}

		LOGTW_INFO << "The total amount of data analyzed assuming 100 frames per second is "
				   << to_string_with_precision(total_frames / 360000.0, 1) << " hours";

		/*
		 the next block prints lines like (to give some examples): 
		 Nonsilence phones as a group account for 74.4% of phone occurrences, with lattice depth (10,50,90-percentile)=(1,2,7) and mean=3.1
		 Phone SIL accounts for 25.5% of phone occurrences, with lattice depth (10,50,90-percentile)=(1,1,4) and mean=2.5
		 Phone Z_E accounts for 2.5% of phone occurrences, with lattice depth (10,50,90-percentile)=(1,2,6) and mean=2.9
		 ...
		*/
		//Sort the phones in decreasing order of count = sort them ascending by count and then access them from end to begin 
		// because they are sorted already in the multimap asceding. For the sorting by count we just flip the key,value pair in
		// _phone_to_lengths. The result is a new multimap (must be multimap because there can be several the same keys now
		// because of the flip!).
		LD_MAPF phone_depth_counts_flipped = flip_map(phone_depth_counts);
		LD_MAPF::reverse_iterator ritm1;
		for (ritm1 = phone_depth_counts_flipped.rbegin(); ritm1 != phone_depth_counts_flipped.rend(); ++ritm1)
		{
			const LD_MAP1 * _depths = &ritm1->first; //IMPORTANT: the map is flipped and the value (depth) is now the key!
			int phone = ritm1->second;
			int sum_depths_values = 0;
			for (const auto& pair2 : *_depths)
				sum_depths_values += pair2.second;
			double frequency_percentage = sum_depths_values * 100.0 / total_frames;
			if (frequency_percentage < frequencyCutoffPercentage)
				continue;

			int depth_percentile_10 = GetPercentile(*_depths, 0.1);
			int depth_percentile_50 = GetPercentile(*_depths, 0.5);
			int depth_percentile_90 = GetPercentile(*_depths, 0.9);
			double depth_mean = GetMean(*_depths);
			std::string preamble;
			if (phone > 0) 
			{
				INT2TXT_MAP::iterator itm = phone_int2text.find(phone);
				if (itm == phone_int2text.end()) {
					LOGTW_ERROR << "Phone " << phone << " is not covered on phones.txt (lang/alignment mismatch?)";
					return -1;
				}
				std::string phone_text = phone_int2text[phone];				
				preamble = "Phone " + phone_text + " accounts for " + to_string_with_precision(frequency_percentage,1) + "% of frames, with";
			}
			else if (phone == 0)
			{
				preamble = "Nonsilence phones as a group account for " + to_string_with_precision(frequency_percentage, 1) + "% of frames, with";
			}
			else {
				assert(phone == -1);
				preamble = "Overall,";
			}

			LOGTW_INFO << preamble << " lattice depth (10, 50, 90-percentile)=("
				<< depth_percentile_10 << ", "
				<< depth_percentile_50 << ", "
				<< depth_percentile_90 << ") and mean=" << to_string_with_precision(depth_mean, 1);
		} ///for ritm1
	}
	catch (const std::exception& ex) {
		LOGTW_FATALERROR << ex.what() << ". (language directory mismatch?)";
		return -1;
	}

	return 0;
}
