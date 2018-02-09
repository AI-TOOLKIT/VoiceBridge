/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
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
using PL_MAP = std::map<std::string, std::map<int, std::map<int, int>>>;
using PL_MAP1 = std::map<int, std::map<int, int>>; //one level deeper in PL_MAP
using PL_MAP1F = std::multimap<std::map<int, int>, int>; //one level deeper in PL_MAP and flipped!
using PL_MAP2 = std::map<int, int>; //two levels deeper in PL_MAP

/*
If length_to_count is a map from length-in-frames to count, return the length-in-frames that equals the (fraction * 100)'th
percentile of the distribution.
*/
static int GetPercentile(const PL_MAP2 & _length_to_count, double fraction)
{//NOTE: _length_to_count is automatically sorted!
	int total_phones = 0;
	for (const auto& pair2 : _length_to_count)
		total_phones += pair2.second;

	if (total_phones == 0)
		return 0;
	else {
		int count_cutoff = (int)(fraction * total_phones);
		int cur_count_total = 0;
		for (const auto& pair2 : _length_to_count)
		{
			int length = pair2.first;
			int count = pair2.second;
			assert(pair2.second >= 0);
			cur_count_total += count;
			if (cur_count_total >= count_cutoff) {
				return length;
			}
		}
		assert(false); //we shouldn't reach here.
		return 0;
	}
}


static double GetMean(const PL_MAP2 & _length_to_count)
{
	int total_phones = 0, total_frames = 0;
	for (const auto& pair2 : _length_to_count)
		total_phones += pair2.second;
	if (total_phones == 0)
		return 0;
	for (const auto& pair2 : _length_to_count)
		total_frames += (pair2.first * pair2.second);
	return 1.0 * total_frames / total_phones;
}


/*
This function reads stats created in AnalyzeAlignments to print information about phone lengths in alignments. 
It's principally useful in order to see whether there is a reasonable amount of silence at the beginning and ends 
of segments.
*/
int AnalyzePhoneLengthStats(
	std::vector<fs::path> & _phonestats,	//input phone stats files
	fs::path lang,							//lang directory	
	float frequencyCutoffPercentage			//default = 0.5, Cutoff, expressed as a percentage (between 0 and 100), 
											//of frequency at which we print stats for a phone. 
	)
{
	INT2TXT_MAP phone_int2text;

	try	{
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

		/* phone_lengths is a dict of dicts of dicts;
		   The main key is boundary_type which can have one of the following 3 elements 'begin', 'end', 'all'.
		   The mapped values are tuples of (phone, length) to a count of occurrences.
		   Where phone is an integer phone-id, and length is the length of the phone instance in frames.
		   NOTE: for the 'begin' and 'end' boundary-types, we group all nonsilence phones into phone-id zero.
		*/
		PL_MAP phone_lengths;

		std::vector<std::string> _boundary_type = { "begin", "end", "all" };

		// total_phones is a dict from boundary_type to total count[of phone occurrences]
		MAPSI total_phones;
		// total_frames is a dict from boundary_type to total number of frames.
		MAPSI total_frames;
		// total_frames is a dict from num-frames to count of num-utterances with that num-frames.

		//NOTE: the below code will automatically allocate memory when needed and will count all phones
		for each(fs::path p in _phonestats) 
		{
			StringTable t_; //format: '1 all 1 101'
			if (ReadStringTable(p.string(), t_) < 0) return -1;
			for (StringTable::const_iterator it(t_.begin()), it_end(t_.end()); it != it_end; ++it)
			{
				int count = std::stoi((*it)[0]);
				std::string boundary_type = (*it)[1];
				int phone = std::stoi((*it)[2]);
				int length = std::stoi((*it)[3]);
				//
				total_phones[boundary_type] += count;
				total_frames[boundary_type] += count * length;
				//
				PL_MAP2 m2;
				m2.emplace(length, 0);
				PL_MAP1 m1;
				m1.emplace(phone, m2);
				PL_MAP::iterator itm = phone_lengths.find(boundary_type);
				if (itm != phone_lengths.end()) 
				{
					PL_MAP1 * pm1 = &itm->second;
					PL_MAP1::iterator itm1 = pm1->find(phone);
					if (itm1 != pm1->end())
					{
						PL_MAP2 * pm2 = &itm1->second;
						PL_MAP2::iterator itm2 = pm2->find(length);
						if (itm2 == pm2->end())
							pm2->emplace(length, 0);
					}
					else {
						pm1->emplace(phone, m2);
					}
				}
				else {
					phone_lengths.emplace(boundary_type, m1);
				}
				phone_lengths[boundary_type][phone][length] += count;
				//
				if (nonsilence.find(phone) != nonsilence.end()) 
				{
					//add nonsilence_phone; first check if the index already exists, if not then allocate memory
					int nonsilence_phone = 0;
					PL_MAP1 * pm1 = &phone_lengths[boundary_type];
					PL_MAP1::iterator itm1 = pm1->find(nonsilence_phone);
					if (itm1 != pm1->end())
					{ //nonsilence_phone exists; check if length exists
						PL_MAP2 * pm2 = &itm1->second;
						PL_MAP2::iterator itm2 = pm2->find(length);
						if (itm2 == pm2->end())
							pm2->emplace(length, 0); //NOTE: count=0
					}
					else {
						PL_MAP2 m2; m2.emplace(length, 0); //NOTE: count=0
						pm1->emplace(nonsilence_phone, m2);
					}
					phone_lengths[boundary_type][nonsilence_phone][length] += count;
				}			
			}
		}

		if (phone_lengths.size()<1) {
			LOGTW_ERROR << "Failed to read phone stat files.";
			return -1;
		}

		//check if all data is consistent with phone_int2text and _boundary_type
		for (auto& pair : phone_lengths)
		{
			std::string boundary_type = pair.first;
			//check boudary type			
			if (std::find(_boundary_type.begin(), _boundary_type.end(), boundary_type) == _boundary_type.end()) 
			{ //not in _boundary_type
				LOGTW_ERROR << "Wrong boundary type " << boundary_type;
				return -1;
			}
			//check phone
			PL_MAP1 * pm1 = &pair.second;
			for (auto& pair1 : *pm1)
			{
				int phone = pair1.first;
				INT2TXT_MAP::iterator itmp2t = phone_int2text.find(phone);
				if (itmp2t == phone_int2text.end()) 
				{ //not in phone_int2text
					LOGTW_ERROR << "Unexpected phone " << phone << " seen (lang directory mismatch?): " << lang.string() << ".";
					return -1;
				}
			}
		}

		// work out the optional-silence phone
		int optional_silence_phone;
		std::string optional_silence_phone_text;
		StringTable t_osilence; //
		bool bOptSilOK = false;
		if (ReadStringTable((lang / "phones" / "optional_silence.int").string(), t_osilence) >= 0)
		{
			if (is_positive_int(t_silence[0][0])) {
				optional_silence_phone = std::stoi(t_silence[0][0]);
				optional_silence_phone_text = phone_int2text[optional_silence_phone];
				bOptSilOK = true;
			}
		}
		if (!bOptSilOK) {
			LOGTW_WARNING << "Was expecting the optional-silence phone to be a member of the silence phones, it is not. This diagnostics may not work correctly.";
			int largest_count = 0;
			optional_silence_phone = 1;
			for (const auto& pair : phone_int2text)
			{
				if (pair.first > 0 &&
					nonsilence.find(pair.first) == nonsilence.end()) //not in nonsilence
				{
					//sum all of the (length * count) for this phone
					int this_count=0;
					PL_MAP2 * pm2 = &phone_lengths["all"][pair.first];
					for (const auto& pair2 : *pm2)
					{
						this_count += (pair2.first * pair2.second);
					}

					if (this_count > largest_count) {
						largest_count = this_count;
						optional_silence_phone = pair.first;
					}
				}
			}
			optional_silence_phone_text = phone_int2text[optional_silence_phone];
			LOGTW_INFO << "Could not get optional-silence phone from " << (lang / "phones" / "optional_silence.int").string()
					  << " guessing that it's " << optional_silence_phone_text << " from the stats.";
		}

		
		/*
		 Analyze frequency, median and mean of optional-silence at beginning and end of utterances.
		 The next block will print something like:
		  - "At utterance begin, SIL is seen 15.0% of the time; when seen, duration (median, mean) is (5, 7.6) frames."
		  - "At utterance end, SIL is seen 14.6% of the time; when seen, duration (median, mean) is (4, 6.1) frames."
		*/
		for (const auto& s : _boundary_type)
		{
			if (s == "all") continue; //only working on 'begin' and 'end'!
			PL_MAP1 * _phone_to_lengths = &phone_lengths[s];
			int num_utterances = total_phones[s];
			assert(num_utterances > 0);
			PL_MAP2 * _opt_sil_lengths = &((*_phone_to_lengths)[optional_silence_phone]);
			int sum_opt_sil_lengths_values = 0;
			for (const auto& pair2 : *_opt_sil_lengths)
				sum_opt_sil_lengths_values += pair2.second;
			double frequency_percentage = sum_opt_sil_lengths_values * 100.0 / num_utterances;

			/*
			 The reason for this warning is that the tradition in speech recognition is to supply a little silence at 
			 the beginning and end of utterances... up to maybe half a second. If your database is not like this, 
			 you should know; you may want to mess with the segmentation to add more silence.
			*/
			if (frequency_percentage < 80.0) {
				LOGTW_WARNING << "Optional-silence " << optional_silence_phone_text << " is seen only " 
					<< std::fixed << std::setprecision(1) << frequency_percentage << "% of the time at utterance " << s << ". This may not be optimal.";
			}
		}

		/*
		 The next will control a sentence that we print and prints lines like:
			- At utterance begin, SIL accounts for 98.4% of phone occurrences, with duration (median, mean, 95-percentile) is (57,59.9,113) frames.
			- At utterance end, nonsilence accounts for 4.2% of phone occurrences, with duration (median, mean, 95-percentile) is (13,13.3,22) frames.
			- Overall, R_I accounts for 3.2% of phone occurrences, with duration (median, mean, 95-percentile) is (6,6.9,12) frames.
		*/
		MAPSS boundary_to_text;
		boundary_to_text.emplace("begin", "At utterance begin");
		boundary_to_text.emplace("end", "At utterance end");
		boundary_to_text.emplace("all", "Overall");

		for (const auto& s : _boundary_type)
		{
			PL_MAP1 * _phone_to_lengths = & phone_lengths[s];
			int tot_num_phones = total_phones[s];
			assert(tot_num_phones > 0);
			//Sort the phones in decreasing order of count = sort them ascending by count and then access them from end to begin 
			// because they are sorted already in the multimap asceding. For the sorting by count we just flip the key,value pair in
			// _phone_to_lengths. The result is a new multimap (must be multimap because there can be several the same keys now
			// because of the flip!).
			PL_MAP1F _phone_to_lengths_flipped = flip_map(*_phone_to_lengths);
			PL_MAP1F::reverse_iterator ritm1;
			for (ritm1 = _phone_to_lengths_flipped.rbegin(); ritm1 != _phone_to_lengths_flipped.rend(); ++ritm1)
			{
				const PL_MAP2 * _lengths = &ritm1->first; //IMPORTANT: the map is flipped and the value is now the key!
				int phone = ritm1->second;
				int sum_lengths_values = 0;
				for (const auto& pair2 : *_lengths)
					sum_lengths_values += pair2.second;
				double frequency_percentage = sum_lengths_values * 100.0 / tot_num_phones;
				if (frequency_percentage < frequencyCutoffPercentage)
					continue;

				int duration_median = GetPercentile(*_lengths, 0.5);
				int duration_percentile_95 = GetPercentile(*_lengths, 0.95);
				double duration_mean = GetMean(*_lengths);

				std::string text = boundary_to_text[s];  //e.g. 'At utterance begin'.

				//
				INT2TXT_MAP::iterator itmp2t = phone_int2text.find(phone);
				if (itmp2t == phone_int2text.end())
				{ //not in phone_int2text
					LOGTW_ERROR << "Phone " << phone << " is not covered in phones.txt (lang/alignment mismatch?).";
					return -1;
				}
				std::string phone_text = itmp2t->second;

				LOGTW_INFO << text << ", " << phone_text << " accounts for " 
					<< std::fixed << std::setprecision(1) << frequency_percentage
						<< "% of phone occurrences, with duration (median, mean, 95-percentile) is ("
					<< std::setprecision(0) << duration_median << ", " 
					<< std::setprecision(1) << duration_mean << ", " 
					<< std::setprecision(0) << duration_percentile_95 << ") frames.";
			} ///for ritm1
		} ///for _boundary_type

		/*
		Print stats on frequency and average length of word-internal optional-silences.
		For optional-silence only, subtract the begin and end-utterance stats from the 'all' stats, 
		to get the stats excluding initial and final phones.
		*/
		total_frames["internal"] = total_frames["all"] - total_frames["begin"] - total_frames["end"];
		total_phones["internal"] = total_phones["all"] - total_phones["begin"] - total_phones["end"];

		PL_MAP2 internal_opt_sil_phone_lengths = phone_lengths["all"][optional_silence_phone];
		for (const auto& pair2 : internal_opt_sil_phone_lengths)
		{
			//subtract the counts for begin and end from the overall counts to get the word - internal count.
			internal_opt_sil_phone_lengths[pair2.first] -= (phone_lengths["begin"][optional_silence_phone][pair2.first] +
													  phone_lengths["end"][optional_silence_phone][pair2.first]);
		}
		if (total_phones["internal"] != 0.0) 
		{
			int total_internal_optsil_frames=0, total_optsil_frames=0;
			for (const auto& pair2 : internal_opt_sil_phone_lengths)
				total_internal_optsil_frames += (pair2.first * pair2.second);
			for (const auto& pair2 : phone_lengths["all"][optional_silence_phone])
				total_optsil_frames += (pair2.first * pair2.second);

			double opt_sil_internal_frame_percent = total_internal_optsil_frames * 100.0 / total_frames["internal"];
			double opt_sil_total_frame_percent = total_optsil_frames * 100.0 / total_frames["all"];
			double internal_frame_percent = total_frames["internal"] * 100.0 / total_frames["all"];

			LOGTW_INFO << "The optional-silence phone " << optional_silence_phone_text << " occupies " 
				<< std::fixed << std::setprecision(1) << opt_sil_total_frame_percent << "% of frames overall.";

			double hours_total = total_frames["all"] / 360000.0;
			double hours_nonsil = (total_frames["all"] - total_optsil_frames) / 360000.0;

			LOGTW_INFO << std::fixed << "Limiting the stats to the " << std::setprecision(1) << internal_frame_percent
					  << "% of frames not covered by an utterance-[begin/end] phone, optional-silence " << optional_silence_phone_text
					  << " occupies " << std::setprecision(1) << opt_sil_internal_frame_percent << "% of frames.";
			  
			LOGTW_INFO << std::fixed << "Assuming 100 frames per second, the alignments represent " << std::setprecision(2) << hours_total
					  << " hours of data, or " << std::setprecision(2) << hours_nonsil << " hours if "
					  << optional_silence_phone_text << " frames are excluded.";
			int sum = 0;
			for (const auto& pair2 : internal_opt_sil_phone_lengths)
				sum += pair2.second;
			double opt_sil_internal_phone_percent = (sum * 100.0 / total_phones["internal"]);
			double duration_median = GetPercentile(internal_opt_sil_phone_lengths, 0.5);
			double duration_mean = GetMean(internal_opt_sil_phone_lengths);
			double duration_percentile_95 = GetPercentile(internal_opt_sil_phone_lengths, 0.95);

			LOGTW_INFO << "Utterance-internal optional-silences " << optional_silence_phone_text << " comprise " 
				<< std::fixed << std::setprecision(1) << opt_sil_internal_phone_percent
				<< "% of utterance-internal phones, with duration (median, mean, 95-percentile) = ("
				<< std::setprecision(0) << duration_median << ", "
				<< std::setprecision(1) << duration_mean << ", "
				<< std::setprecision(0) << duration_percentile_95 << ").";
		}
	} 
	catch (const std::exception& ex) {
		LOGTW_FATALERROR << ex.what() << ".";
		return -1;
	}

	return 0;
}
