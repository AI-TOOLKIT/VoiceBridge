/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt. 

	Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/
#include "kaldi-win\scr\kaldi_scr.h"

//NOTE: Applies the map 'map' to all input text, where each line of the map is interpreted as a map from the first field 
//to the list of the other fields. It doesn't assume the things being mapped to are single tokens, they could be sequences of tokens.
//e.g.: map file contains:
//	A a1 a2
//	B b
// input_txt contains:
//	A B
//will produce in output_txt:
//	a1 a2 b
//NOTE: <field-range> = (field_begin - field_end) can look like 4-5, or 4- (if field_end = -1), or 1 (if field_begin = field_end), 
//		it means the field range in the input to apply the map to.
//		field_begin zero based index of fields
//		field_end zero based index of fields, if < 0 then no end limit!
int ApplyMap(StringTable input_map, StringTable input_txt, fs::path output_txt, int field_begin, int field_end, bool bPermissive)
{
	if(field_begin > field_end && field_end > -1) {
		std::swap(field_begin, field_end);		
	}
	//open output file
	//NOTE: ofstream write adds an '\r' in front of the '\n' automatically and some code in Kaldi crashes. To prevent this
	//		add std::ios::binary option to each ofstream to make sure that '\r' is not added!
	fs::ofstream file_output_txt(output_txt, std::ios::binary);
	if (!file_output_txt) {
		LOGTW_ERROR << "Can't open output file: " << output_txt.string() << ".";
		return -1;
	}
	//the first field of the map is the key and the rest of the fields are the value in the map
	using MAPT = std::unordered_map<std::string, std::string>;
	MAPT map;
	for (StringTable::const_iterator it(input_map.begin()), it_end(input_map.end()); it != it_end; ++it)
	{
		int c = 0;
		std::string sKey(""), sVal("");
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
		{			
			if (c > 1) sVal.append(" "); //separator
			if (c > 0) sVal.append(*itc1);
			else sKey.append(*itc1);
			c++;
		}
		map.emplace(sKey, sVal);
	}
	//
	for (StringTable::const_iterator it(input_txt.begin()), it_end(input_txt.end()); it != it_end; ++it)
	{
		int c = 0, m = 0;
		std::string mapped("");
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
		{
			if ((field_begin < 0 || c >= field_begin) && (field_end < 0 || c <= field_end)) 
			{
				std::string a = *itc1;
				MAPT::iterator itm = map.find(a);
				if(itm != map.end()) 
				{ //found
					if(m>0) mapped.append(" "); //separator
					mapped.append(itm->second); //add found value from map
					m++;
				}
				else 
				{//not found
					if (!bPermissive) {
						LOGTW_ERROR << "Undefined key '" << a << "' in map file.";
						return -1;
					}
					else {
						LOGTW_WARNING << "Missing key '" << a << "' in map file.";
					}
				}
			}
			c++;
		}
		//write output file
		file_output_txt << mapped << "\n";
	}
	file_output_txt.flush(); file_output_txt.close();

	return 0;
}
