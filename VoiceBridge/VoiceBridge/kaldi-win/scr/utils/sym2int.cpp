/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2012 Microsoft Corporation
		   Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/
#include "kaldi-win\scr\kaldi_scr.h"

//NOTE: <field-range> = (field_begin - field_end) can look like 4-5, or 4- (if field_end = -1), or 1 (if field_begin = field_end), 
//		it means the field range in the input to apply the map to.
//		field_begin zero based index of fields
//		field_end zero based index of fields, if < 0 then no end limit!
// smap_oov can be the symbol or it can be the integer value in string form of the OOV.
int Sym2Int(StringTable symtab, StringTable input_txt, fs::path output_txt, 
			int field_begin, int field_end, std::string smap_oov)
{
	//bool ignore_oov = false;
	int num_warning = 0;
	int max_warning = 20;
	int map_oov = -1; //-1 not defined

	if (field_begin > field_end && field_end > -1) {
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
	using MAPT = std::unordered_map<std::string, int>;
	MAPT sym2int;
	for (StringTable::const_iterator it(symtab.begin()), it_end(symtab.end()); it != it_end; ++it)
	{
		if ((*it).size() != 2) {
			LOGTW_ERROR << "Two fields are expected in symbol table but got " << (*it).size() << ".";
			return -1;
		}
		if (!(is_positive_int((*it)[1]))) {
			LOGTW_ERROR << "The second field in the symbol table should be an integer but got " << (*it)[1] << ".";
			return -1;
		}
		else 
			sym2int.emplace((*it)[0], std::stoi((*it)[1]));
	}
	//get map_oov
	if (smap_oov != "" && !(is_positive_int(smap_oov))) { //not numeric->look it up		
		MAPT::iterator itm = sym2int.find(smap_oov);
		if (itm == sym2int.end())
		{// not found
			LOGTW_ERROR << "OOV symbol " << smap_oov << " is not defined.";
			return -1;
		}
		else {
			map_oov = itm->second;
		}
	}
	else if (smap_oov != "" && is_positive_int(smap_oov))
	{
		map_oov = std::stoi(smap_oov);
	}

	for (StringTable::const_iterator it(input_txt.begin()), it_end(input_txt.end()); it != it_end; ++it)
	{
		int c = 0;
		std::vector<std::string> mapped;
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
		{
			if ((field_begin < 0 || c >= field_begin) && (field_end < 0 || c <= field_end))
			{
				std::string a = *itc1;
				MAPT::iterator itm = sym2int.find(a);
				if (itm != sym2int.end())
				{ //found
					std::string s = std::to_string(itm->second);
					mapped.push_back(s); //add found value from map
				}
				else
				{//not found
					if (map_oov > -1)
					{
						if (num_warning++ < max_warning) {
							LOGTW_WARNING << " replacing '" << a << "' with " << map_oov << " (OOV symbol).";
							if (num_warning == max_warning) {
								LOGTW_WARNING << " not warning for OOVs any more times.";
							}
						}
						std::string s = std::to_string(map_oov);
						mapped.push_back(s);
					}
					else {
						LOGTW_ERROR << "Undefined symbol '" << a << "' (in position " << (c+1) << ").";
						return -1;
					}
				}
			}
			else {
				//save all fields, also which are not converted here
				mapped.push_back(*itc1);
			}

			c++;
		}
		//write output file
		file_output_txt << join_vector(mapped, " ") << "\n";
	}
	file_output_txt.flush(); file_output_txt.close();

	if (num_warning > 0) {
		LOGTW_WARNING << " Replaced " << num_warning << " instances of OOVs with " << smap_oov << " (" << map_oov << ").";
	}

	return 0;

}
