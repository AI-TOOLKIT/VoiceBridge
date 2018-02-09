/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2012 Microsoft Corporation  
							   Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"

int int_2_sym (std::string a, int pos, std::unordered_map<int, std::string> & int2sym, std::string & sym)
{
	//check if all digits
	int ai = StringToNumber<int>(a, -9999);
	if (ai == -9999) {
		LOGTW_ERROR << "Found noninteger token " << a << " [in position " << pos << "]";
		return -1;
	}
	std::unordered_map<int, std::string>::iterator itm = int2sym.find(ai);
	if (itm == int2sym.end())
	{ //not found
		LOGTW_ERROR << "Integer " << a << " not in symbol table.";
		return -1;
	}

	sym = itm->second;
	return 0;
}

/*
	field_begin and field_end are zero based indexes of fields. -1 means from the start or till the end.
*/
int Int2Sym(StringTable symtab, StringTable input_txt, fs::path output_txt,
			int field_begin, int field_end)
{
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
	std::unordered_map<int, std::string> _int2sym;
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
			_int2sym.emplace(std::stoi((*it)[1]), (*it)[0]);
	}	

	for (StringTable::const_iterator it(input_txt.begin()), it_end(input_txt.end()); it != it_end; ++it)
	{
		int c = 0;
		std::vector<std::string> mapped;
		for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
		{
			if ((field_begin < 0 || c >= field_begin) && (field_end < 0 || c <= field_end))
			{
				std::string sym;
				if (int_2_sym(*itc1, (c+1), _int2sym, sym) < 0) return -1;
				mapped.push_back(sym);
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
	return 0;
}
