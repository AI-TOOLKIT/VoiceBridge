/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on :	Copyright 2010-2012 Microsoft Corporation, Apache2
			Johns Hopkins University (author: Daniel Povey)
*/

#include "kaldi-win\scr\kaldi_scr.h"

/*
This function takes a list of utterance-ids or any file whose first field of each line is an utterance-id, 
and filters an scp file (or any file whose "n-th" field is an utterance id), printing out only those 
lines whose "n-th" field is in id_list. The index of the "n-th" field is 1, by default, but can be 
changed by using the -f <n> switch

field : zero based index of the fields/columns
*/
int FilterScp(fs::path idlist, fs::path in_scp, fs::path out_scp, bool exclude, int field)
{
	string_vec seen;
	StringTable table_idlist, table_in_scp;
	if (ReadStringTable(idlist.string(), table_idlist) < 0) return -1;
	if (ReadStringTable(in_scp.string(), table_in_scp) < 0) return -1;
	int line = 1;
	for (StringTable::const_iterator it(table_idlist.begin()), it_end(table_idlist.end()); it != it_end; ++it)
	{
		if ( !((*it).size() >= 1) ) {
			LOGTW_ERROR << " Invalid id-list data in " << idlist.string() << " at line " << line << ".";
			return -1;
		}
		seen.push_back((*it)[0]);
		line++;
	}

	fs::ofstream file_out_scp(out_scp, std::ios::binary);
	if (!file_out_scp) {
		LOGTW_ERROR << " Can't open output file: " << out_scp.string() << ".";
		return -1;
	}
	line = 1;
	for (StringTable::const_iterator it(table_in_scp.begin()), it_end(table_in_scp.end()); it != it_end; ++it)
	{
		if (!((*it).size() >= field+1)) {
			LOGTW_ERROR << " Invalid input scp data in " << in_scp.string() << " at line " << line << ".";
			return -1;
		}
		if ((!exclude && std::find(seen.begin(), seen.end(), (*it)[field]) != seen.end() ) 
			|| (exclude && std::find(seen.begin(), seen.end(), (*it)[field]) == seen.end())) 
		{
			int c = 0;
			for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
			{
				if (c > 0) file_out_scp << " ";
				file_out_scp << *itc1;
				c++;
			}
			file_out_scp << "\n";
		}
		line++;
	}

	file_out_scp.flush(); file_out_scp.close();

	return 0;
}
