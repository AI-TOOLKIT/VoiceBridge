/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"

/*
	Checks if the entries of a file containing disambiguation symbols (word or phone level) are all valid. To be valid the symbols
	- must start with the hash mark '#', 
	- must not contain any whitespace,
	- must not be equal to '#-1' (disallowed because it is used internally in some FST stuff).

	In case the option '--allow-numeric' is used with 'false', the symbols must also be non-numeric 
	(to avoid overlap with the automatically generated symbols).

	Allowed options:
	bAllowNumeric (true|false) : Default true. If false, disallow numeric disambiguation symbols like #0, #1 and so on.
*/
int ValidateDisambigSymFile(fs::path disambig_sym_file, bool bAllowNumeric)
{
	StringTable table_disambig_sym = readData(disambig_sym_file.string());
	static const boost::regex rexp1("^#(.*)$");
	static const boost::regex rexp2("^[0-9]+$");
	static const boost::regex rexp3("\\s+"); //NOTE: the \s must be written as \\s because the compiler eats the first \ as escape!
	static const boost::regex rexp4("^#");
	boost::match_results<std::string::const_iterator> results;
	for (StringTable::const_iterator it(table_disambig_sym.begin()), it_end(table_disambig_sym.end()); it != it_end; ++it)
	{ 
		std::string symbol((*it)[0]);
		if (boost::regex_match(symbol, results, rexp1)) {
			std::string sympart = boost::regex_replace(symbol, rexp4, "");
			if (sympart == "") {
				LOGTW_ERROR << " Only symbol " << symbol << " is not allowed as a disambiguation symbol in file: " << disambig_sym_file.string() << ".";
				return -1;
			}
			else if (sympart == "-1") {
				LOGTW_ERROR << " The disambiguation symbol " << symbol << " is not allowed in file: " << disambig_sym_file.string() << ".";
				return -1;
			}
			else if (boost::regex_match(sympart, results, rexp3))
			{
				LOGTW_ERROR << " The disambiguation symbol " << symbol << " contains only whitespace in file: " << disambig_sym_file.string() << ".";
				return -1;
			}
			else if (!bAllowNumeric && boost::regex_match(sympart, results, rexp2))
			{
				LOGTW_ERROR << " Since " << symbol << " is supposed to be an extra disambiguation symbol, it must not be numeric in file: " << disambig_sym_file.string() << ".";
				return -1;
			}
		}
		else {
			LOGTW_ERROR << " The disambiguation symbol " << symbol << " does not start with a '#' in file: " << disambig_sym_file.string() << ".";
			return -1;
		}
	}

	return 0;
}
