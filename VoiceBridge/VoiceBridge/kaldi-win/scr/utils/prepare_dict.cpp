/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on : Copyright 2014 Vassil Panayotov, Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"
#include "..\phonetisaurus\Phonetisaurus.h"
#include "kaldi-win/scr/Params.h"

/*
	Generalized function which creates all necessary file for the dictionary folder.
	In case the reference dictionary does not exist then it expects to have a ready lexicon in the
	input folder!
*/
VOICEBRIDGE_API int PrepareDict(fs::path refDict,
	const std::map<std::string, std::string> & silphones, //silence phones e.g. !SIL SIL, <UNK> SPN
	const std::map<std::string, std::string> & optsilphones) //optional silence phones e.g. !SIL SIL
{

	//lexicon.txt, lexicon_nosil.txt
	//prepare lexicon - the lexicon is a word list with pronounciations. It must be created by using a reference
	//					lexicon and with a trained model on the reference lexicon. 
	//					E.g. for English we use cmudict.dict. Such reference pronounciation
	//					dictionary exists for all languages. In case we have new words this step will
	//					add the new word with the identified pronunciation to the final lexicon.
	fs::path out_dict(voicebridgeParams.pth_project_input / "lexicon.txt");
	bool bLexiconOK = false;
	if (refDict=="" || !fs::exists(refDict) || fs::is_empty(refDict)) {
		LOGTW_WARNING << "The reference dictionary does not exist or empty. Expecting to have a ready lexicon...";
		if (!fs::exists(out_dict) || fs::is_empty(out_dict)) {
			LOGTW_ERROR << "The reference dictionary does not exist and there is no lexicon provided!";
			return -1;
		}
		bLexiconOK = true;
		LOGTW_INFO << "Found already existing lexicon and using it...";
	}	

	fs::path model(refDict.string() + ".model");
	fs::path in_dict(voicebridgeParams.pth_data / "vocab.txt"); //NOTE: this is made automatically by PrepareData()!
	bool bLexiconCreated = false;
	//only make the pronunciation model if necessary 
	if (!bLexiconOK) 
	{ 
		//this step is necessary if there is a reference dictionary!

		if (!fs::exists(model) || fs::last_write_time(model) < fs::last_write_time(refDict))
		{
			LOGTW_INFO << "Training pronunciation model for lexicon creation...";
			if (Phonetisaurus::TrainModel(refDict, model) < 0) return -1;
		}
		else LOGTW_INFO << "The pronunciation model is up to date. Skipping model training.";
		
		if (!fs::exists(in_dict)) {
			LOGTW_ERROR << in_dict.string() << " does not exist.";
			return -1;
		}

		if (!fs::exists(voicebridgeParams.pth_project_input / "lexicon.txt")
			|| fs::last_write_time(voicebridgeParams.pth_project_input / "lexicon.txt") < fs::last_write_time(model))
		{
			/*
			Use phonetisaurus to make a g2p model from the reference lexicon.
			Use the g2p model to develop a special lexicon for our project.
			-- this will make sure that there are no words in the lexicon which are not needed (less processing time and memory)
			-- in case of extra words not in the ref lexicon it will estimate the pronounciation
			*/
			LOGTW_INFO << "Creating lexicon with trained model...";
			if (Phonetisaurus::GetPronunciation(refDict, model, in_dict, out_dict, false) < 0) return -1;

			//lexicon_nosil.txt
			//the lexicon we have just made is the lexicon_nosil.txt without SIL phones
			try {
				fs::copy_file(voicebridgeParams.pth_project_input / "lexicon.txt", voicebridgeParams.pth_project_input / "lexicon_nosil.txt", fs::copy_option::overwrite_if_exists);
			}
			catch (const std::exception& e)
			{
				LOGTW_ERROR << e.what();
				return -1;
			}
			bLexiconCreated = true;
		}
		else {
			LOGTW_INFO << "The lexicon lexicon.txt is up to date. Skipping lexicon creation.";
			LOGTW_INFO << "  NOTE: The model used to create the lexicon is up to date but you may recreate the lexicon by deleting: ";
			LOGTW_INFO << "  " << (voicebridgeParams.pth_project_input / "lexicon.txt").string();
		}
	}

	try {
		if (!fs::exists(voicebridgeParams.pth_silence_phones_txt.branch_path()))
			fs::create_directories(voicebridgeParams.pth_silence_phones_txt.branch_path());
	}
	catch (std::exception const& e)
	{
		LOGTW_FATALERROR << " Exception: " << e.what();
		return -1;
	}
	catch (...)
	{
		LOGTW_FATALERROR << " Exception: Unknown";
		return -1;
	}

	fs::ofstream file_silence_phones_txt(voicebridgeParams.pth_silence_phones_txt, std::ios::binary | std::ios::out);
	if (!file_silence_phones_txt) {
		LOGTW_ERROR << " Can't open output file: " << voicebridgeParams.pth_silence_phones_txt.string();
		return -1;
	}
	fs::ofstream file_optional_silence_txt(voicebridgeParams.pth_optional_silence_txt, std::ios::binary | std::ios::out);
	if (!file_optional_silence_txt) {
		LOGTW_ERROR << " Can't open output file: " << voicebridgeParams.pth_optional_silence_txt.string();
		return -1;
	}

	//get a unique list of silence phones
	std::vector<std::string> _s;
	for (auto & pair : silphones)
		_s.push_back(pair.second);
	_s.erase(std::unique(_s.begin(), _s.end()), _s.end());
	//get a unique list of non-silence phones
	std::vector<std::string> _os;
	for (auto & pair : optsilphones)
		_os.push_back(pair.second);
	_os.erase(std::unique(_os.begin(), _os.end()), _os.end());
	//write out both
	for (std::string s : _s)
		file_silence_phones_txt << s << "\n";
	for (std::string s : _os)
		file_optional_silence_txt << s << "\n";
	file_silence_phones_txt.flush(); file_silence_phones_txt.close();
	file_optional_silence_txt.flush(); file_optional_silence_txt.close();

	//extract and save non-silence phones
	fs::ofstream file_nonsilence_phones_txt(voicebridgeParams.pth_nonsilence_phones_txt, std::ios::binary | std::ios::out);
	if (!file_nonsilence_phones_txt) {
		LOGTW_ERROR << " Can't open output file: " << voicebridgeParams.pth_nonsilence_phones_txt.string();
		return -1;
	}
	StringTable _lexicon;
	if (ReadStringTable((voicebridgeParams.pth_project_input / "lexicon_nosil.txt").string(), _lexicon) < 0) return -1;
	std::map<std::string, string_vec> _stmp;
	static const boost::regex rexp("[0-9]+");
	for (string_vec & _s : _lexicon) {
		if (_s.size() < 2) continue; //normally should never happen
		for (int i = 1; i < _s.size(); i++) {
			//add the phone as it is (e.g. AE0) and add it also without the variant index (e.g. AE0 -> AE)
			std::string key = boost::regex_replace(_s[i], rexp, "");
			std::map<std::string, string_vec>::iterator cit = _stmp.find(key);
			if (cit == _stmp.end()) {
				if (_s[i] == key)
					_stmp.emplace(key, string_vec{}); //do not add if not variant phone
				else _stmp.emplace(key, string_vec{ _s[i] });
			}
			else {
				if (_s[i] != key) //do not add if not variant phone
					cit->second.push_back(_s[i]);
			}
		}
	}
	//at this point we have all variant phones per phone in a sorted map; we just need to sort them and make unique
	for (auto & pair : _stmp) {
		std::sort(pair.second.begin(), pair.second.end());
		pair.second.erase(std::unique(pair.second.begin(), pair.second.end()), pair.second.end());

		//save the results
		file_nonsilence_phones_txt << pair.first;
		for (std::string s : pair.second)
			file_nonsilence_phones_txt << " " << s;
		file_nonsilence_phones_txt << "\n";
	}
	file_nonsilence_phones_txt.flush(); file_nonsilence_phones_txt.close();

	//Extract and save extra questions - will be added to those obtained by automatically clustering the "real" phones.
	//These ask about stress; there's also one for silence. 
	fs::ofstream file_extra_questions_txt(voicebridgeParams.pth_extra_questions_txt, std::ios::binary | std::ios::out);
	if (!file_extra_questions_txt) {
		LOGTW_ERROR << " Can't open output file: " << voicebridgeParams.pth_extra_questions_txt.string();
		return -1;
	}
	//1. add silence phones
	int cc = 0;
	for (std::string s : _s) {
		if (cc>0) file_extra_questions_txt << " ";
		file_extra_questions_txt << s;
		cc++;
	}
	//add non silence phones
	/*final format example:
	SIL SPN AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH
	AA2 AE2 AH2 AO2 AW2 AY2 EH2 ER2 EY2 IH2 IY2 OW2 OY2 UH2 UW2
	AA1 AE1 AH1 AO1 AW1 AY1 EH1 ER1 EY1 IH1 IY1 OW1 OY1 UH1 UW1
	AA0 AE0 AH0 AO0 AW0 AY0 EH0 ER0 EY0 IH0 IY0 OW0 OY0 UH0 UW0
	*/
	StringTable _nsp;
	if (ReadStringTable(voicebridgeParams.pth_nonsilence_phones_txt.string(), _nsp) < 0) return -1;
	//2. add all main phones
	for (string_vec & _s : _nsp)
		file_extra_questions_txt << " " << _s[0];
	file_extra_questions_txt << "\n";
	//3. add phone variants from back to front
	//find out the maximum number of columns
	int maxc = 0;
	for (string_vec & _s : _nsp)
		if (maxc < _s.size()) maxc = _s.size();
	for (int i = maxc - 1; i > 0; i--) {
		cc = 0;
		for (string_vec & _s : _nsp) {
			if (_s.size() >= i + 1) {
				if (cc>0) file_extra_questions_txt << " ";
				file_extra_questions_txt << _s[i];
				cc++;
			}
		}
		if (cc>0) file_extra_questions_txt << "\n";
	}
	file_extra_questions_txt.flush(); file_extra_questions_txt.close();

	LOGTW_INFO << "Silence phones saved to: ";
	LOGTW_INFO << "  " << voicebridgeParams.pth_silence_phones_txt.string();
	LOGTW_INFO << "Optional silence saved to: ";
	LOGTW_INFO << "  " << voicebridgeParams.pth_optional_silence_txt.string();
	LOGTW_INFO << "Non-silence phones saved to: ";
	LOGTW_INFO << "  " << voicebridgeParams.pth_nonsilence_phones_txt.string();
	LOGTW_INFO << "Extra triphone clustering-related questions saved to: ";
	LOGTW_INFO << "  " << voicebridgeParams.pth_extra_questions_txt.string();

	//add silence phones to lexicon.txt
	if (bLexiconCreated) 
	{//NOTE: skip this step if we use the existing lexicon because otherwise things will be added twice!
		try {
			fs::copy_file(voicebridgeParams.pth_project_input / "lexicon.txt", voicebridgeParams.pth_project_input / "lexicon.txt.tmp");
			fs::ofstream outputFile(voicebridgeParams.pth_project_input / "lexicon.txt", std::ios::binary | std::ios::out);
			fs::ifstream inputFile(voicebridgeParams.pth_project_input / "lexicon.txt.tmp");
			//add silence phones
			for (auto & pair : silphones)
				outputFile << pair.first << " " << pair.second << "\n";
			outputFile << inputFile.rdbuf();
			inputFile.close();
			outputFile.flush(); outputFile.close();
			fs::remove(voicebridgeParams.pth_project_input / "lexicon.txt.tmp");
			//copy to local dict folder
			fs::copy_file(voicebridgeParams.pth_project_input / "lexicon.txt", voicebridgeParams.pth_lexicon_txt);
		}
		catch (const std::exception& e)
		{
			LOGTW_ERROR << e.what();
			return -1;
		}

		LOGTW_INFO << "Lexicon text file saved as: ";
		LOGTW_INFO << "  " << (voicebridgeParams.pth_project_input / "lexicon.txt").string();
	}
	else {
		//just copy to local dict folder
		try	{
			fs::copy_file(voicebridgeParams.pth_project_input / "lexicon.txt", voicebridgeParams.pth_lexicon_txt);
		}
		catch (const std::exception& e)
		{
			LOGTW_ERROR << e.what();
			return -1;
		}
	}

	return 0;
}
