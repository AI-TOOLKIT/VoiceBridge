/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2012 Microsoft Corporation
		   Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0.
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "mitlm/mitlm.h"
#include "kaldi-win/scr/Params.h"
#include "kaldi-win/utility/Utility2.h"

/*
	SaveUtt2Spk : automatically identifies the speaker id's and saves them into the utt2spk file.
	idtype :	  0 - directory name is the speaker id (spaces replaced with underscore), NOTE: all wav files in the directory are from a spesific speaker
				  1 - utt_id derived from the file name is the speaker id,  NOTE: this will not identify separate speakers DEFAULT
				  idtype>1 the first idtype number of characters from the file name is the speaker id, the wav files can be in the same directory
*/
int SaveUtt2Spk(fs::path out, fs::path pscp, int idtype)
{
	std::map<std::string, std::string> map; //sorted and unique!
	try	{
		fs::ofstream ofile(out, std::ios::binary | std::ios::out);
		if (!ofile) {
			LOGTW_ERROR << " Can't open output file: " << out.string();
			return -1;
		}

		//read in the file line by line and after the first space (after the utt_id or file_id) is the file path
		fs::ifstream ifs(pscp);
		if (!ifs) {
			throw std::runtime_error("Error opening file.");
		}
		std::string line;
		while (std::getline(ifs, line)) {
			boost::algorithm::trim(line);
			//find the first space and delete untill the space
			std::string::size_type i = line.find(" ");
			if (i != std::string::npos) {
				std::string utt_id = line.substr(0, i);
				line.erase(0, i+1);
				fs::path p(line);
				if (!fs::exists(p)) {
					LOGTW_ERROR << "Input file does not exist: " << p.string();
					return -1;
				}
				if (idtype == 0)
				{ //the speaker id = directory name
					std::string dir = p.parent_path().filename().string();
					//sanity check
					std::string::size_type j = dir.find(":");
					if (j == std::string::npos)
					{ //OK
						//make sure that the speaker id does not contain spaces
						ReplaceStringInPlace(dir, " ", "_");
						map.emplace(utt_id, dir);
					}
					else 
					{ //drive letter, ignoring directory as speaker id
						map.emplace(utt_id, utt_id);
					}
				}
				else if (idtype == 1)
				{ //the speaker id = utt_id
					map.emplace(utt_id, utt_id);
				}
				else if (idtype > 1)
				{ //the speaker id = the first idtype number of characters from the file name
					std::string filename = p.stem().string();
					std::string speaker_id = filename.substr(0, idtype);
					//make sure that the speaker id does not contain spaces
					ReplaceStringInPlace(speaker_id, " ", "_");
					map.emplace(utt_id, speaker_id);
				}
				else {
					LOGTW_ERROR << "Wrong idtype in PrepareData.";
					return -1;
				}
			}
		}
		ifs.close();
		//save
		for (auto & pair : map)
		{
			ofile << pair.first << " " << pair.second << '\n';
		}
		ofile.flush(); ofile.close();
	}
	catch (std::exception const& e)
	{
		LOGTW_FATALERROR << " " << e.what() << ".";
		return -1;
	}
	catch (...)
	{
		LOGTW_FATALERROR << "Unknown error while writing " << out.string();
		return -1;
	}
	return 0;
}

/*
	percentageTrain : the percentage of training data
	transc_ext :	  the extension of the transcription files (must have the same file name as the wav files
					  but with this extension;
	idtype :		  0 - directory name is the speaker id (spaces replaced with underscore), NOTE: all wav files in the directory are from a spesific speaker
					  1 - utt_id derived from the file name is the speaker id,  NOTE: this will not identify separate speakers DEFAULT
					  idtype>1 the first idtype number of characters from the file name is the speaker id, the wav files can be in the same directory
*/
VOICEBRIDGE_API int PrepareData(int percentageTrain, std::string transc_ext, int orderngram, int idtype)
{

	fs::path waves_dir_path = fs::path(voicebridgeParams.waves_dir, fs::native);

	if (!fs::exists(waves_dir_path)) {
		LOGTW_ERROR << " The path does not exist: " << waves_dir_path.string() << ".";
		return -1;
	}

	//In case the data directory exists then make a backup and clear the directory
	if (fs::exists(voicebridgeParams.pth_data)) 
	{
		LOGTW_INFO << "Creating backup of existing data directory...";
		try	{
			if (Zip(voicebridgeParams.pth_data) < 0) {
				//second attempt
				SleepWait(2); //wait 1 second
				if (Zip(voicebridgeParams.pth_data) < 0) {
					LOGTW_ERROR << " Could not backup and delete data directory " << voicebridgeParams.pth_data << ". Please make a backup and delete it manually.";
					return -1;
				}
			}
			for (fs::directory_iterator end_dir_it, it(voicebridgeParams.pth_data); it != end_dir_it; ++it) {
				fs::remove_all(it->path());
			}
			//rename the archive - append the current time
			std::stringstream sT;
			auto t = std::time(nullptr);
			auto tm = *std::localtime(&t);
			sT << std::put_time(&tm, "%Y%m%d-%H%M%S");
			fs::path newpath(voicebridgeParams.pth_project_base / ("backup-data" + sT.str() + ".zip"));
			if (fs::exists(newpath))
				fs::remove(newpath);
			fs::rename(voicebridgeParams.pth_project_base / "data.zip", newpath);
			LOGTW_INFO << "Data succesfully backed up to " << newpath;
			LOGTW_INFO << "Preparing new data...";
		}
		catch (const std::exception&) {
			LOGTW_ERROR << " Could not backup and delete data directory " << voicebridgeParams.pth_data << ". Please make a backup and delete it manually.";
			return -1;
		}
	}

	std::string  train_scp_file_name(voicebridgeParams.train_base_name);
	train_scp_file_name.append("_wav.scp");
	std::string  test_scp_file_name(voicebridgeParams.test_base_name);
	test_scp_file_name.append("_wav.scp");
	std::string  train_txt_file_name(voicebridgeParams.train_base_name);
	train_txt_file_name.append(".txt");
	std::string  test_txt_file_name(voicebridgeParams.test_base_name);
	test_txt_file_name.append(".txt");

	//make the project directory structure
	fs::path pth_waves_all_list(voicebridgeParams.pth_local / "waves_all.list");
	fs::path pth_waves_test_list(voicebridgeParams.pth_local / "waves.test");
	fs::path pth_waves_train_list(voicebridgeParams.pth_local / "waves.train");
	fs::path pth_train_scp_list(voicebridgeParams.pth_local / train_scp_file_name);
	fs::path pth_test_scp_list(voicebridgeParams.pth_local / test_scp_file_name);
	fs::path pth_train_txt_list(voicebridgeParams.pth_local / train_txt_file_name);
	fs::path pth_test_txt_list(voicebridgeParams.pth_local / test_txt_file_name);

	try
	{
		if (!fs::exists(pth_waves_all_list.branch_path()))
			fs::create_directories(pth_waves_all_list.branch_path());
	}
	catch (std::exception const& e)
	{
		LOGTW_FATALERROR << " " << e.what() << ".";
		return -1;
	}
	catch (...)
	{
		LOGTW_FATALERROR << " Unknown Error.";
		return -1;
	}

	fs::ofstream file_waves_all_list(pth_waves_all_list, std::ios::binary);
	fs::ofstream file_waves_train_list(pth_waves_train_list, std::ios::binary);
	fs::ofstream file_waves_test_list(pth_waves_test_list, std::ios::binary);
	fs::ofstream file_train_scp_list(pth_train_scp_list, std::ios::binary);
	fs::ofstream file_test_scp_list(pth_test_scp_list, std::ios::binary);
	fs::ofstream file_train_txt_list(pth_train_txt_list, std::ios::binary);
	fs::ofstream file_test_txt_list(pth_test_txt_list, std::ios::binary);
	fs::ofstream file_full_txt(voicebridgeParams.pth_data / "full_text.txt", std::ios::binary); //full text

	if (!file_waves_all_list) {
		LOGTW_ERROR << " can't open output file: " << pth_waves_all_list.string() << ".";
		return -1;
	}
	if (!file_waves_train_list) {
		LOGTW_ERROR << " can't open output file: " << pth_waves_train_list.string() << ".";
		return -1;
	}
	if (!file_waves_test_list) {
		LOGTW_ERROR << " can't open output file: " << pth_waves_test_list.string() << ".";
		return -1;
	}
	if (!file_train_scp_list) {
		LOGTW_ERROR << " can't open output file: " << pth_train_scp_list.string() << ".";
		return -1;
	}
	if (!file_test_scp_list) {
		LOGTW_ERROR << " can't open output file: " << pth_test_scp_list.string() << ".";
		return -1;
	}
	if (!file_train_txt_list) {
		LOGTW_ERROR << " can't open output file: " << pth_train_txt_list.string() << ".";
		return -1;
	}
	if (!file_test_txt_list) {
		LOGTW_ERROR << " can't open output file: " << pth_test_txt_list.string() << ".";
		return -1;
	}
	if (!file_full_txt) {
		LOGTW_ERROR << " can't open output file: " << (voicebridgeParams.pth_data / "full_text.txt").string() << ".";
		return -1;
	}

	typedef std::vector<fs::path> vec_fsp;
	vec_fsp v = GetFilesInDir(waves_dir_path, ".wav", true);   //recursive!
	int count = (int)(v.end() - v.begin());
	int k = 0;

	//check if the transc_ext has a dot, if not then add it
	if (transc_ext.find_first_of('.') != 0) transc_ext = "." + transc_ext;

	for (vec_fsp::const_iterator it(v.begin()), it_end(v.end()); it != it_end; ++it)
	{
		k++;

		//NOTE: file paths may contain spaces on Windows but this does not seem to be a problem
		std::string filepath((*it).string()); //full path 

		std::string filename((*it).filename().string()); //only the file name without the path!
		std::string filebasename((*it).stem().string()); //only the file name without the path and without the extension
		//replacing spaces in file names with underscore for the file_id and utt_id!
		std::string FILE_ID = filebasename;
		ReplaceStringInPlace(FILE_ID, " ", "_");

		file_waves_all_list << filename << '\n';

		//make the file name of the transcription file
		fs::path ftrn(filepath);
		ftrn = fs::change_extension(ftrn, transc_ext);

		//- Get the first line from the transcription file NOTE: it is assumed that there is only 1 line!		
		std::string sline;
		try {
			sline = GetFirstLineFromFile(ftrn.string());
		}
		catch (const std::exception& e)
		{
			LOGTW_ERROR << e.what();
			return -1;
		}
		//- Convert the text to lower case.
		ConvertToCaseUtf8(sline, false);
		
		std::string strans;
		if (k <= count * percentageTrain / 100) {
			//waves.train - one wav file name per line
			file_waves_train_list << filename << '\n';
			//wav.scp
			//Indexing files to unique ids. Can use file names as file_ids.
			//<file_id> <wave filename with path> 
			//e.g.: 0_1_0_0_1_0_1_1 waves_yesno/0_1_0_0_1_0_1_1.wav 
			file_train_scp_list << FILE_ID << " " << filepath << '\n';
			//.txt:
			//Essentially, transcripts. An utterance per line. Can use filenames without extensions as utt_ids.
			//<utt_id> <transcript> 
			//e.g.: 0_0_1_1_1_1_0_0 NO NO YES YES YES YES NO NO
			//search for the file with file name the same as the wav file but ending in '.trn'
			try {
				file_train_txt_list << FILE_ID << " " << sline << '\n';

				//make a full text file for creating the arpa model
				//NOTE: this file is new and not included in the original Kaldi version. Contains all text (train+test)!
				file_full_txt << sline << '\n';
			}
			catch (const std::exception& e) {
				LOGTW_ERROR << " error reading transcriptions: " << ftrn.string() << " Reason: " << e.what() << ".";
				return -1;
			}
		}
		else {
			//waves.test
			file_waves_test_list << filename << '\n';
			//wav.scp
			file_test_scp_list << FILE_ID << " " << filepath << '\n';
			//.txt:
			//search for the file with file name the same as the wav file but ending in '.trn'
			try {
				file_test_txt_list << FILE_ID << " " << sline << '\n';

				//make a full text file for creating the arpa model
				//NOTE: this file is new and not included in the original Kaldi version. Contains all text (train+test)!
				file_full_txt << sline << '\n';
			}
			catch (const std::exception& e) {
				LOGTW_ERROR << " error reading transcriptions: " << ftrn.string() << " Reason: " << e.what() << ".";
				return -1;
			}
		}
	}
	file_waves_all_list.flush(); file_waves_all_list.close();
	file_waves_train_list.flush(); file_waves_train_list.close();
	file_waves_test_list.flush(); file_waves_test_list.close();
	file_train_scp_list.flush(); file_train_scp_list.close();
	file_test_scp_list.flush(); file_test_scp_list.close();
	file_train_txt_list.flush(); file_train_txt_list.close();
	file_test_txt_list.flush(); file_test_txt_list.close();
	file_full_txt.flush(); file_full_txt.close();

	//arpa start -----
	//Make an arpa language model from the full text
	fs::path pth_task_arpabo_input(voicebridgeParams.pth_project_input / voicebridgeParams.task_arpabo_name);
	//backup language model if already exists
	if (fs::exists(pth_task_arpabo_input))
	{
		LOGTW_INFO << "Creating backup of language model...";
		try {
			if (Zip(pth_task_arpabo_input) < 0) {
				LOGTW_ERROR << " Could not backup " << pth_task_arpabo_input.string() << ". Please make a backup manually and then delete the file.";
				return -1;
			}
			//rename the archive - append the current time
			std::stringstream sT;
			auto t = std::time(nullptr);
			auto tm = *std::localtime(&t);
			sT << std::put_time(&tm, "%Y%m%d-%H%M%S");
			fs::path newpath(voicebridgeParams.pth_project_input / (pth_task_arpabo_input.filename().stem().string() + sT.str() + ".zip"));
			if (fs::exists(newpath))
				fs::remove(newpath);
			fs::rename(voicebridgeParams.pth_project_input / (pth_task_arpabo_input.filename().stem().string() + ".zip"), newpath);
			LOGTW_INFO << "Language model succesfully backed up to " << newpath;
		}
		catch (const std::exception&) {
			LOGTW_ERROR << " Could not backup " << pth_task_arpabo_input.string() << ". Please make a backup manually and then delete the file.";
			return -1;
		}
	}
	//create LM
	string_vec options;
	options.push_back("-text");
	options.push_back((voicebridgeParams.pth_data / "full_text.txt").string());
	//save arpa lm
	options.push_back("-wl");
	options.push_back(pth_task_arpabo_input.string());
	//set n in n-gram
	options.push_back("-o");
	options.push_back(std::to_string(orderngram));
	//save the vocab also
	options.push_back("-wv");
	options.push_back((voicebridgeParams.pth_data / "vocab.txt.temp").string());
	//
	StrVec2Arg args(options);
	if (EstimateNgram(args.argc(), args.argv()) < 0) {
		LOGTW_ERROR << "Could not create arpa language model from " << (voicebridgeParams.pth_data / "full_text.txt").string();
		return -1;
	}
	//arpa end ---

	//NOTE: the vocab returned from EstimateNgram() must be cleaned from unwanted symbols
	try	{
		fs::ifstream ifs_vocab(voicebridgeParams.pth_data / "vocab.txt.temp");
		fs::ofstream ofs_vocab(voicebridgeParams.pth_data / "vocab.txt", std::ios::binary | std::ios::out);
		std::string line;
		while (std::getline(ifs_vocab, line)) {
			if (line.find("</s>") == std::string::npos && line.find("<s>") == std::string::npos)
				ofs_vocab << line << "\n";
		}
		ofs_vocab.flush(); ofs_vocab.close();
		ifs_vocab.close();
		fs::remove(voicebridgeParams.pth_data / "vocab.txt.temp");
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << ex.what();
		return -1;
	}
	
	fs::path pth_task_arpabo_target(voicebridgeParams.pth_local / "lm_tg.arpa");
	fs::path pth_train_base(voicebridgeParams.pth_data / voicebridgeParams.train_base_name);
	fs::path pth_test_base(voicebridgeParams.pth_data / voicebridgeParams.test_base_name);
	fs::path pth_train_base_wav(pth_train_base / "wav.scp");
	fs::path pth_test_base_wav(pth_test_base / "wav.scp");
	fs::path pth_train_base_txt(pth_train_base / "text");
	fs::path pth_test_base_txt(pth_test_base / "text");
	try
	{
		fs::copy_file(pth_task_arpabo_input, pth_task_arpabo_target, fs::copy_option::overwrite_if_exists);
		if (!fs::exists(pth_train_base)) fs::create_directories(pth_train_base);
		if (!fs::exists(pth_test_base)) fs::create_directories(pth_test_base);
		fs::copy_file(pth_train_scp_list, pth_train_base_wav, fs::copy_option::overwrite_if_exists);
		fs::copy_file(pth_test_scp_list, pth_test_base_wav, fs::copy_option::overwrite_if_exists);
		fs::copy_file(pth_train_txt_list, pth_train_base_txt, fs::copy_option::overwrite_if_exists);
		fs::copy_file(pth_test_txt_list, pth_test_base_txt, fs::copy_option::overwrite_if_exists);
	}
	catch (std::exception const& e)
	{
		LOGTW_FATALERROR << " " << e.what() << ".";
		return -1;
	}
	catch (...)
	{
		LOGTW_FATALERROR << " Unknown Error.";
		return -1;
	}

	//For each utterance, mark which speaker spoke it. 
	//<utt_id> <speaker_id>
	//e.g. 0_0_1_0_1_0_1_1 speakername
	//we have only one speaker in this example, using <utt_id> as speaker_id
	fs::path pth_train_base_utt2spk(pth_train_base / "utt2spk");
	fs::path pth_test_base_utt2spk(pth_test_base / "utt2spk");
	//
	if(SaveUtt2Spk(pth_train_base_utt2spk, pth_train_scp_list, idtype) < 0)  return -1;
	if(SaveUtt2Spk(pth_test_base_utt2spk, pth_test_scp_list, idtype) < 0)  return -1;

	//convert the utt2spk file to a spk2utt file ----------------------------------->
	//Simply inverse indexed utt2spk (<speaker_id> <all_hier_utterences>)

	//NOTE: could do this easier but doing the conversion because maybe it is needed later
	fs::path pth_train_base_spk2utt(pth_train_base / "spk2utt");
	fs::path pth_test_base_spk2utt(pth_test_base / "spk2utt");
	fs::ofstream file_train_base_spk2utt(pth_train_base_spk2utt, std::ios::binary);
	fs::ofstream file_test_base_spk2utt(pth_test_base_spk2utt, std::ios::binary);
	if (!file_train_base_spk2utt) {
		LOGTW_ERROR << " can't open output file: " << pth_train_base_spk2utt.string() << ".";
		return -1;
	}
	if (!file_test_base_spk2utt) {
		LOGTW_ERROR << " can't open output file: " << pth_test_base_spk2utt.string() << ".";
		return -1;
	}

	try
	{
		std::map<std::string, std::string> map; //sorted and unique!

		//TRAIN ------------------
		StringTable tableTrain_utt2spk = readData(pth_train_base_utt2spk.string());
		StringTable tableTest_utt2spk = readData(pth_test_base_utt2spk.string());
		string_vec speakers;
		using spk2utt2_map = std::unordered_map<std::string, string_vec>;
		spk2utt2_map hashtable;

		for (StringTable::const_iterator it(tableTrain_utt2spk.begin()), it_end(tableTrain_utt2spk.end()); it != it_end; ++it)
		{
			if ((*it).size() != 2) throw std::runtime_error("There should be 2 columns in the utt2spk file!");
			if (std::find(speakers.begin(), speakers.end(), (*it)[1]) == speakers.end())
			{ //not found, add it
				speakers.push_back((*it)[1]);
			}
			spk2utt2_map::iterator itm = hashtable.find((*it)[1]);
			if (itm == hashtable.end())
			{//did not find
				std::vector<std::string> uttvec;
				uttvec.push_back((*it)[0]);
				hashtable.emplace((*it)[1], uttvec);
			}
			else {
				itm->second.push_back((*it)[0]);
			}
		}
		for (string_vec::const_iterator it(speakers.begin()), it_end(speakers.end()); it != it_end; ++it)
		{
			spk2utt2_map::iterator itm = hashtable.find(*it);
			if (itm != hashtable.end()) {
				std::string s;
				int c = 0;
				for (string_vec::const_iterator itu(itm->second.begin()), itu_end(itm->second.end()); itu != itu_end; ++itu)
				{
					if(c>0) s.append(" ");
					s.append(*itu);
					c++;
				}
				map.emplace(*it, s);
			}
		}

		//save sorted on first field
		for (auto & pair : map)
		{
			file_train_base_spk2utt << pair.first << " " << pair.second << '\n';
		}
		file_train_base_spk2utt.flush(); file_train_base_spk2utt.close();

		//TEST ---------------------
		speakers.clear();
		hashtable.clear();
		map.clear();
		for (StringTable::const_iterator it(tableTest_utt2spk.begin()), it_end(tableTest_utt2spk.end()); it != it_end; ++it)
		{
			if ((*it).size() != 2) throw std::runtime_error("There should be 2 columns in the utt2spk file!");
			if (std::find(speakers.begin(), speakers.end(), (*it)[1]) == speakers.end())
			{ //not found, add it
				speakers.push_back((*it)[1]);
			}
			spk2utt2_map::iterator itm = hashtable.find((*it)[1]);
			if (itm == hashtable.end())
			{//did not find
				std::vector<std::string> uttvec;
				uttvec.push_back((*it)[0]);
				hashtable.emplace((*it)[1], uttvec);
			}
			else {
				itm->second.push_back((*it)[0]);
			}
		}
		for (string_vec::const_iterator it(speakers.begin()), it_end(speakers.end()); it != it_end; ++it)
		{
			spk2utt2_map::iterator itm = hashtable.find(*it);
			if (itm != hashtable.end()) {
				std::string s;
				int c = 0;
				for (string_vec::const_iterator itu(itm->second.begin()), itu_end(itm->second.end()); itu != itu_end; ++itu)
				{
					if (c>0) s.append(" ");
					s.append(*itu);
					c++;
				}
				map.emplace(*it, s);
			}
		}
		//save sorted on first field
		for (auto & pair : map)
		{
			file_test_base_spk2utt << pair.first << " " << pair.second << '\n';
		}
		file_test_base_spk2utt.flush(); file_test_base_spk2utt.close();
	}
	catch (std::exception const& e)
	{
		LOGTW_FATALERROR << " " << e.what() << ".";
		file_test_base_spk2utt.close();
		file_train_base_spk2utt.close();
		return -1;
	}
	catch (...)
	{
		LOGTW_FATALERROR << " Unknown Error.";
		file_test_base_spk2utt.close();
		file_train_base_spk2utt.close();
		return -1;
	}	
	
	//<----------------------------------------------------- convert the utt2spk file to a spk2utt file

	LOGTW_INFO << "Data preparation succeeded!";
	return 0;
}

