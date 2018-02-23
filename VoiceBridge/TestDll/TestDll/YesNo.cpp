/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/

/*
INFO
====

LOGGING:
	The VoiceBridge logging mechanism works as follows:
	- use LOGTW_INFO, LOGTW_WARNING, LOGTW_ERROR, LOGTW_FATALERROR, LOGTW_DEBUG to write to the global log file and to the
		screen in case of a console app (std::cout).
	- number formating must be set for each invocation of the log object or macro. 
	  Example: LOGTW_INFO << "double " << std::fixed << std::setprecision(2) << 25.2365874;
	- old Kaldi messages: 
		- KALDI_ERR		=> goes to LOGTW_ERROR
		- KALDI_WARN	=> goes to LOGTW_WARNING
		- KALDI_LOG		=> goes to local log file or to LOGTW_INFO if local log is not defined
		- KALDI_VLOG(v) => goes to local log file or to LOGTW_INFO if local log is not defined
		  , where local log file means the log file to store spesific information from special modules 
		  e.g. training.
*/

#include "ExamplesUtil.h"


//main program
int TestYesNo()
{
	//Initialize Parameters
	//set project directory
	wchar_t buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	fs::path exepath(buffer);
	//use relative path, will work even if the source code is moved
	fs::path project(exepath.branch_path() / "../../../../../VoiceBridgeProjects/YesNo");
	//canonical path will normalize the path and remove "..\"
	project = fs::canonical(project).string();
	bool ret = voicebridgeParams.Init("train_yesno", "test_yesno", project.string(),
		(project / "input").string(),
		(project / "waves_yesno").string());
	if (!ret) {
		LOGTW_ERROR << "Can not find input data.";
		std::getchar();
		return -1;
	}

	fs::path training_dir(voicebridgeParams.pth_data / voicebridgeParams.train_base_name);
	fs::path test_dir(voicebridgeParams.pth_data / voicebridgeParams.test_base_name);

	//in case the data did not change the model will not be retrained unless it is forced with this option
	bool FORCE_RETRAIN_MODEL = false;

	//init general app level log 
	fs::path general_log(voicebridgeParams.pth_project_base / "General.log");
	oTwinLog.init(general_log.string());
	LOGTW_INFO << "***************************************";
	LOGTW_INFO << "* WELCOME TO VOICEBRIDGE FOR WINDOWS! *";
	LOGTW_INFO << "***************************************";

	//try to determine the number of hardware threads supported (cores x processors)	
	//NOTE: you could also decrease this number by one if you would want to leave one core for an UI thread
	int numthreads = concurentThreadsSupported;
	if (numthreads < 1) numthreads = 1;

	//set which language models we want for the test
	std::vector<std::string> lms;
	//NOTE: there can be several language models. Here we just use one LM designated with 'tg'. Other models could be used and
	//		then the directory name would be data\\lang_test_{lm} where {lm} is replaced with the LM id.
	lms.push_back("tg");

	//check if the model needs to be retrained
	//NOTE: We use here model names similar to the names used in Kaldi in order to make it easier for Kaldi users to 
	//		understand the code.
	//		All data structures will also be the same as in Kaldi except that VoiceBridge does not compress files.
	bool needToRetrainModel =
		FORCE_RETRAIN_MODEL ||
		NeedToRetrainModel(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "mono0a",
			voicebridgeParams.pth_project_input,
			voicebridgeParams.waves_dir,
			voicebridgeParams.pth_project_base / "conf" / "train.conf");

	if (needToRetrainModel)
	{
		//Prepare data ----------------------------------------------------------------------------------------------------------

		/*
		PrepareData : this steps prepares the training and test data and also generates automatically an
		arpra n-gram language model if needed. The language model is made from the full text extracted
		from the transcriptions. This step saves also a 'vocab.txt' with all unique words
		used in the text which will be used for making the lexicon with the pronunciations.
		*/
		LOGTW_INFO << "Preparing data...";
		/*
			NOTE: The percentage of training data (the rest of the data will be used for testing), the transcription
			file extension and the idtype are the parameters.
			All necessary data and directory structure will be created automatically for you by this function.

			NOTE: idtype parameter to PrepareData(): it can be 0, 1 or >1; when 0 then the directory name in which the
			wav file is placed will be used as speaker id; when 1 each wav file name (without extension) will be used
			as speaker id (=no speaker separation); when > 1 the first idtype number of characters from the file 
			name will be used as speaker id. The default value is 1!
		*/
		if (PrepareData(90, ".wav.trn") < 0) { 
			LOGTW_ERROR << "***********************************";
			LOGTW_ERROR << "* Error while preparing the data! *";
			LOGTW_ERROR << "***********************************";
			std::getchar();
			return -1;
		}

		/*
		PrepareDict : this step prepares the pronunciation lexicon and all dictionary files automatically. It expects
		a reference lexicon and from that, if necessary, it trains a pronunciation model which can be used
		to determine all pronunciations in the project. You can also add silence phones to the dictionary.
		*/
		LOGTW_INFO << "Preparing dictionary...";		
		fs::path refDict(""); //NOTE: empty means that we have a lexicon already and we do not need to create it!
							  //fs::path refDict(voicebridgeParams.pth_project_input / "cmudict.dict"); 
		std::map<std::string, std::string> optsilphones = { { "<SIL>","SIL" } };
		std::map<std::string, std::string> silphones = { { "<SIL>","SIL" } };
		if (PrepareDict(refDict, silphones, optsilphones) < 0) {
			LOGTW_ERROR << "*****************************************";
			LOGTW_ERROR << "* Error while preparing the dictionary! *";
			LOGTW_ERROR << "*****************************************";
			std::getchar();
			return -1;
		}

		//Prepare language features
		LOGTW_INFO << "Preparing language features...";
		//@test if (PrepareLang(true, "", "D:\\_WORK2\\KALDI\\Kaldi_YesNo\\_Project\\input\\test_phonesymboltable.txt", "") < 0)
		if (PrepareLang(false, "", "", "") < 0)
		{
			LOGTW_ERROR << "*****************************************";
			LOGTW_ERROR << "*  Error while preparing the language!  *";
			LOGTW_ERROR << "*****************************************";
			std::getchar();
			return -1;
		}

		//Prepare language models for test
		LOGTW_INFO << "Preparing language models for test...";
		if (PrepareTestLms(lms) < 0)
		{
			LOGTW_ERROR << "*****************************************";
			LOGTW_ERROR << "* Error while preparing language models!*";
			LOGTW_ERROR << "*****************************************";
			std::getchar();
			return -1;
		}

		/*
			Feature extraction 
			NOTE: mfcc.conf may contain extra parameters!
		*/
		LOGTW_INFO << "\n\n";
		LOGTW_INFO << "Starting MFCC features extraction...";
		if (MakeMfcc(training_dir, voicebridgeParams.pth_project_base / "conf\\mfcc.conf", numthreads) < 0 ||
			MakeMfcc(test_dir, voicebridgeParams.pth_project_base / "conf\\mfcc.conf", numthreads) < 0)
		{
			LOGTW_ERROR << "Feature extraction failed.";
			std::getchar();
			return -1;
		}

		/*
			NOTE: The configuration file format:

			--param1=value
			--param2=value
			--param3=value

			Notice the name of the parameter proceeded by -- and then there is no space between the equal sign and
			the param name and value! Each parameter is on a new line.
		*/

		/*
			Cepstral mean and variance statistics.
		*/
		if (ComputeCmvnStats(training_dir) < 0 ||
			ComputeCmvnStats(test_dir) < 0)
		{
			LOGTW_ERROR << "Feature extraction failed at computing cepstral mean and variance statistics.";
			std::getchar();
			return -1;
		}

		/*
			Fix data directory
		*/
		if (FixDataDir(training_dir) < 0 ||
			FixDataDir(test_dir) < 0)
		{
			LOGTW_ERROR << "Feature extraction failed at fixing data directory.";
			std::getchar();
			return -1;
		}

		/*
			Train monophone model
			NOTE: train.conf may contain extra training parameters!
		*/
		if (TrainGmmMono(training_dir, voicebridgeParams.pth_lang, training_dir / "mono0a", voicebridgeParams.pth_project_base / "conf\\train.conf", numthreads) < 0)
		{
			LOGTW_ERROR << "Training failed.";
			std::getchar();
			return -1;
		}

		/*
			Graph compilation : compile the graph for each language model
		*/
		for (std::string lm : lms) {
			if (MkGraph(voicebridgeParams.pth_data / ("lang_test_" + lm), training_dir / "mono0a", training_dir / "mono0a" / ("graph_" + lm)) < 0)
			{
				LOGTW_ERROR << "Graph compilation failed for language model " << lm;
				std::getchar();
				return -1;
			}
		}

	} ///if (needToRetrainModel
	else {
		LOGTW_INFO << "Recently trained model found. Skipping training step. (NOTE: in case you want to force to retrain the model then open and save one of the input or config files.)";
	}

	//Decode ----------------------------------------------------------------------------------------------------------

	// Decoding of each language model
	UMAPSS wer_ref_filter;						//ref filter NOTE: can be empty but must be defined!
	UMAPSS wer_hyp_filter; 						//hyp filter NOTE: can be empty but must be defined!
	for (std::string lm : lms) 
	{
		LOGTW_INFO << "\n\n";
		LOGTW_INFO << "Decoding language model " << lm << "...";
		if (Decode(
			training_dir / "mono0a" / ("graph_" + lm),									//graph_dir
			voicebridgeParams.pth_data / voicebridgeParams.test_base_name,				//data_dir
			training_dir / "mono0a" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm),	//decode_dir
			training_dir / "mono0a" / "final.mdl",					//when not specified "final.mdl" is taken automatically
			"",														//trans_dir is the directory to find fMLLR transforms; this option won't normally be used, but it can be used if you want to supply existing fMLLR transforms when decoding			
			wer_ref_filter,											//
			wer_hyp_filter,											//
			"",														//iteration of model to test e.g. 'final', if the model is given then this option is not needed
			numthreads												//the number of parallel threads to use in the decoding; must be the same as in the data preparation
																	//all other parameters are used with threir default values
		) < 0)
		{
			LOGTW_ERROR << "Decoding failed for language model " << lm;
			std::getchar();
			return -1;
		}
	}

	//---------------------------------------------------------------------------------------------------------------------------------
	//NOTE: at this point we have a monophone model working properly with a given accuracy. It is still possible to improve the
	//		accuracy with 2-5% with a more sophisticated model trained based on the monophone model.
	//---------------------------------------------------------------------------------------------------------------------------------

	//pause the console application
	LOGTW_INFO << "\n\n";
	LOGTW_INFO << "*****************";
	LOGTW_INFO << "**** ALL OK! ****";
	LOGTW_INFO << "*****************";
	std::getchar();

	return 0;
}
