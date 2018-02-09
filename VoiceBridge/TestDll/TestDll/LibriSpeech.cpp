/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/
#include "ExamplesUtil.h"

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
		  , where local log file means the log file to store specific information from special modules 
		  e.g. training.
*/

int TestLibriSpeech_TrainMonoGmm(std::vector<std::string> lms);
int TestLibriSpeech_TrainDelta(std::vector<std::string> lms);
int TestLibriSpeech_TrainLDAMLLT(std::vector<std::string> lms);
int TestLibriSpeech_TrainLDAMLLTSAT(std::vector<std::string> lms);
int TestLibriSpeech_TrainDeltaSAT(std::vector<std::string> lms);

/*
	This sequence of commands will train a DELTA+SAT model. You can stop at any model if you do not want to go further with
	the extension of the model. Each step is based on the former step thus non of the former steps can be omitted.
	Each model is trained and fully tested. The results are shown on the screen and in the global (and per task) log files.
*/
int TestLibriSpeech()
{
	LOGTW_INFO << "***************************************";
	LOGTW_INFO << "* WELCOME TO VOICEBRIDGE FOR WINDOWS! *";
	LOGTW_INFO << "***************************************";
	LOGTW_INFO << "\n\n";

	int ret = 0;

	//Initialize Parameters

	//set project directory
	wchar_t buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	fs::path exepath(buffer);
	//use relative path, will work even if the source code is moved
	fs::path project(exepath.branch_path() / "../../../../../VoiceBridgeProjects/LibriSpeech");
	//canonical path will normalize the path and remove "..\"
	project = fs::canonical(project).string();
	//init global Params
	bool bret = voicebridgeParams.Init("train_librispeech", "test_librispeech", project.string(),
		(project / "input").string(),
		(project / "LibriSpeech").string(),
		"<UNK>");
	if (!bret) {
		LOGTW_ERROR << "Can not find input data.";
		ret = -1;
	}

	//init general app level log 
	fs::path general_log(voicebridgeParams.pth_project_base / "General.log");
	oTwinLog.init(general_log.string());

	//set which language models we want for the test
	std::vector<std::string> lms;
	//NOTE: there can be several language models. 
	//		the directory name is data/lang_test_{lm} where {lm} is replaced with the LM id.
	lms.push_back("tg");
	
	//this trains a monophone model
	if (ret>-1) ret = TestLibriSpeech_TrainMonoGmm(lms);

	//this trains a delta + delta-delta triphone model built upon the monophone model
	if(ret>-1) ret = TestLibriSpeech_TrainDelta(lms);

	//NOTE: you could check the accuracy of the delta and lda+mllt models and use the one which has better accuracy for subsequent
	//		model building. Here we just choose in the code.
	bool bUseDeltaSAT = true;
	if (bUseDeltaSAT) 
	{
		//this trains a DELTA+SAT model built on delta + delta-delta
		if (ret > -1) ret = TestLibriSpeech_TrainDeltaSAT(lms);

		//NOTE: in case of the LibriSpeech example (and used example data) this path gives the best accuracy (WER ~5.9%)
		//		with much less training time (~1/5) than LDA+MLLT+SAT.
	}
	else {
		//this trains an LDA+MLLT model built on the delta + delta-delta model
		if (ret>-1) ret = TestLibriSpeech_TrainLDAMLLT(lms);

		//this trains an LDA+MLLT+SAT model built on the LDA+MLLT model
		if (ret > -1) ret = TestLibriSpeech_TrainLDAMLLTSAT(lms);
	}

	fs::path dir;
	if (bUseDeltaSAT)
		dir = voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri3c";
	else dir = voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri3b";


	//final message
	if (ret > -1) {
		LOGTW_INFO << "\n\n";
		LOGTW_INFO << "*****************";
		LOGTW_INFO << "**** ALL OK! ****";
		LOGTW_INFO << "*****************";
	}
	else {
		LOGTW_INFO << "\n\n";
		LOGTW_INFO << "*****************";
		LOGTW_INFO << "****  ERROR! ****";
		LOGTW_INFO << "*****************";
	}
	std::getchar();
	return ret;
}

/*
	Train a mono gmm model with the LibriSpeech data.

	NOTE: this project is a more realistic project where we create the lexicon also.
	The only input that we expect are the WAV files with the transcriptions. Short WAV files
	with the aligned transcriptions (the text which is spoken) are made before this step.
*/
int TestLibriSpeech_TrainMonoGmm(std::vector<std::string> lms)
{
	fs::path training_dir(voicebridgeParams.pth_data / voicebridgeParams.train_base_name);
	fs::path test_dir(voicebridgeParams.pth_data / voicebridgeParams.test_base_name);

	//in case the data did not change the model will not be retrained unless it is forced with this option
	bool FORCE_RETRAIN_MODEL = false;
	//in case the model did not change the test set will not be decoded again
	bool FORCE_DECODE = false;

	//try to determine the number of hardware threads supported (cores x processors)	
	//NOTE: you could also decrease this number by one if you would want to leave one core for an UI thread
	int numthreads = concurentThreadsSupported;
	if (numthreads < 1) numthreads = 1;

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
		/*
			IMPORTANT NOTE: It is assumed at this point that the transcriptions contain only normalized text. 
							Normalized text do not contain e.g. numbers, dates with numbers, etc. but only
							real text! In a pre-processing step all non-standard words e.g. numbers, dates, etc.
							must be converted to standard words containing only text because the audio is text only.
		*/
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
				  name will be used as speaker id.

			NOTE: We take in this example the directory name as speaker_id because the file names have no standard length.
				  It would be better if the file names were prepared with standard length and then e.g. the 
				  speaker id's could be extracted automatically as e.g. the first 10 characters... 
				  This will cause a small error on speaker dependent calculations because most probably
				  some directories will contain more than one speaker in LibriSpeech. But using the directory name as
				  speaker id's the accuracy improves with about 1% in this example compared to using the file names only.
		*/
		if (PrepareData(90, ".lab", 4, 0) < 0) {
			LOGTW_ERROR << "***********************************";
			LOGTW_ERROR << "* Error while preparing the data! *";
			LOGTW_ERROR << "***********************************";
			return -1;  
		}

		/*
			PrepareDict : this step prepares the pronunciation lexicon and all dictionary files automatically. It expects
						  a reference lexicon and from that, if necessary, it trains a pronunciation model which can be used
						  to determine all pronunciations in the project. You can also add silence phones to the dictionary.
		*/
		LOGTW_INFO << "Preparing dictionary...";
		fs::path refDict(voicebridgeParams.pth_project_input / "cmudict.dict");
		std::map<std::string, std::string> optsilphones = { {"!SIL","SIL"} };
		std::map<std::string, std::string> silphones = { { "!SIL","SIL" }, { "<SPOKEN_NOISE>","SPN" }, { "<UNK>","SPN" } };
		if (PrepareDict(refDict, silphones, optsilphones) < 0) {
			LOGTW_ERROR << "*****************************************";
			LOGTW_ERROR << "* Error while preparing the dictionary! *";
			LOGTW_ERROR << "*****************************************";
			return -1;
		}

		/*
			PrepareLang : prepares all language features. 
						  NOTE: oov word is set above in voicebridgeParams.Init
		*/
		LOGTW_INFO << "Preparing language features...";
		if (PrepareLang(true, "", "", "") < 0)
		{
			LOGTW_ERROR << "********************************************"; 
			LOGTW_ERROR << "* Error while preparing language features! *";
			LOGTW_ERROR << "********************************************";
			return -1;
		}

		/*
			Prepare language models for test (lms is defined above)
		*/
		LOGTW_INFO << "Preparing language models for test...";
		if (PrepareTestLms(lms) < 0)
		{
			LOGTW_ERROR << "*****************************************";
			LOGTW_ERROR << "* Error while preparing language models!*";
			LOGTW_ERROR << "*****************************************";
			return -1;
		}

		/* 
			Feature extraction
			NOTE: mfcc.conf may contain extra parameters!
		*/
		LOGTW_INFO << "\n\n";
		LOGTW_INFO << "Starting MFCC features extraction...";
		if (MakeMfcc(training_dir, voicebridgeParams.pth_project_base / "conf" / "mfcc.conf", numthreads) < 0 ||
			MakeMfcc(test_dir, voicebridgeParams.pth_project_base / "conf" / "mfcc.conf", numthreads) < 0)
		{
			LOGTW_ERROR << " Feature extraction failed.";
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
			LOGTW_ERROR << " Feature extraction failed at computing cepstral mean and variance statistics.";
			return -1;
		}

		/*
			Fix data directory
		*/
		if (FixDataDir(training_dir) < 0 ||
			FixDataDir(test_dir) < 0)
		{
			LOGTW_ERROR << " Feature extraction failed at fixing data directory.";
			return -1;
		}

		/*
			Train monophone model
			NOTE: train.conf may contain extra training parameters!
		*/
		if (TrainGmmMono(training_dir, voicebridgeParams.pth_lang, training_dir / "mono0a", voicebridgeParams.pth_project_base / "conf" / "train.conf", numthreads) < 0)
		{
			LOGTW_ERROR << "Training failed.";
			return -1;
		}
	} ///if (needToRetrainModel
	else {
		LOGTW_INFO << "\n";
		LOGTW_INFO << "Monophone model found. Skipping training.";
	}


	/*
		Decoding of each language model

		NOTE: decoding means that we use the above trained model on the test set or with other words we identify the spoken text
			  and then in the scoring we determine the accuracy of the model by comparing the results to the real text.
	*/
	UMAPSS wer_ref_filter;						//ref filter NOTE: can be empty but must be defined!
	UMAPSS wer_hyp_filter; 						//hyp filter NOTE: can be empty but must be defined!
	for (std::string lm : lms) 
	{
		//NOTE: in order to decide if we need to decode the test set we check the creation date of the first lattice file: (decode_dir / "lat.JOBID")
		//		and compare it to the creation date of the model file. You may also use other means to decide this.
		bool needToDecode =
			FORCE_DECODE ||
			NeedToDecode(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "mono0a" / "final.mdl",
						 training_dir / "mono0a" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm));

		if (needToDecode) {
			/*
			Graph compilation : compile the graph for each language model
			*/
			if (MkGraph(voicebridgeParams.pth_data / ("lang_test_" + lm), training_dir / "mono0a", training_dir / "mono0a" / ("graph_" + lm)) < 0)
			{
				LOGTW_ERROR << "Graph compilation failed for language model " << lm;
				return -1;
			}

			LOGTW_INFO << "\n\n";
			LOGTW_INFO << "Decoding language model " << lm << "...";
			if (Decode(
				training_dir / "mono0a" / ("graph_" + lm),												//graph_dir
				voicebridgeParams.pth_data / voicebridgeParams.test_base_name,							//data_dir
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
				return -1;
			}
		}
	}

	//---------------------------------------------------------------------------------------------------------------------------------
	//NOTE: at this point we have a monophone model working properly with a given accuracy. It is still possible to improve the
	//		accuracy with 2-5% with a more sophisticated model trained based on the monophone model.
	//		You may select a subset of the data to train the new model on but we will use the same set for subsequent training here.
	//---------------------------------------------------------------------------------------------------------------------------------

	LOGTW_INFO << "\n\n";
	LOGTW_INFO << "***********************";
	LOGTW_INFO << "**** MONOPHONE OK! ****";
	LOGTW_INFO << "***********************";

	return 0;
}


int TestLibriSpeech_TrainDelta(std::vector<std::string> lms)
{
	fs::path training_dir(voicebridgeParams.pth_data / voicebridgeParams.train_base_name);
	fs::path test_dir(voicebridgeParams.pth_data / voicebridgeParams.test_base_name);

	//in case the data did not change the model will not be retrained unless it is forced with this option
	bool FORCE_RETRAIN_MODEL = false;
	//in case the model did not change the test set will not be decoded again
	bool FORCE_DECODE = false;

	//try to determine the number of hardware threads supported (cores x processors)	
	int numthreads = concurentThreadsSupported;
	if (numthreads < 1) numthreads = 1;

	//------------------------------------------------------------------------------------------------------------------------->
	//	delta + delta-delta triphone model built upon the monophone model
	//-------------------------------------------------------------------------------------------------------------------------

	//check base model requirements
	if (!fs::exists(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "mono0a" / "final.mdl")) {
		LOGTW_ERROR << "A monophone base model is needed for this step.";
		return -1;
	}

	bool needToRetrainModel =
		FORCE_RETRAIN_MODEL ||
		NeedToRetrainModel(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri1",
			voicebridgeParams.pth_project_input,
			voicebridgeParams.waves_dir,
			voicebridgeParams.pth_project_base / "conf" / "tri1.conf",
			voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "mono0a"); //base model

	if (needToRetrainModel) {
		//Train a delta + delta-delta triphone system
		int reti = AlignSi(training_dir, voicebridgeParams.pth_lang, training_dir / "mono0a", training_dir / "mono_ali", numthreads, 1.25);
		if (reti < 0)
		{
			LOGTW_ERROR << "Training failed for delta + delta-delta triphone system (AlignSi).";
			return -1;
		}
		reti = TrainDeltas(training_dir, voicebridgeParams.pth_lang, training_dir / "mono_ali", training_dir / "tri1", numthreads, voicebridgeParams.pth_project_base / "conf" / "tri1.conf", 1.25);
		if (reti < 0)
		{
			LOGTW_ERROR << "Training failed for delta + delta-delta triphone system (TrainDeltas).";
			return -1;
		}
	}
	else {
		LOGTW_INFO << "\n";
		LOGTW_INFO << "delta + delta-delta model found. Skipping training.";
	}

	//Decode using the tri1 model
	/*
	Decoding of each language model

	NOTE: decoding means that we use the above trained model on the test set or with other words we identify the spoken text
	and then in the scoring we determine the accuracy of the model by comparing the results to the real text.
	*/
	for (std::string lm : lms)
	{
		//NOTE: in order to decide if we need to decode the test set we check the creation date of the first lattice file: (decode_dir / "lat.JOBID")
		//		and compare it to the creation date of the model file.
		bool needToDecode =
			FORCE_DECODE ||
			NeedToDecode(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri1" / "final.mdl",
				training_dir / "tri1" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm));

		if (needToDecode)
		{
			/*
			Graph compilation : compile the graph for the language model
			*/
			if (MkGraph(voicebridgeParams.pth_data / ("lang_test_" + lm), training_dir / "tri1", training_dir / "tri1" / ("graph_" + lm)) < 0)
			{
				LOGTW_ERROR << "Graph compilation failed for language model " << lm;
				return -1;
			}

			UMAPSS wer_ref_filter;						//ref filter NOTE: can be empty but must be defined!
			UMAPSS wer_hyp_filter; 						//hyp filter NOTE: can be empty but must be defined!

			LOGTW_INFO << "\n\n";
			LOGTW_INFO << "Decoding language model " << lm << "...";
			if (Decode(
				training_dir / "tri1" / ("graph_" + lm),											//graph_dir
				voicebridgeParams.pth_data / voicebridgeParams.test_base_name,						//data_dir
				training_dir / "tri1" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm),	//decode_dir
				training_dir / "tri1" / "final.mdl",					//when not specified "final.mdl" is taken automatically
				"",														//trans_dir is the directory to find fMLLR transforms; this option won't normally be used, but it can be used if you want to supply existing fMLLR transforms when decoding			
				wer_ref_filter,											//
				wer_hyp_filter,											//
				"",														//iteration of model to test e.g. 'final', if the model is given then this option is not needed
				numthreads												//the number of parallel threads to use in the decoding; must be the same as in the data preparation
																		//all other parameters are used with threir default values
			) < 0)
			{
				LOGTW_ERROR << "Decoding failed for language model " << lm;
				return -1;
			}
		}
	}

	/*
	LM RESCORING:
	At this point you could do LM rescoring.
	LM rescoring is replacing LM scores on a lattice with LM scores from a more complicated LM.
	The reason behind it is that, decoding with a large LM directly could be slower,
	and therefore you could first decode with a small LM and then rescore with a large LM.
	*/
	/* NOTE: Const-arpa LM rescoring is the recommended method at the moment.
	In order to be able to do this we must first build a constant arpa language model from our normal arpa model
	and then use a constant Arpa LmRescore routine to do the rescoring.
	*/

	//<------------------------------------------------------------------------------------- delta + delta-delta triphone model

	LOGTW_INFO << "\n\n";
	LOGTW_INFO << "***********************";
	LOGTW_INFO << "****   DELTA OK!   ****";
	LOGTW_INFO << "***********************";

	return 0;
}

int TestLibriSpeech_TrainLDAMLLT(std::vector<std::string> lms)
{
	fs::path training_dir(voicebridgeParams.pth_data / voicebridgeParams.train_base_name);
	fs::path test_dir(voicebridgeParams.pth_data / voicebridgeParams.test_base_name);

	//in case the data did not change the model will not be retrained unless it is forced with this option
	bool FORCE_RETRAIN_MODEL = false;
	//in case the model did not change the test set will not be decoded again
	bool FORCE_DECODE = false;

	//try to determine the number of hardware threads supported (cores x processors)	
	int numthreads = concurentThreadsSupported;
	if (numthreads < 1) numthreads = 1;

	//------------------------------------------------------------------------------------------------------------------------->
	//	LDA+MLLT model built upon the delta + delta-delta model
	//-------------------------------------------------------------------------------------------------------------------------

	//check base model requirements
	if (!fs::exists(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri1" / "final.mdl")) {
		LOGTW_ERROR << "A delta + delta-delta base model is needed for this step.";
		return -1;
	}

	/*
	LDA+MLLT refers to the way we transform the features after computing the MFCCs: we splice across several frames,
	reduce the dimension (to 40 by default) using Linear Discriminant Analysis), and then later estimate, over
	multiple iterations, a diagonalizing transform known as MLLT or CTC.
	*/
	bool needToRetrainModel =
		FORCE_RETRAIN_MODEL ||
		NeedToRetrainModel(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri2b",
			voicebridgeParams.pth_project_input,
			voicebridgeParams.waves_dir,
			voicebridgeParams.pth_project_base / "conf" / "tri2b.conf",
			voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri1"); //base model

	if (needToRetrainModel) {
		//Train a LDA+MLLT model
		int reti = AlignSi(training_dir, voicebridgeParams.pth_lang, training_dir / "tri1", training_dir / "tri1_ali", numthreads, 1.0);
		if (reti < 0)
		{
			LOGTW_ERROR << "Training failed for LDA+MLLT model (AlignSi).";
			return -1;
		}

		/*
			IMPORTANT NOTE: in the configuration file you can set a parameter containing several other parameters in the following way:

			--splice-opts=--left-context=3 --right-context=3

			Here the parameter is 'splice-opts' which contains two other parameters.
			1. notice that there is no space between the equal signs and the properties
			2. after the first equal sign there is no " or ' and the properties are listed with a space between them!
		*/

		//Train
		reti = TrainLdaMllt(training_dir, voicebridgeParams.pth_lang, training_dir / "tri1_ali", training_dir / "tri2b", numthreads, voicebridgeParams.pth_project_base / "conf" / "tri2b.conf", 1.0, 2500, 15000);
		if (reti < 0)
		{
			LOGTW_ERROR << "Training failed for LDA+MLLT model (TrainLdaMllt).";
			return -1;
		}
	}
	else {
		LOGTW_INFO << "\n";
		LOGTW_INFO << "LDA+MLLT model found. Skipping training.";
	}

	//Decode using the tri2b model
	/*
	Decoding of each language model

	NOTE: decoding means that we use the above trained model on the test set or with other words we identify the spoken text
	and then in the scoring we determine the accuracy of the model by comparing the results to the real text.
	*/
	for (std::string lm : lms)
	{
		//NOTE: in order to decide if we need to decode the test set we check the creation date of the first lattice file: (decode_dir / "lat.JOBID")
		//		and compare it to the creation date of the model file.
		bool needToDecode =
			FORCE_DECODE ||
			NeedToDecode(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri2b" / "final.mdl",
				training_dir / "tri2b" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm));

		if (needToDecode)
		{
			/*
			Graph compilation : compile the graph for the language model
			*/
			if (MkGraph(voicebridgeParams.pth_data / ("lang_test_" + lm), training_dir / "tri2b", training_dir / "tri2b" / ("graph_" + lm)) < 0)
			{
				LOGTW_ERROR << "Graph compilation failed for language model " << lm;
				return -1;
			}

			UMAPSS wer_ref_filter;						//ref filter NOTE: can be empty but must be defined!
			UMAPSS wer_hyp_filter; 						//hyp filter NOTE: can be empty but must be defined!

			LOGTW_INFO << "\n\n";
			LOGTW_INFO << "Decoding language model " << lm << "...";
			if (Decode(
				training_dir / "tri2b" / ("graph_" + lm),											//graph_dir
				voicebridgeParams.pth_data / voicebridgeParams.test_base_name,						//data_dir
				training_dir / "tri2b" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm),	//decode_dir
				training_dir / "tri2b" / "final.mdl",					//when not specified "final.mdl" is taken automatically
				"",														//trans_dir is the directory to find fMLLR transforms; this option won't normally be used, but it can be used if you want to supply existing fMLLR transforms when decoding			
				wer_ref_filter,											//
				wer_hyp_filter,											//
				"",														//iteration of model to test e.g. 'final', if the model is given then this option is not needed
				numthreads												//the number of parallel threads to use in the decoding; must be the same as in the data preparation
																		//all other parameters are used with threir default values
			) < 0)
			{
				LOGTW_ERROR << "Decoding failed for language model " << lm;
				return -1;
			}
		}
	}

	/*
		LM RESCORING: At this point you could do LM rescoring.
	*/

	//<---------------------------------------------------------------------------------------------------------- LDA+MLLT model

	LOGTW_INFO << "\n\n";
	LOGTW_INFO << "***********************";
	LOGTW_INFO << "****  LDA+MLLT OK! ****";
	LOGTW_INFO << "***********************";

	return 0;
}

int TestLibriSpeech_TrainLDAMLLTSAT(std::vector<std::string> lms)
{
	fs::path training_dir(voicebridgeParams.pth_data / voicebridgeParams.train_base_name);
	fs::path test_dir(voicebridgeParams.pth_data / voicebridgeParams.test_base_name);

	//in case the data did not change the model will not be retrained unless it is forced with this option
	bool FORCE_RETRAIN_MODEL = false;
	//in case the model did not change the test set will not be decoded again
	bool FORCE_DECODE = false;

	//try to determine the number of hardware threads supported (cores x processors)	
	int numthreads = concurentThreadsSupported;
	if (numthreads < 1) numthreads = 1;

	//------------------------------------------------------------------------------------------------------------------------->
	//	LDA+MLLT+SAT model built upon the LDA+MLLT
	/*
		This does Speaker Adapted Training (SAT), i.e. train on fMLLR-adapted features. It can be done on top of either 
		LDA+MLLT, or delta + delta-delta features. If there are no transforms supplied in the alignment directory, 
		it will estimate transforms itself before building the tree (and in any case, it estimates transforms a number 
		of times during training).
	*/
	//-------------------------------------------------------------------------------------------------------------------------

	//check base model requirements
	if (!fs::exists(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri2b" / "final.mdl")) {
		LOGTW_ERROR << "An LDA+MLLT base model is needed for this step.";
		return -1;
	}

	/*
	LDA+MLLT refers to the way we transform the features after computing the MFCCs: we splice across several frames,
	reduce the dimension (to 40 by default) using Linear Discriminant Analysis), and then later estimate, over
	multiple iterations, a diagonalizing transform known as MLLT or CTC.
	*/
	bool needToRetrainModel =
		FORCE_RETRAIN_MODEL ||
		NeedToRetrainModel(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri3b",
			voicebridgeParams.pth_project_input,
			voicebridgeParams.waves_dir,
			voicebridgeParams.pth_project_base / "conf" / "tri3b.conf",
			voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri2b"); //base model

	if (needToRetrainModel) {
		//Train a LDA+MLLT+SAT model
		int reti = AlignSi(training_dir, voicebridgeParams.pth_lang, training_dir / "tri2b", training_dir / "tri2b_ali", numthreads, 1.0);
		if (reti < 0)
		{
			LOGTW_ERROR << "Training failed for LDA+MLLT+SAT model (AlignSi).";
			return -1;
		}

		/*
		IMPORTANT NOTE: in the configuration file you can set a parameter containing several other parameters in the following way:

		--splice-opts=--left-context=3 --right-context=3

		Here the parameter is 'splice-opts' which contains two other parameters.
		1. notice that there is no space between the equal signs and the properties
		2. after the first equal sign there is no " or ' and the properties are listed with a space between them!
		*/

		//Train
		reti = TrainSat(training_dir, voicebridgeParams.pth_lang,
			training_dir / "tri2b_ali",  
			training_dir / "tri3b", 
			numthreads, 
			voicebridgeParams.pth_project_base / "conf" / "tri3b.conf",
			1.0, 
			2500, 15000);
		if (reti < 0)
		{
			LOGTW_ERROR << "Training failed for LDA+MLLT+SAT model (TrainLdaMlltSat)."; 
			return -1;
		}
	}
	else {
		LOGTW_INFO << "\n";
		LOGTW_INFO << "LDA+MLLT+SAT model found. Skipping training.";		
	}

	//Decode using the tri3b model
	/*
	Decoding of each language model

	NOTE: decoding means that we use the above trained model on the test set or with other words we identify the spoken text
	and then in the scoring we determine the accuracy of the model by comparing the results to the real text.
	*/
	for (std::string lm : lms)
	{
		//NOTE: in order to decide if we need to decode the test set we check the creation date of the first lattice file: (decode_dir / "lat.JOBID")
		//		and compare it to the creation date of the model file.
		bool needToDecode =
			FORCE_DECODE ||
			NeedToDecode(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri3b" / "final.mdl",
				training_dir / "tri3b" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm));

		if (needToDecode)
		{
			/*
			Graph compilation : compile the graph for the language model
			*/
			if (MkGraph(voicebridgeParams.pth_data / ("lang_test_" + lm), training_dir / "tri3b", training_dir / "tri3b" / ("graph_" + lm)) < 0)
			{
				LOGTW_ERROR << "Graph compilation failed for language model " << lm;
				return -1;
			}

			UMAPSS wer_ref_filter;						//ref filter NOTE: can be empty but must be defined!
			UMAPSS wer_hyp_filter; 						//hyp filter NOTE: can be empty but must be defined!

			LOGTW_INFO << "\n\n";
			LOGTW_INFO << "Decoding language model " << lm << "...";

			if (DecodeFmllr(
				training_dir / "tri3b" / ("graph_" + lm),											//graph_dir
				voicebridgeParams.pth_data / voicebridgeParams.test_base_name,						//data_dir
				training_dir / "tri3b" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm),	//decode_dir
				"",				//when not specified "final.mdl" is taken automatically
				"",				//
				"",				//
				numthreads		//the number of parallel threads to use in the decoding; must be the same as in the data preparation
								//all other parameters are used with threir default values
			) < 0)
			{
				LOGTW_ERROR << "Decoding failed for language model " << lm;
				return -1;
			}
		}
	}

	/*
		LM RESCORING: At this point you could do LM rescoring.
	*/

	//<---------------------------------------------------------------------------------------------------------- LDA+MLLT+SAT model

	LOGTW_INFO << "\n\n";
	LOGTW_INFO << "***************************";
	LOGTW_INFO << "****  LDA+MLLT+SAT OK! ****";
	LOGTW_INFO << "***************************";

	return 0;
}


int TestLibriSpeech_TrainDeltaSAT(std::vector<std::string> lms)
{
	fs::path training_dir(voicebridgeParams.pth_data / voicebridgeParams.train_base_name);
	fs::path test_dir(voicebridgeParams.pth_data / voicebridgeParams.test_base_name);

	//in case the data did not change the model will not be retrained unless it is forced with this option
	bool FORCE_RETRAIN_MODEL = false;
	//in case the model did not change the test set will not be decoded again
	bool FORCE_DECODE = false;

	//try to determine the number of hardware threads supported (cores x processors)	
	int numthreads = concurentThreadsSupported;
	if (numthreads < 1) numthreads = 1;

	//------------------------------------------------------------------------------------------------------------------------->
	//	LDA+MLLT+SAT model built upon the delta + delta-delta
	/*
	This does Speaker Adapted Training (SAT), i.e. train on fMLLR-adapted features. It can be done on top of either
	LDA+MLLT, or delta + delta-delta features. If there are no transforms supplied in the alignment directory,
	it will estimate transforms itself before building the tree (and in any case, it estimates transforms a number
	of times during training).
	*/
	//-------------------------------------------------------------------------------------------------------------------------

	//check base model requirements
	if (!fs::exists(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri1" / "final.mdl")) {
		LOGTW_ERROR << "An delta + delta-delta base model is needed for this step.";
		return -1;
	}

	/*
	LDA+MLLT refers to the way we transform the features after computing the MFCCs: we splice across several frames,
	reduce the dimension (to 40 by default) using Linear Discriminant Analysis), and then later estimate, over
	multiple iterations, a diagonalizing transform known as MLLT or CTC.
	*/
	bool needToRetrainModel =
		FORCE_RETRAIN_MODEL ||
		NeedToRetrainModel(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri3c",
			voicebridgeParams.pth_project_input,
			voicebridgeParams.waves_dir,
			voicebridgeParams.pth_project_base / "conf" / "tri3c.conf",
			voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri1"); //base model

	if (needToRetrainModel) {
		//Train a DELTA+SAT model
		int reti = AlignSi(training_dir, voicebridgeParams.pth_lang, training_dir / "tri1", training_dir / "tri1b_ali", numthreads, 1.0);
		if (reti < 0)
		{
			LOGTW_ERROR << "Training failed for DELTA+SAT model (AlignSi).";
			return -1;
		}

		//Train
		reti = TrainSat(training_dir, voicebridgeParams.pth_lang,
			training_dir / "tri1b_ali",
			training_dir / "tri3c",
			numthreads,
			voicebridgeParams.pth_project_base / "conf" / "tri3c.conf",
			1.0, 
			2500, 15000);
		if (reti < 0)
		{
			LOGTW_ERROR << "Training failed for DELTA+SAT model (TrainLdaMlltSat).";
			return -1;
		}
	}
	else {
		LOGTW_INFO << "\n";
		LOGTW_INFO << "DELTA+SAT model found. Skipping training.";
	}

	//Decode using the tri3c model
	/*
	Decoding of each language model

	NOTE: decoding means that we use the above trained model on the test set or with other words we identify the spoken text
	and then in the scoring we determine the accuracy of the model by comparing the results to the real text.
	*/
	for (std::string lm : lms)
	{
		//NOTE: in order to decide if we need to decode the test set we check the creation date of the first lattice file: (decode_dir / "lat.JOBID")
		//		and compare it to the creation date of the model file.
		bool needToDecode =
			FORCE_DECODE ||
			NeedToDecode(voicebridgeParams.pth_data / voicebridgeParams.train_base_name / "tri3c" / "final.mdl",
				training_dir / "tri3c" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm));

		if (needToDecode)
		{
			/*
			Graph compilation : compile the graph for the language model
			*/
			if (MkGraph(voicebridgeParams.pth_data / ("lang_test_" + lm), training_dir / "tri3c", training_dir / "tri3c" / ("graph_" + lm)) < 0)
			{
				LOGTW_ERROR << "Graph compilation failed for language model " << lm;
				return -1;
			}

			UMAPSS wer_ref_filter;						//ref filter NOTE: can be empty but must be defined!
			UMAPSS wer_hyp_filter; 						//hyp filter NOTE: can be empty but must be defined!

			LOGTW_INFO << "\n\n";
			LOGTW_INFO << "Decoding language model " << lm << "...";

			if (DecodeFmllr(
				training_dir / "tri3c" / ("graph_" + lm),											//graph_dir
				voicebridgeParams.pth_data / voicebridgeParams.test_base_name,						//data_dir
				training_dir / "tri3c" / ("decode_" + voicebridgeParams.test_base_name + "_" + lm),	//decode_dir
				"",				//when not specified "final.mdl" is taken automatically
				"",				//
				"",				//
				numthreads		//the number of parallel threads to use in the decoding; must be the same as in the data preparation
								//all other parameters are used with threir default values
			) < 0)
			{
				LOGTW_ERROR << "Decoding failed for language model " << lm;
				return -1;
			}
		}
	}

	/*
		LM RESCORING: At this point you could do LM rescoring.
	*/

	//<---------------------------------------------------------------------------------------------------------- LDA+MLLT+SAT model

	LOGTW_INFO << "\n\n";
	LOGTW_INFO << "***************************";
	LOGTW_INFO << "****   DELTA+SAT OK!   ****";
	LOGTW_INFO << "***************************";

	return 0;
}


