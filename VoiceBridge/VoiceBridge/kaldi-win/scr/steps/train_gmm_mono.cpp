/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"
#include <kaldi-win/utility/strvec2arg.h>

#include "util/common-utils.h" //for ParseOptions

/*
	IMPORTANT NOTES: 
	 - using 'JOBID' in file names which will be replaced by the numerical id of the job (parallel process). The word JOBID
	   should not be in any input file name.
*/

static int ApplyCmvnSequence(int jobid, string_vec options_applycmvn, string_vec options_adddeltas, fs::path sdata, std::string & out);
static void LaunchJobCompileTrainGraphs(
	int argc1, char *argv1[], 
	fs::path symtab, fs::path input_txt, fs::path output_txt, int field_begin, int field_end, std::string smap_oov, 
	std::string fst_out, fs::path log);

static void LaunchJobAlignData(
	int argc1, char *argv1[], //params for AlignEqualCompiled
	int argc2, char *argv2[], //params for GmmAccStatsAli
	int jobid, string_vec options_applycmvn, string_vec options_adddeltas, fs::path sdata, //params for ApplyCmvnSequence
	fs::path log);

static void LaunchJobGmmAlignCompiled(
	int argc1, char *argv1[], //params for GmmAlignCompiled
	int jobid, string_vec options_applycmvn, string_vec options_adddeltas, fs::path sdata, //params for ApplyCmvnSequence
	fs::path log);

static void LaunchJobGmmAccStatsAli(
	int argc1, char *argv1[], //params for GmmAccStatsAli
	int jobid, string_vec options_applycmvn, string_vec options_adddeltas, fs::path sdata, //params for ApplyCmvnSequence
	fs::path log);

//the return values from each thread/job
static std::vector<int> _ret;


/*
Flat start and monophone training, with delta-delta features. This script applies cepstral mean normalization (per speaker).
*/
VOICEBRIDGE_API int TrainGmmMono(
	fs::path datadir,			//data directory		=> data
	fs::path langdir,			//lang directory		=> lang
	fs::path traindir,			//training directory	=> dir
	fs::path config,			//config file path with various options //TODO:... document
	int nj,						//default: 4, number of parallel jobs
	int stage					//stage; can be used to skip some steps done before
)
{
	fs::path logdir = datadir / "log";	
	std::string name = datadir.stem().string();

	LOGTW_INFO << "Starting the training of the monophone system...";

	//Read the config file -------------------------------------------------------------------
	//NOTE: the config file here must be parsed manually because the variables in it are
	//		used directly in the code!
	kaldi::ParseOptions po("");
	//Define and Register the options; default values set
	std::vector<std::string> _scale_opts = { "--transition-scale=1.0","--acoustic-scale=0.1","--self-loop-scale=0.1" };
	std::string scale_opts("");
	int num_iters = 40;
	int max_iter_inc = 30;
	int totgauss = 1000;
	bool careful = false;
	double boost_silence = 1.0;
	std::vector<int> _realign_iters = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38 };
	std::string realign_iters("");
	double power = 0.25;
	bool norm_vars = false;
	std::string cmvn_opts("");
	po.Register("scale-opts", &scale_opts, "Scale options for gmm-align-compiled.");
	po.Register("num-iters", &num_iters, "Number of iterations of training.");
	po.Register("max-iter-inc", &max_iter_inc, "Last iter to increase #Gauss on.");
	po.Register("totgauss", &totgauss, "Target #Gaussians.");
	po.Register("careful", &careful, ".");
	po.Register("boost-silence", &boost_silence, "Factor by which to boost silence likelihoods in alignment.");
	po.Register("realign-iters", &realign_iters, ".");
	po.Register("stage", &stage, ".");
	po.Register("power", &power, "Exponent to determine number of gaussians from occurrence counts.");
	po.Register("norm-vars", &norm_vars, "Deprecated, prefer --cmvn-opts '--norm - vars = false'.");
	po.Register("cmvn_opts", &cmvn_opts, "Can be used to add extra options to cmvn.");
	if (config!="" && fs::exists(config) && !fs::is_empty(config)) 
	{//all parameters will be overwritten with the parameters defined in the config file
		string_vec options;
		options.push_back("--config=" + config.string());
		StrVec2Arg args(options);
		LOGTW_INFO << "Reading configuration file:";
		int ret;
		try {
			ret = po.Read(args.argc(), args.argv());
		}
		catch (const std::exception&)
		{
			LOGTW_ERROR << "Configuration file syntax error. Check " << config.string();
			return -1;
		}
		if (ret < 0) {
			LOGTW_ERROR << "Configuration file syntax error. Check " << config.string();
			return -1;
		}
	}
	//----------------------------------------------------------------------------------------

	//in case the realign-iters is read from the config file it must be transferred to _realign_iters int vector
	try	{
		if (realign_iters != "") {
			_realign_iters.clear();
			strtk::parse(realign_iters, " ", _realign_iters, strtk::split_options::compress_delimiters);
		}
	} catch (const std::exception&) {
		LOGTW_ERROR << "Wrong value for parameter 'realign-iters' in configuration file.";
		return -1;
	}
	//in case the scale-opts is read from the config file it must be transferred to _scale_opts int vector
	try {
		if (scale_opts != "") {
			_scale_opts.clear();
			strtk::parse(scale_opts, " ", _scale_opts, strtk::split_options::compress_delimiters);
		}
	}
	catch (const std::exception&) {
		LOGTW_ERROR << "Wrong value for parameter 'scale-opts' in configuration file.";
		return -1;
	}

	if (CreateDir(traindir / "log", true) < 0) return -1;

	if (CheckFileExistsAndNotEmpty(langdir / "oov.int", true) < 0) return -1;		
	StringTable t_oov;
	if (ReadStringTable((langdir / "oov.int").string(), t_oov) < 0) return -1;
	std::string smap_oov(t_oov[0][0]);

	StringTable t_njs;
	string_vec _njs = {std::to_string(nj)};
	t_njs.push_back(_njs);
	if (SaveStringTable((traindir / "num_jobs").string(), t_njs) < 0) return -1;

	fs::path sdata (datadir / ("split"+ std::to_string(nj)));
	
	if ( !(fs::exists(sdata) && fs::is_directory(sdata) && fs::last_write_time(datadir / "feats.scp") < fs::last_write_time(sdata))) 
	{
		//split data directory
		if (SplitData(datadir, nj) < 0) return -1;
	}

	try	{
		fs::copy_file(langdir / "phones.txt", traindir / "phones.txt", fs::copy_option::overwrite_if_exists);
	} catch (const std::exception&)	{
		LOGTW_ERROR << "Could not copy " << langdir / "phones.txt" << " to " << traindir.string() << ".";
		return -1;
	}

	if (norm_vars) {
		if (cmvn_opts != "") cmvn_opts.append(" ");
		cmvn_opts.append("--norm-vars=true");
	}
	
	//keep track of options to CMVN.
	fs::ofstream file_cmvn_opts(traindir / "cmvn_opts", std::ios::binary);
	if (file_cmvn_opts) {
		file_cmvn_opts << cmvn_opts << "\n";
		file_cmvn_opts.flush(); file_cmvn_opts.close();
	}

	if (!(fs::exists(langdir / "phones" / "sets.int"))) {
		LOGTW_ERROR << "Can't find file: " << (langdir / "phones" / "sets.int").string() << ".";
		return -1;
	}

	std::string shared_phones_opt = "--shared-phones=" + (langdir / "phones" / "sets.int").string();

	//prepare the options for apply-cmvn
	string_vec options_applycmvn, options_adddeltas, options_feattodim;
	options_applycmvn.push_back("--print-args=false"); //NOTE: do not print arguments
	//parse and add cmvn_opts (space delimited collection of options on one line)
	string_vec _cmvn_opts;
	strtk::parse(cmvn_opts, " ", _cmvn_opts, strtk::split_options::compress_delimiters);
	for each(std::string s in _cmvn_opts)
		options_applycmvn.push_back(s);
	//NOTE: JOBID will need to be replaced later!
	//cmvn
	options_applycmvn.push_back("--print-args=false");
	options_applycmvn.push_back("--utt2spk=ark:" + (sdata / "JOBID" / "utt2spk").string());
	options_applycmvn.push_back("scp:" + (sdata / "JOBID" / "cmvn.scp").string());
	options_applycmvn.push_back("scp:" + (sdata / "JOBID" / "feats.scp").string());
	options_applycmvn.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //NOTE: this is the output of apply-cmvn!
	//deltas
	options_adddeltas.push_back("--print-args=false");
	options_adddeltas.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //input from apply-cmvn
	//IMPORTANT: the next option must be the last option because it is read later as output path!
	options_adddeltas.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
	//feat to dim
	options_feattodim.push_back("--print-args=false");
	options_feattodim.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //input: ouput from add-deltas!
	options_feattodim.push_back("ark,t:" + (sdata / "JOBID" / "feattodim.temp").string());  //output from FeatToDim
	//NOTE: we output the result of FeatToDim to a text file (,t)

	//STAGE 1 ----------------------------------
	if (stage <= -3)
	{
		LOGTW_INFO << "STAGE 1: Initializing monophone GMM...";

		//NOTE: JOB=1 just uses the 1st part of the features-- we only need a subset anyway.
		//----feat-to-dim------->
		int jobid = 1;
		std::string search("JOBID"), replace(std::to_string(jobid));
		std::string outcmvn;
		if(ApplyCmvnSequence(jobid, options_applycmvn, options_adddeltas, sdata, outcmvn) < 0) return -1;
		//Feat to Dim
		string_vec options;
		//must replace JOBID
		for each(std::string s in options_feattodim) {
			//replace JOBID with 1
			ReplaceStringInPlace(s, search, replace);
			options.push_back(s);
		}
		StrVec2Arg args(options);
		if (FeatToDim(args.argc(), args.argv()) < 0) return -1;
		//NOTE: FeatToDim writes a file which contains the number of columns for each key. The number of columns should be the same everywhere!
		//read in the output and delete temporary file
		StringTable tbl_feattodim;
		//get the output file
		std::string out(options[options.size()-1]);
		std::string search1("ark,t:"), replace1("");
		ReplaceStringInPlace(out, search1, replace1);
		if (ReadStringTable(out, tbl_feattodim) < 0) return -1;
		//NOTE: the 2nd column contains the number of columns what we need as feat_dim!
		int feat_dim = StringToNumber<int>(tbl_feattodim[0][1], -1);
		if (feat_dim < 0) {
			LOGTW_ERROR << "Error getting feature dimension.";
			return -1;
		}
		//<----feat-to-dim-------
		//subset-feats:
		string_vec options_subsetfeats, options_gmminitmono;
		options_subsetfeats.push_back("--print-args=false"); //NOTE: do not print arguments
		options_subsetfeats.push_back("--n=10");
		options_subsetfeats.push_back("ark:" + outcmvn); //output from ApplyCmvnSequence
		options_subsetfeats.push_back("ark:" + outcmvn + "sf"); //==> add_deltas.tempsf
		StrVec2Arg argssf(options_subsetfeats);
		fs::ofstream file_log(traindir / "log" / "init.log", fs::ofstream::binary | fs::ofstream::out);
		if (!file_log) LOGTW_WARNING << " log file is not accessible " << (traindir / "log" / "init.log").string() << ".";
		if(SubsetFeats(argssf.argc(), argssf.argv(), file_log) < 0) return -1;
		//gmm-init-mono:
		options_gmminitmono.push_back("--print-args=false");
		options_gmminitmono.push_back("--train-feats=ark:" + outcmvn + "sf"); //output from SubsetFeats above!
		//NOTE: we specify that the train-feats file is in archive format (ark:)!
		options_gmminitmono.push_back(shared_phones_opt);
		options_gmminitmono.push_back((langdir / "topo").string());
		options_gmminitmono.push_back(std::to_string(feat_dim));
		options_gmminitmono.push_back((traindir / "0.mdl").string());
		options_gmminitmono.push_back((traindir / "tree").string());
		StrVec2Arg arggmm(options_gmminitmono);
		if (GmmInitMono(arggmm.argc(), arggmm.argv()) < 0) return -1;

		//clean up
		try {
			// remove all temp files
			std::string temppath1((sdata / "JOBID" / "apply_cmvn.temp").string());
			std::string temppath2((sdata / "JOBID" / "add_deltas.temp").string());
			std::string temppath3((sdata / "JOBID" / "feattodim.temp").string());
			ReplaceStringInPlace(temppath1, search, replace);
			ReplaceStringInPlace(temppath2, search, replace);
			ReplaceStringInPlace(temppath3, search, replace);
			if (fs::exists(temppath1)) fs::remove(temppath1);
			if (fs::exists(temppath2)) fs::remove(temppath2);
			if (fs::exists(temppath3)) fs::remove(temppath3);
			if (fs::exists(outcmvn + "sf")) fs::remove(outcmvn + "sf");
		}
		catch (const std::exception&) {}
	}

	//determine number of gaussians:
	int numgauss = -1;
	string_vec options_gmminfo;
	options_gmminfo.push_back("--print-args=false");
	options_gmminfo.push_back((traindir / "0.mdl").string());
	options_gmminfo.push_back((traindir / "0.mdl.info").string());
	StrVec2Arg arggmmi(options_gmminfo);
	if (GmmInfo(arggmmi.argc(), arggmmi.argv()) < 0) return -1;
	//number-of-gaussians
	StringTable t_mdlinfo;
	if (ReadStringTable((traindir / "0.mdl.info").string(), t_mdlinfo) < 0) return -1;
	for (StringTable::const_iterator it(t_mdlinfo.begin()), it_end(t_mdlinfo.end()); it != it_end; ++it)
	{
		if ((*it)[0] == "number-of-gaussians" && (*it).size() > 1)
			numgauss = StringToNumber<int>((*it)[1], -1);
	}
	if (numgauss < 1) {
		LOGTW_ERROR << "Wrong property number of gaussians in " << (traindir / "0.mdl.info").string() << ".";
		return -1;
	}
	//per - iter increment for #Gauss:
	int incgauss = (totgauss - numgauss) / max_iter_inc;

	//STAGE 2 ----------------------------------
	if (stage <= -2)
	{
		LOGTW_INFO << "STAGE 2: Compiling training graphs...";
		std::vector<std::thread> _threads;
		//NOTE: must keep the parameters to the function call started in different threads in order that the threads can access it
		std::vector<StrVec2Arg *> _args1;
		_ret.clear();

		std::vector<string_vec> _options;

		//CompileTrainGraphs [options] <tree-in> <model-in> <lexicon-fst-in> <transcriptions-rspecifier> <graphs-wspecifier>
		//start several parallel threads
		for (int JOBID = 1; JOBID <= nj; JOBID++) 
		{
			//-------->Sym2Int
			fs::path symtab(langdir / "words.txt");
			fs::path input_txt(sdata / std::to_string(JOBID) / "text");
			fs::path output_txt(traindir / ("trnrspec." + std::to_string(JOBID) + ".temp"));
			int field_begin = 1; //NOTE: zero based index of fields! 2nd field is index=1
			int field_end = -1;			

			//-------->
			//add all options for compute-mfcc-feats
			string_vec options;
			options.push_back("--print-args=false");
			options.push_back("--read-disambig-syms="+(langdir / "phones" / "disambig.int").string());
			options.push_back((traindir / "tree").string());
			options.push_back((traindir / "0.mdl").string());
			options.push_back((langdir / "L.fst").string());
			//the output from sym2int:
			options.push_back("ark,t:" + (traindir / "trnrspec.JOBID.temp").string());
			//the output from CompileTrainGraphs //NOTE: must still gzip it at the end!
			options.push_back("ark:" + (traindir / "fsts.JOBID").string());

			//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
			for (std::string &s : options)
			{//NOTE: accesing by ref for in place editing
				ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
			}
			_options.push_back(options);
			StrVec2Arg *args1 = new StrVec2Arg(_options[JOBID-1]);

			_args1.push_back(args1);

			//-------->
			//logfile 
			fs::path log(traindir / "log" / ("compile_graphs." + std::to_string(JOBID) + ".log"));
			//
			_threads.emplace_back(
				LaunchJobCompileTrainGraphs,
				_args1[JOBID - 1]->argc(), _args1[JOBID - 1]->argv(),				//params for compile-train-graphs
				symtab, input_txt, output_txt, field_begin, field_end, smap_oov,	//params for Sym2Int
				(traindir / ("fsts." + std::to_string(JOBID))).string(),
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}
		//clean up
		try {
			for (int JOBID = 1; JOBID <= nj; JOBID++) {
				delete _args1[JOBID - 1];
				//NOTE: the *options is deleted in the StrVec2Arg object automatically when the list is deleted by the destructor!

				//delete temp files and fst's
				/*NOTE: because the gzip is removed we must not delete these files
				if (fs::exists(traindir / ("fsts." + std::to_string(JOBID))))
					fs::remove(traindir / ("fsts." + std::to_string(JOBID)));
				*/
				if (fs::exists(traindir / ("trnrspec." + std::to_string(JOBID) + ".temp")))
					fs::remove(traindir / ("trnrspec." + std::to_string(JOBID) + ".temp"));
			}
			_args1.clear();
		}
		catch (const std::exception& ex)
		{
			LOGTW_WARNING << "Could not free up memory and/or delete temporary files. Reason: " << ex.what() << ".";
		}
		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret[JOBID - 1] < 0)
				return -1;
		}
	}///STAGE 2

	//STAGE 3 ----------------------------------
	if (stage <= -1)
	{
		LOGTW_INFO << "STAGE 3: Aligning data equally (pass 0)...";
		std::vector<std::thread> _threads;
		//NOTE: must keep the parameters to the function call started in different threads in order that the threads can access it
		std::vector<StrVec2Arg *> _args1,  //AlignEqualCompiled
								  _args2;  //GmmAccStatsAli
		_ret.clear();

		std::vector<string_vec> _optionsAEC, _optionsGMA;

		//AlignEqualCompiled <graphs-rspecifier> <features-rspecifier> <alignments-wspecifier>
		//GmmAccStatsAli [options] <model-in> <feature-rspecifier> <alignments-rspecifier> <stats-out>
		//start several parallel threads
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//------>
			//add all options for AlignEqualCompiled
			string_vec optionsAEC;
			optionsAEC.push_back("--print-args=false");
			//NOTE: removed gziping the fsts.JOBID files because they are zipped and unzipped all the time
			optionsAEC.push_back("ark:" + (traindir / "fsts.JOBID").string());
			optionsAEC.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from ApplyCmvnSequence 
			optionsAEC.push_back("ark,t:"+(traindir / "0.JOBID.acc.temp").string()); //output => goes to GmmAccStatsAli

			//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
			for (std::string &s : optionsAEC)
			{//NOTE: accesing by ref for in place editing
				ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
			}
			_optionsAEC.push_back(optionsAEC);
			StrVec2Arg *args1 = new StrVec2Arg(_optionsAEC[JOBID-1]);
			_args1.push_back(args1);

			//------>
			//add all options for GmmAccStatsAli
			string_vec optionsGMA;
			optionsGMA.push_back("--print-args=false");
			optionsGMA.push_back("--binary=true");
			optionsGMA.push_back((traindir / "0.mdl").string());
			optionsGMA.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from ApplyCmvnSequence 
			optionsGMA.push_back("ark,t:" + (traindir / "0.JOBID.acc.temp").string());		//input from AlignEqualCompiled!
			optionsGMA.push_back((traindir / "0.JOBID.acc").string());

			//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
			for (std::string &s : optionsGMA)
			{//NOTE: accesing by ref for in place editing
				ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
			}
			_optionsGMA.push_back(optionsGMA);
			StrVec2Arg *args2 = new StrVec2Arg(_optionsGMA[JOBID - 1]);
			_args2.push_back(args2);

			//------->
			//logfile 
			fs::path log(traindir / "log" / ("align.0." + std::to_string(JOBID) + ".log"));
			//
			_threads.emplace_back(
				LaunchJobAlignData,
				_args1[JOBID - 1]->argc(), _args1[JOBID - 1]->argv(),				//params for AlignEqualCompiled
				_args2[JOBID - 1]->argc(), _args2[JOBID - 1]->argv(),				//params for GmmAccStatsAli
				JOBID, options_applycmvn, options_adddeltas, sdata,					//params for ApplyCmvnSequence
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}
		//clean up
		try {
			for (int JOBID = 1; JOBID <= nj; JOBID++) {
				delete _args1[JOBID - 1];
				delete _args2[JOBID - 1];
				//delete temp files:
				if (fs::exists((sdata / std::to_string(JOBID) / "add_deltas.temp"))) 
					fs::remove((sdata / std::to_string(JOBID) / "add_deltas.temp"));
				if (fs::exists((traindir / ("0." + std::to_string(JOBID) + ".acc.temp"))))
					fs::remove((traindir / ("0." + std::to_string(JOBID) + ".acc.temp")));
			}
			_args1.clear();
			_args2.clear();
		}
		catch (const std::exception& ex)
		{
			LOGTW_WARNING << "Could not free up memory and/or delete temporary files. Reason: " << ex.what() << ".";
		}
		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret[JOBID - 1] < 0)
				return -1;
		}
	}///STAGE 3

	// In the following steps, the --min-gaussian-occupancy=3 option is important, otherwise
	// we fail to est "rare" phones and later on, they never align properly.
	
	//STAGE 4 ----------------------------------
	if (stage <= 0)
	{
		LOGTW_INFO << "STAGE 4: Maximum Likelihood re-estimation of GMM-based acoustic model...";
		fs::ofstream file_log(traindir / "log" / "update.0.log", fs::ofstream::binary | fs::ofstream::out);
		if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (traindir / "log" / "update.0.log").string() << ".";
		//---------->
		string_vec options1;
		options1.push_back("--print-args=false");
		options1.push_back((traindir / ("0.accsum.temp")).string()); //output => goes to GmmEst
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			options1.push_back((traindir / ("0." + std::to_string(JOBID) + ".acc")).string());			
		}
		StrVec2Arg args1(options1);
		if (GmmSumAccs(args1.argc(), args1.argv(), file_log) < 0) return -1;

		//---------->
		string_vec options;
		options.push_back("--print-args=false");
		options.push_back("--min-gaussian-occupancy=3");
		options.push_back("--mix-up="+ std::to_string(numgauss));
		options.push_back("--power=" + std::to_string(power));
		options.push_back((traindir / "0.mdl").string());
		options.push_back((traindir / ("0.accsum.temp")).string()); //output from GmmSumAccs
		options.push_back((traindir / "1.mdl").string());
		StrVec2Arg args(options);
		if (GmmEst(args.argc(), args.argv(), file_log) < 0) return -1;
		
		//clean up temporary files
		try {
			for (int JOBID = 1; JOBID <= nj; JOBID++) {
				if (fs::exists(traindir / ("0." + std::to_string(JOBID) + ".acc")))
					fs::remove(traindir / ("0." + std::to_string(JOBID) + ".acc"));
				if (fs::exists(traindir / ("0.accsum.temp")))
					fs::remove(traindir / ("0.accsum.temp"));
			}
		}
		catch (const std::exception& ex) {
			LOGTW_WARNING << " Could not delete temporary files. Reason: " << ex.what() << ".";
		}

	}///STAGE 4

	LOGTW_INFO << "STAGE 5: Starting main training iteration...";

	int beam = 6; // will change to 10 below after 1st pass
	// note: using slightly wider beams for WSJ vs.RM.
	int x = 1;
	fs::ofstream file_log(traindir / "log" / "stage5.log", fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << " log file is not accessible " << (traindir / "log" / "stage5.log").string() << ".";
	
	//get the contents of lang/phones/optional_silence.csl
	std::string optional_silence("");
	try	{
		optional_silence = GetFirstLineFromFile((langdir / "phones" / "optional_silence.csl").string());
	} catch (const std::exception&){
		LOGTW_ERROR << " can not read " << (langdir / "phones" / "optional_silence.csl").string() << ".";
		return -1;
	}

	while (x < num_iters) {
		LOGTW_INFO << " >> Pass " << x << ".";
		if (stage <= x) {
			if (std::find(_realign_iters.begin(), _realign_iters.end(), x) != _realign_iters.end())
			{
				//LOGTW_INFO << "    aligning data...\n";
				string_vec options;
				options.push_back("--print-args=false");
				options.push_back("--boost=" + std::to_string(boost_silence));
				options.push_back(optional_silence);	//e.g.: "1:2:3"
				//NOTE: subsequent x.mdl files are made below
				options.push_back((traindir / (std::to_string(x) +".mdl")).string());
				options.push_back((traindir / (std::to_string(x) + ".bs.temp")).string()); //output
				StrVec2Arg args(options);
				if (GmmBoostSilence(args.argc(), args.argv(), file_log) < 0) return -1;

				//-------------------->
				//multiple threads
				std::vector<std::thread> _threads; 
				std::vector<StrVec2Arg *> _args1;
				_ret.clear();

				std::vector<string_vec> _optionsGAC;

				for (int JOBID = 1; JOBID <= nj; JOBID++)
				{
					//------>
					//add all options for GmmAlignCompiled
					string_vec optionsGAC;
					optionsGAC.push_back("--print-args=false");
					for each(std::string s in _scale_opts) optionsGAC.push_back(s);
					optionsGAC.push_back("--beam=" + std::to_string(beam));
					optionsGAC.push_back("--retry-beam=" + std::to_string(4*beam));
					std::string scareful = (careful ? "true" : "false");
					optionsGAC.push_back("--careful=" + scareful);
					optionsGAC.push_back((traindir / (std::to_string(x) + ".bs.temp")).string()); //input from GmmBoostSilence
					//NOTE: removed gziping the fsts.JOBID files because they are zipped and unzipped all the time
					optionsGAC.push_back("ark:" + (traindir / "fsts.JOBID").string());
					optionsGAC.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from ApplyCmvnSequence 
					//NOTE: removed gziping the ali.JOBID files because they are zipped and unzipped all the time
					optionsGAC.push_back("ark,t:" + (traindir / "ali.JOBID").string()); //output
					//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
					for (std::string &s : optionsGAC)
					{//NOTE: accesing by ref for in place editing
						ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
					}
					_optionsGAC.push_back(optionsGAC);
					StrVec2Arg *args1 = new StrVec2Arg(_optionsGAC[JOBID-1]);
					_args1.push_back(args1);

					//------->
					//logfile
					fs::path log(traindir / "log" / ("align." + std::to_string(x) + "." + std::to_string(JOBID) + ".log"));
					//
					_threads.emplace_back(
						LaunchJobGmmAlignCompiled,
						_args1[JOBID - 1]->argc(), _args1[JOBID - 1]->argv(),				//params for GmmAlignCompiled
						JOBID, options_applycmvn, options_adddeltas, sdata,					//params for ApplyCmvnSequence
						log);
				}
				//wait for the threads till they are ready
				for (auto& t : _threads) {
					t.join();
				}
				//clean up
				try {
					for (int JOBID = 1; JOBID <= nj; JOBID++) {
						delete _args1[JOBID - 1];

						//delete temp files:
						if (fs::exists((sdata / std::to_string(JOBID) / "add_deltas.temp")))
							fs::remove((sdata / std::to_string(JOBID) / "add_deltas.temp"));
					}
					if (fs::exists(traindir / (std::to_string(x) + ".bs.temp")))
						fs::remove(traindir / (std::to_string(x) + ".bs.temp"));
					_args1.clear();
				}
				catch (const std::exception& ex)
				{
					LOGTW_WARNING << " Could not free up memory and/or delete temporary files. Reason: " << ex.what() << ".";
				}
				//check return values from the threads/jobs
				for (int JOBID = 1; JOBID <= nj; JOBID++) {
					if (_ret[JOBID - 1] < 0)
						return -1;
				}

			} ///if (std::find(

			//GmmAccStatsAli ---------------------------------------------------------------------
			{ 
				std::vector<std::thread> _threads;
				std::vector<StrVec2Arg *> _args1;
				_ret.clear();

				std::vector<string_vec> _optionsGAC;

				for (int JOBID = 1; JOBID <= nj; JOBID++)
				{
					//------>
					//add all options for GmmAccStatsAli
					string_vec optionsGAC;
					optionsGAC.push_back("--print-args=false");
					optionsGAC.push_back((traindir / (std::to_string(x) + ".mdl")).string());
					optionsGAC.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from ApplyCmvnSequence 
					//NOTE: removed gziping the ali.JOBID files because they are zipped and unzipped all the time
					optionsGAC.push_back("ark,t:" + (traindir / "ali.JOBID").string());
					optionsGAC.push_back((traindir / (std::to_string(x)+".JOBID.acc")).string()); //output
					//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
					for (std::string &s : optionsGAC)
					{//NOTE: accesing by ref for in place editing
						ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
					}

					_optionsGAC.push_back(optionsGAC);
					StrVec2Arg *args1 = new StrVec2Arg(_optionsGAC[JOBID - 1]);
					_args1.push_back(args1);

					//------->
					//logfile
					fs::path log(traindir / "log" / ("acc." + std::to_string(x) + "." + std::to_string(JOBID) + ".log"));
					//
					_threads.emplace_back(
						LaunchJobGmmAccStatsAli,
						_args1[JOBID - 1]->argc(), _args1[JOBID - 1]->argv(),				//params for GmmAccStatsAli
						JOBID, options_applycmvn, options_adddeltas, sdata,					//params for ApplyCmvnSequence
						log);
				}
				//wait for the threads till they are ready
				for (auto& t : _threads) {
					t.join();
				}
				//clean up
				try {
					for (int JOBID = 1; JOBID <= nj; JOBID++) {
						delete _args1[JOBID - 1];

						//delete temp files:
						if (fs::exists((sdata / std::to_string(JOBID) / "add_deltas.temp")))
							fs::remove((sdata / std::to_string(JOBID) / "add_deltas.temp"));
					}
					_args1.clear();
				}
				catch (const std::exception& ex)
				{
					LOGTW_WARNING << "Could not free up memory and/or delete temporary files. Reason: " << ex.what() << ".";
				}
				//check return values from the threads/jobs
				for (int JOBID = 1; JOBID <= nj; JOBID++) {
					if (_ret[JOBID - 1] < 0)
						return -1;
				}
			}
			//---------------------------------------------------------------------------------------

			// make x.mdl -------------------------------------
			{ 
				fs::ofstream file_log(traindir / "log" / ("update."+std::to_string(x)+".log"), fs::ofstream::binary | fs::ofstream::out);
				if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (traindir / "log" / ("update." + std::to_string(x) + ".log")).string() << ".";

				//---------->
				string_vec options1;
				options1.push_back("--print-args=false");
				options1.push_back((traindir / (std::to_string(x)+".accsum.temp")).string()); //output => goes to GmmEst
				for (int JOBID = 1; JOBID <= nj; JOBID++)
				{
					options1.push_back((traindir / (std::to_string(x)+"." + std::to_string(JOBID) + ".acc")).string());
				}
				StrVec2Arg args1(options1);
				if (GmmSumAccs(args1.argc(), args1.argv(), file_log) < 0) return -1;

				//---------->
				string_vec options;
				options.push_back("--print-args=false");
				options.push_back("--write-occs="+ (traindir / (std::to_string(x + 1) + ".occs")).string());
				options.push_back("--mix-up=" + std::to_string(numgauss));
				options.push_back("--power=" + std::to_string(power));
				options.push_back((traindir / (std::to_string(x) + ".mdl")).string());
				options.push_back((traindir / (std::to_string(x) + ".accsum.temp")).string()); //output from GmmSumAccs
				options.push_back((traindir / (std::to_string(x+1) + ".mdl")).string());
				StrVec2Arg args(options);
				if (GmmEst(args.argc(), args.argv(), file_log) < 0) return -1;

				//clean up temporary files
				try {
					for (int JOBID = 1; JOBID <= nj; JOBID++) {
						if (fs::exists(traindir / (std::to_string(x) + "." + std::to_string(JOBID) + ".acc")))
							fs::remove(traindir / (std::to_string(x) + "." + std::to_string(JOBID) + ".acc"));
						if (fs::exists(traindir / (std::to_string(x) + ".accsum.temp")))
							fs::remove(traindir / (std::to_string(x) + ".accsum.temp"));
					}
					if (fs::exists(traindir / (std::to_string(x) + ".mdl"))) 
						fs::remove(traindir / (std::to_string(x) + ".mdl"));
					if (fs::exists(traindir / (std::to_string(x) + ".occs")))
						fs::remove(traindir / (std::to_string(x) + ".occs"));
				}
				catch (const std::exception& ex) {
					LOGTW_WARNING << " Could not delete temporary files. Reason: " << ex.what() << ".";
				}

			}
			//---------------------------------------------------------------------------------------	

		}
		if(x <= max_iter_inc)
			numgauss += incgauss;
		beam = 10;
		x++;
	} ///while 

	try	{
		if (fs::exists(traindir / "final.mdl")) fs::remove(traindir / "final.mdl");
		if (fs::exists(traindir / "final.occs")) fs::remove(traindir / "final.occs");
		fs::rename(traindir / (std::to_string(x) + ".mdl"), traindir / "final.mdl");
		fs::rename(traindir / (std::to_string(x) + ".occs"), traindir / "final.occs");
	} catch (const std::exception& ex)	{
		LOGTW_ERROR << " Could not save final model. Reason: " << ex.what() << ".";
		return -1;
	}

	//diagnostics:
	if (AnalyzeAlignments(langdir, traindir) < 0) return -1;

	//NOTE: these functions could summarize info about the logs which can be viewed manually also:
	//utils/summarize_warnings
	//steps/info/gmm_dir_info

	//clean up 
	try {
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			fs::path temppath(sdata / std::to_string(JOBID) / "apply_cmvn.temp");
			if (fs::exists(temppath)) fs::remove(temppath);
		}
	}
	catch (const std::exception&) {}

	LOGTW_INFO << "Done training monophone system in " << traindir.string() << ".";
	return 0;
}


//NOTE: all input files should be sorted! This is being checked and enforced several times in the data preparation process,
//		therefore no need for further check here.
static int ApplyCmvnSequence(int jobid, 
							 string_vec options_applycmvn, 
							 string_vec options_adddeltas, 
							 fs::path sdata, std::string & out)
{
	string_vec optionscmvn, optionsdelta;
	std::string search("JOBID"), replace(std::to_string(jobid));
	for each(std::string s in options_applycmvn) {
		//replace JOBID with jobid
		ReplaceStringInPlace(s, search, replace);
		optionscmvn.push_back(s);
	}
	for each(std::string s in options_adddeltas) {
		//replace JOBID with jobid
		ReplaceStringInPlace(s, search, replace);
		optionsdelta.push_back(s);
	}
	StrVec2Arg argcmvn(optionscmvn), argdelta(optionsdelta);
	//ApplyCmvn	
	std::string log((sdata / "JOBID" / "apply_cmvn.log").string()); //redirect Kaldi logging to the log file
	ReplaceStringInPlace(log, search, replace);
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log << ".";

	try	{
		//ApplyCmvn
		if (ApplyCmvn(argcmvn.argc(), argcmvn.argv(), file_log) < 0) {
			LOGTW_ERROR << "Error getting feature dimension while applying CMVN. See file: " << log << ".";
			return -1;
		}
		//AddDeltas
		if (AddDeltas(argdelta.argc(), argdelta.argv()) < 0) {
			LOGTW_ERROR << "Error getting feature dimension while applying deltas.";
			return -1;
		}
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ApplyCmvn, AddDeltas). Reason: " << ex.what();
		return -1;
	}

	//get the output file path
	out = optionsdelta[optionsdelta.size() - 1];
	std::string search1("ark:"), replace1("");
	ReplaceStringInPlace(out, search1, replace1);

	return 0;
}

//include section for gzip:
#include <boost/iostreams/device/file.hpp> 
#include <boost/iostreams/filtering_stream.hpp> 
#include <boost/iostreams/stream.hpp> 
#include <boost/iostreams/copy.hpp> 
#include <boost/iostreams/filter/gzip.hpp> 

namespace io = boost::iostreams;

//NOTE: this will be called from several threads
static void LaunchJobCompileTrainGraphs(
	int argc1, char *argv1[], 
	fs::path symtab, fs::path input_txt, fs::path output_txt, int field_begin, int field_end, std::string smap_oov,
	std::string fst_out,
	fs::path log)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	//run sym2int first to have the input for CompileTrainGraphs: traindir / "transcriptions_rspecifier.temp"
	StringTable t_symtab, t_input;
	if (ReadStringTable((symtab).string(), t_symtab) < 0) {
		_ret.push_back(-1);
		return;
	}
	if (ReadStringTable((input_txt).string(), t_input) < 0) {
		_ret.push_back(-1);
		return;
	}

	try	{
		if (Sym2Int(t_symtab, t_input, output_txt, field_begin, field_end, smap_oov) < 0) {
			_ret.push_back(-1);
			return;
		}
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (Sym2Int). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	int ret1 = 0;

	try	{
		ret1 = CompileTrainGraphs(argc1, argv1, file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (CompileTrainGraphs). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret1);
}


//NOTE: this will be called from several threads
static void LaunchJobAlignData(
	int argc1, char *argv1[], //params for AlignEqualCompiled
	int argc2, char *argv2[], //params for GmmAccStatsAli
	int jobid, string_vec options_applycmvn, string_vec options_adddeltas, fs::path sdata, //params for ApplyCmvnSequence
	fs::path log)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << " log file is not accessible " << log.string() << ".";

	std::string outcmvn;
	int ret1 = 0;

	try	{
		ret1 = ApplyCmvnSequence(jobid, options_applycmvn, options_adddeltas, sdata, outcmvn);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ApplyCmvnSequence). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	if (ret1 < 0) {
		//do not proceed if failed
		_ret.push_back(ret1);
		return;
	}

	try	{
		ret1 = AlignEqualCompiled(argc1, argv1, file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (AlignEqualCompiled). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	if (ret1 < 0) {
		//do not proceed if failed
		_ret.push_back(ret1);
		return;
	}

	try	{
		ret1 = GmmAccStatsAli(argc2, argv2, file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmAccStatsAli). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret1);
}

//NOTE: this will be called from several threads
static void LaunchJobGmmAlignCompiled(
	int argc1, char *argv1[], //params for 
	int jobid, string_vec options_applycmvn, string_vec options_adddeltas, fs::path sdata, //params for ApplyCmvnSequence
	fs::path log)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	std::string outcmvn;
	int ret1 = 0;

	try	{
		ret1 = ApplyCmvnSequence(jobid, options_applycmvn, options_adddeltas, sdata, outcmvn);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ApplyCmvnSequence). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	if (ret1 < 0) {
		//do not proceed if failed
		_ret.push_back(ret1);
		return;
	}

	try	{
		ret1 = GmmAlignCompiled(argc1, argv1, file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmAlignCompiled). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret1);
}

//NOTE: this will be called from several threads
static void LaunchJobGmmAccStatsAli(
	int argc1, char *argv1[], //params for 
	int jobid, string_vec options_applycmvn, string_vec options_adddeltas, fs::path sdata, //params for ApplyCmvnSequence
	fs::path log)
{
	//we redirect Kaldi logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	std::string outcmvn;
	int ret1 = 0;
	
	try	{
		ret1 = ApplyCmvnSequence(jobid, options_applycmvn, options_adddeltas, sdata, outcmvn);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ApplyCmvnSequence). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret1 < 0) {
		//do not proceed if failed
		_ret.push_back(ret1);
		return;
	}

	try	{
		ret1 = GmmAccStatsAli(argc1, argv1, file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmAccStatsAli). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret1);
}
