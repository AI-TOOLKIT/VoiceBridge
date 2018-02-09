/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

	Based on : Copyright 2012  Johns Hopkins University (Author: Daniel Povey), Apache 2.0
*/

#include "kaldi-win/scr/kaldi_scr.h"
#include "kaldi-win/src/kaldi_src.h"
#include "util/common-utils.h" //for ParseOptions
#include "kaldi-win/utility/Utility2.h"

static void LaunchJobTreeStats(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_acctreestats,
	fs::path sdata,
	fs::path log
);

static void LaunchJobConvertAli(
	int JOBID,
	string_vec options,
	fs::path log
);

static void LaunchJobCompileTrainGraphs(
	int JOBID,
	string_vec options,
	std::string symtab, std::string input_txt, std::string output_txt, int field_begin, int field_end, std::string oov,	//params for Sym2Int
	fs::path log
);

static void LaunchJobGmmAlignCompiled(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_gmmalignedcomp,
	fs::path sdata,
	fs::path log
);

static void LaunchJobGmmAccStatsAli(
	string_vec optionsGAC,
	int jobid, 
	string_vec options_applycmvn, string_vec options_adddeltas, 
	fs::path sdata,
	fs::path log);

//the return values from each thread/job
static std::vector<int> _ret;

/*
	Train Deltas
*/
int TrainDeltas(
	fs::path data,			//data directory
	fs::path lang,			//language directory
	fs::path alidir,		//directory with the aligned base model (mono0a)
	fs::path dir,			//output directory
	int nj,					//number of threads
	fs::path config,		//config file path with various options //TODO:... document
	double boost_silence,	//factor by which to boost silence likelihoods in alignment
	int numleaves,			//gauss algo param
	int totgauss,			//gauss algo param
	int stage 				//stage; can be used to skip some steps done before
	)
{
	LOGTW_INFO << "Starting to train deltas...";

	//Read the config file -------->
	//NOTE: the config file here must be parsed manually because the variables in it are
	//		used directly in the code!
	kaldi::ParseOptions po("");
	//Define and Register the options; default values set
	std::vector<std::string> _scale_opts = { "--transition-scale=1.0","--acoustic-scale=0.1","--self-loop-scale=0.1" };
	std::string scale_opts;
	int num_iters = 35;		//Number of iterations of training
	int max_iter_inc = 25;	//Last iter to increase #Gauss on.
	int beam = 10, retry_beam = 40;
	bool careful = false;
	std::vector<int> _realign_iters = { 10, 20, 30 };
	std::string realign_iters;
	double power = 0.25;		// exponent for number of gaussians according to occurrence counts
	int cluster_thresh = -1;	// for build-tree control final bottom-up clustering of leaves
	bool norm_vars = false;
	std::string cmvn_opts, delta_opts, context_opts;
	po.Register("scale-opts", &scale_opts, "Scale options for gmm-align-compiled.");
	po.Register("num-iters", &num_iters, "Number of iterations of training.");
	po.Register("max-iter-inc", &max_iter_inc, "Last iter to increase #Gauss on.");
	po.Register("totgauss", &totgauss, "Target #Gaussians.");
	po.Register("careful", &careful, ".");
	po.Register("beam", &beam, ".");
	po.Register("retry-beam", &retry_beam, ".");
	po.Register("boost-silence", &boost_silence, "Factor by which to boost silence likelihoods in alignment.");
	po.Register("realign-iters", &realign_iters, ".");
	po.Register("stage", &stage, ".");
	po.Register("power", &power, "Exponent to determine number of gaussians from occurrence counts.");
	po.Register("cluster-thresh", &cluster_thresh, "For build-tree control final bottom-up clustering of leaves.");
	po.Register("norm-vars", &norm_vars, "Deprecated, prefer --cmvn-opts '--norm - vars = false'.");
	po.Register("cmvn-opts", &cmvn_opts, "Can be used to add extra options to cmvn.");
	po.Register("delta-opts", &delta_opts, "Can be used to add extra options to add deltas.");
	po.Register("context-opts", &context_opts, "use '--context-width=5 --central-position=2' for quinphone.");
	std::vector<std::string> _cmvn_opts, _delta_opts, _context_opts;
	//
	if (config != "" && fs::exists(config) && !fs::is_empty(config))
	{//all parameters will be overwritten with the parameters defined in the config file
		string_vec options;
		options.push_back("--config=" + config.string());
		StrVec2Arg args(options);
		LOGTW_INFO << "Reading and using configuration file for parameters:";
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
	//<---- config file end 

	//in case these options are defined in the config file then read them in
	if (GetOptionsVector(scale_opts, _scale_opts) < 0) return -1;
	if (GetOptionsVector(cmvn_opts, _cmvn_opts) < 0) return -1;
	if (GetOptionsVector(delta_opts, _delta_opts) < 0) return -1;
	if (GetOptionsVector(context_opts, _context_opts) < 0) return -1;
	//in case the realign-iters is read from the config file it must be transferred to _realign_iters int vector
	try {
		if (realign_iters != "") {
			_realign_iters.clear();
			strtk::parse(realign_iters, " ", _realign_iters, strtk::split_options::compress_delimiters);
		}
	}
	catch (const std::exception&) {
		LOGTW_ERROR << "Wrong value for parameter 'realign-iters' in configuration file.";
		return -1;
	}

	//check if files exist
	if (CheckIfFilesExist(std::vector<fs::path> {alidir / "final.mdl", alidir / "ali.1", data / "feats.scp", lang / "phones.txt"}) < 0) return -1;

	int numgauss = numleaves;
	int incgauss = (totgauss - numgauss) / max_iter_inc; // per - iter increment for Gauss

	//get oov
	if (CheckFileExistsAndNotEmpty(lang / "oov.int", true) < 0) return -1;
	StringTable t_oov;
	if (ReadStringTable((lang / "oov.int").string(), t_oov) < 0) return -1;
	std::string oov(t_oov[0][0]);

	//get ciphonelist (e.g. 1:2:3:4:5:6:7:8:9:10)
	if (CheckFileExistsAndNotEmpty(lang / "phones" / "context_indep.csl", true) < 0) return -1;
	std::string ciphonelist = GetFirstLineFromFile((lang / "phones" / "context_indep.csl").string());

	//create log directory
	if (CreateDir(dir / "log", true) < 0) return -1;

	//check nj compatibility
	std::string snj;
	int nj_orig;
	try {
		snj = GetFirstLineFromFile((alidir / "num_jobs").string());
	}
	catch (const std::exception&) {}
	nj_orig = StringToNumber<int>(snj, -1);
	if (nj_orig < 1) {
		LOGTW_ERROR << "Could not read number of jobs from file " << (alidir / "num_jobs").string() << ".";
		return -1;
	}
	if (nj != nj_orig)
	{
		LOGTW_WARNING << "The requested number of jobs mismatches alignment data. Resetting nj to " << nj_orig;
		nj = nj_orig;
	}
	//save num_jobs
	StringTable t_njs;
	string_vec _njs = { std::to_string(nj) };
	t_njs.push_back(_njs);
	if (SaveStringTable((dir / "num_jobs").string(), t_njs) < 0) return -1;

	//extra sanity check
	if (CheckPhonesCompatible(lang / "phones.txt", alidir / "phones.txt") < 0) return -1;

	//
	try {
		fs::copy_file(lang / "phones.txt", dir / "phones.txt", fs::copy_option::overwrite_if_exists);
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << ex.what();
		return -1;
	}

	//check if the data is aready split into nj parts; if not then split it!
	fs::path sdata(data / ("split" + std::to_string(nj)));
	if (!(fs::exists(sdata) && fs::is_directory(sdata) && fs::last_write_time(data / "feats.scp") < fs::last_write_time(sdata)))
	{
		//split data directory
		if (SplitData(data, nj) < 0) return -1;
	}

	//check if deprecated property is used and add it
	if (norm_vars) _cmvn_opts.push_back("--norm-vars=true");

	//save all options to output directory for further use
	if (SaveOptionsToFile(dir / "context_opts", _context_opts) < 0) return -1;
	if (SaveOptionsToFile(dir / "cmvn_opts", _cmvn_opts) < 0) return -1;
	if (SaveOptionsToFile(dir / "delta_opts", _delta_opts) < 0) return -1;

	LOGTW_INFO << "NOTE: Ignoring any existing CMVN options from source directory: ";
	LOGTW_INFO << alidir.string();

	//prepare options ------->
	string_vec options_applycmvn, options_adddeltas, options_acctreestats, options_sumtreestats;
	//prepare the options for apply-cmvn
	options_applycmvn.push_back("--print-args=false"); //NOTE: do not print arguments
	//parse and add cmvn_opts (space delimited collection of options on one line)
	for each(std::string s in _cmvn_opts) options_applycmvn.push_back(s);
	//NOTE: JOBID will need to be replaced later!
	//cmvn
	options_applycmvn.push_back("--utt2spk=ark:" + (sdata / "JOBID" / "utt2spk").string());
	options_applycmvn.push_back("scp:" + (sdata / "JOBID" / "cmvn.scp").string());
	options_applycmvn.push_back("scp:" + (sdata / "JOBID" / "feats.scp").string());
	//IMPORTANT: the next option must be the last option because it is read later as output path!
	options_applycmvn.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //NOTE: this is the output of apply-cmvn!
	//add-deltas options
	options_adddeltas.push_back("--print-args=false");
	for each(std::string s in _delta_opts) options_adddeltas.push_back(s);
	options_adddeltas.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //input from apply-cmvn
	//IMPORTANT: the next option must be the last option because it is read later as output path!
	options_adddeltas.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas

	//prepare options for AccTreeStats (options_acctreestats)
	options_acctreestats.push_back("--print-args=false");
	for each(std::string s in _context_opts) options_acctreestats.push_back(s);
	options_acctreestats.push_back("--ci-phones="+ciphonelist);
	options_acctreestats.push_back((alidir / "final.mdl").string());
	options_acctreestats.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
	options_acctreestats.push_back("ark:" + (alidir / "ali.JOBID").string());
	options_acctreestats.push_back((dir / "JOBID.treeacc").string()); //output 

	//split the work in stages in order to be able to skip stages if they are already done before
	if (stage <= -3) {
		LOGTW_INFO << "Accumulating tree stats...";
		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(dir / "log" / ("acc_tree." + std::to_string(JOBID) + ".log"));
			//
			_threads.emplace_back(
				LaunchJobTreeStats,
				JOBID,
				options_applycmvn,
				options_adddeltas,
				options_acctreestats,
				sdata,
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}
		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret[JOBID - 1] < 0)
				return -1;
		}
		//---------------------------------------------------------------------

		try {
			//options
			options_sumtreestats.push_back("--print-args=false");
			options_sumtreestats.push_back((dir / "treeacc").string()); //output
			for (int JOBID = 1; JOBID <= nj; JOBID++)
				options_sumtreestats.push_back((dir / (std::to_string(JOBID) + ".treeacc")).string()); //input files:

			fs::ofstream file_log(dir / "log" / "sum_tree_acc.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << " log file is not accessible " << (dir / "log" / "sum_tree_acc.log").string() << ".";
			StrVec2Arg args(options_sumtreestats);
			if (SumTreeStats(args.argc(), args.argv(), file_log) < 0) return -1;
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (SumTreeStats). Reason: " << ex.what();
			return -1;
		}

		//cleanup
		try {
			for (int JOBID = 1; JOBID <= nj; JOBID++) {
				std::string p((dir / (std::to_string(JOBID) + ".treeacc")).string());
				if (fs::exists(p)) fs::remove(p);
			}
		}
		catch (const std::exception&) {}

	} ///stage 3


	//preparing questions, roots file...
	if (stage <= -2) {
		LOGTW_INFO << "Getting questions for tree-building, via clustering...";

		//preparing questions, roots file...
		//cluster phones
		try {
			//options
			string_vec options;
			options.push_back("--print-args=false");			
			for each(std::string s in _context_opts) options.push_back(s);
			options.push_back((dir / "treeacc").string());
			options.push_back((lang / "phones" / "sets.int").string());
			options.push_back((dir / "questions.int.temp").string());
			StrVec2Arg args(options);
			//log
			fs::ofstream file_log(dir / "log" / "questions.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << " log file is not accessible " << (dir / "log" / "questions.log").string() << ".";
			//			
			if (ClusterPhones(args.argc(), args.argv(), file_log) < 0) return -1;
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (ClusterPhones). Reason: " << ex.what();
			return -1;
		}

		//add extra questions to questions
		fs::path oq(dir / "questions.int");
		if (MergeFiles(std::vector<fs::path>{dir / "questions.int.temp", lang / "phones" / "extra_questions.int"}, oq) < 0) return -1;
		//clean up
		try	{
			fs::remove(dir / "questions.int.temp");
		} catch (const std::exception&){}

		//Compile questions
		try {
			//options
			string_vec options;
			options.push_back("--print-args=false");
			for each(std::string s in _context_opts) options.push_back(s);
			options.push_back((lang / "topo").string());
			options.push_back((dir / "questions.int").string());
			options.push_back((dir / "questions.qst").string());
			StrVec2Arg args(options);
			//log
			fs::ofstream file_log(dir / "log" / "compile_questions.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / "compile_questions.log").string() << ".";
			//			
			if (CompileQuestions(args.argc(), args.argv(), file_log) < 0) return -1;
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (CompileQuestions). Reason: " << ex.what();
			return -1;
		}

		LOGTW_INFO << "Building the tree...";

		//Build Tree
		try {
			//options
			string_vec options;
			options.push_back("--print-args=false");
			for each(std::string s in _context_opts) options.push_back(s);
			options.push_back("--verbose=1");
			options.push_back("--max-leaves="+std::to_string(numleaves));
			options.push_back("--cluster-thresh=" + std::to_string(cluster_thresh));
			options.push_back((dir / "treeacc").string());
			options.push_back((lang / "phones" / "roots.int").string());
			options.push_back((dir / "questions.qst").string());
			options.push_back((lang / "topo").string());
			options.push_back((dir / "tree").string());
			StrVec2Arg args(options);
			//log
			fs::ofstream file_log(dir / "log" / "build_tree.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / "build_tree.log").string() << ".";
			//			
			if (BuildTree(args.argc(), args.argv(), file_log) < 0) return -1;
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (BuildTree). Reason: " << ex.what();
			return -1;
		}

		LOGTW_INFO << "Initializing Gmm model...";

		//Gmm Init Model
		int ret = -1;
		try {
			//options
			string_vec options;
			options.push_back("--print-args=false");
			options.push_back("--write-occs="+(dir/"1.occs").string());
			options.push_back((dir / "tree").string());
			options.push_back((dir / "treeacc").string());
			options.push_back((lang / "topo").string());
			options.push_back((dir / "1.mdl").string());			
			StrVec2Arg args(options);
			//log
			fs::ofstream file_log(dir / "log" / "init_model.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / "init_model.log").string() << ".";
			//	
			ret = GmmInitModel(args.argc(), args.argv(), file_log);
			if (ret < 0) return -1;
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (GmmInitModel). Reason: " << ex.what();
			return -1;
		}

		//NOTE: GmmInitModel returns 2 in case there are warnings about no stats.
		if (ret == 2) {
			LOGTW_INFO << "NOTE: The warnings above about 'no stats' generally mean you have phones (or groups of phones) in your";
			LOGTW_INFO << "phone set that had no corresponding data. You should probably figure out whether something went wrong,";
			LOGTW_INFO << "or whether your data just doesn't happen to have examples of those phones.";
		}

		//Gmm Mixup
		try {
			//options
			string_vec options;
			options.push_back("--print-args=false");
			options.push_back("--mix-up=" + std::to_string(numgauss));
			options.push_back((dir / "1.mdl").string());
			options.push_back((dir / "1.occs").string());
			options.push_back((dir / "1.mdl").string());
			StrVec2Arg args(options);
			//log
			fs::ofstream file_log(dir / "log" / "mixup.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << " log file is not accessible " << (dir / "log" / "mixup.log").string() << ".";
			//	
			ret = GmmMixup(args.argc(), args.argv(), file_log);
			if (ret < 0) return -1;
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << " in (GmmMixup). Reason: " << ex.what();
			return -1;
		}
		//clean up
		try {
			fs::remove(dir / "treeacc");
		}
		catch (const std::exception&) {}

	} ///stage 2


	//Convert the alignments.
	if (stage <= -1) 
	{
		LOGTW_INFO << "Converting alignments from " << alidir.filename().string() << " to use current tree...";

		//ConvertAli
		//options
		string_vec options;
		options.push_back("--print-args=false");
		options.push_back((alidir / "final.mdl").string());
		options.push_back((dir / "1.mdl").string());
		options.push_back((dir / "tree").string());
		options.push_back("ark:"+(alidir / "ali.JOBID").string());
		options.push_back("ark:" + (dir / "ali.JOBID").string());
		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(dir / "log" / ("convert." + std::to_string(JOBID) + ".log"));
			//
			_threads.emplace_back(
				LaunchJobConvertAli,
				JOBID,
				options,
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}
		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret[JOBID - 1] < 0)
				return -1;
		}
		//---------------------------------------------------------------------
	} ///stage 1

	//Compiling graphs of transcripts
	if (stage <= 0)
	{
		LOGTW_INFO << "Compiling graphs of transcripts...";

		//Sym2Int options
		std::string symtab((lang / "words.txt").string());
		std::string input_txt((sdata / "JOBID" / "text").string());
		std::string output_txt((dir / "trnrspec.JOBID.temp").string()); //output from Sym2Int
		//oov is defined above
		int field_begin = 1; //NOTE: zero based index of fields! 2nd field is index=1
		int field_end = -1;
		//CompileTrainGraphs options
		string_vec options;
		options.push_back("--print-args=false");
		options.push_back("--read-disambig-syms=" + (lang / "phones" / "disambig.int").string());
		options.push_back((dir / "tree").string());
		options.push_back((dir / "1.mdl").string());
		options.push_back((lang / "L.fst").string());
		options.push_back("ark:" + output_txt); //output from Sym2Int
		options.push_back("ark:" + (dir / "fsts.JOBID").string());
		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(dir / "log" / ("compile_graphs." + std::to_string(JOBID) + ".log"));
			//
			_threads.emplace_back(
				LaunchJobCompileTrainGraphs,
				JOBID,
				options,
				symtab, input_txt, output_txt, field_begin, field_end, oov,	//params for Sym2Int
				log);
		}
		//wait for the threads till they are ready
		for (auto& t : _threads) {
			t.join();
		}
		//check return values from the threads/jobs
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			if (_ret[JOBID - 1] < 0)
				return -1;
		}
		//---------------------------------------------------------------------
		//cleanup
		try {
			DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
		}
		catch (const std::exception&) {}
	} ///stage 0

	//training iterations
	int x = 1;
	while (x < num_iters) {
		LOGTW_INFO << "Training pass " << x;
		std::string sx(std::to_string(x));
		if (stage <= x) 
		{
			if (std::find(_realign_iters.begin(), _realign_iters.end(), x) != _realign_iters.end())
			{
				LOGTW_INFO << "Aligning data...";

				//1. Boost Silence 
				//get the contents of lang/phones/optional_silence.csl
				fs::ofstream file_log(dir / "log" / ("boostsilence." + sx + ".log"), fs::ofstream::binary | fs::ofstream::out);
				if (!file_log) LOGTW_WARNING << " log file is not accessible " << (dir / "log" / ("boostsilence." + sx + ".log")).string() << ".";
				std::string optional_silence;
				try {
					optional_silence = GetFirstLineFromFile((lang / "phones" / "optional_silence.csl").string());
				}
				catch (const std::exception&) {
					LOGTW_ERROR << " can not read " << (lang / "phones" / "optional_silence.csl").string() << ".";
					return -1;
				}
				string_vec options;
				options.push_back("--print-args=false");
				options.push_back("--boost=" + std::to_string(boost_silence));
				options.push_back(optional_silence);	//e.g.: "1:2:3"
				options.push_back((dir / (sx + ".mdl")).string());
				options.push_back((dir / (sx + ".bs.temp")).string()); //output
				StrVec2Arg args(options);
				if (GmmBoostSilence(args.argc(), args.argv(), file_log) < 0) return -1;

				//2. options for GmmAlignCompiled
				//options GmmAlignCompiled
				string_vec options_gmmalignedcomp;
				options_gmmalignedcomp.push_back("--print-args=false");
				std::string scareful = (careful ? "true" : "false");
				options_gmmalignedcomp.push_back("--careful=" + scareful);
				options_gmmalignedcomp.push_back("--beam=" + std::to_string(beam));
				options_gmmalignedcomp.push_back("--retry-beam=" + std::to_string(retry_beam));
				for each(std::string s in _scale_opts) options_gmmalignedcomp.push_back(s);
				options_gmmalignedcomp.push_back((dir / (sx + ".bs.temp")).string()); //output from boost silence!
				options_gmmalignedcomp.push_back("ark:" + (dir / "fsts.JOBID").string());
				options_gmmalignedcomp.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
				options_gmmalignedcomp.push_back("ark:" + (dir / "ali.JOBID").string()); //output 

				//---------------------------------------------------------------------
				//Start parallel processing
				std::vector<std::thread> _threads;
				_ret.clear();
				for (int JOBID = 1; JOBID <= nj; JOBID++)
				{
					//logfile
					fs::path log(dir / "log" / ("align." + sx +"." + std::to_string(JOBID) + ".log"));
					//
					_threads.emplace_back(
						LaunchJobGmmAlignCompiled,
						JOBID,
						options_applycmvn,
						options_adddeltas,
						options_gmmalignedcomp,
						sdata,
						log);
				}
				//wait for the threads till they are ready
				for (auto& t : _threads) {
					t.join();
				}
				//check return values from the threads/jobs
				for (int JOBID = 1; JOBID <= nj; JOBID++) {
					if (_ret[JOBID - 1] < 0)
						return -1;
				}
				//---------------------------------------------------------------------
			} ///_realign_iters

			string_vec optionsGAC;
			optionsGAC.push_back("--print-args=false");
			optionsGAC.push_back((dir / (sx + ".mdl")).string());
			optionsGAC.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from ApplyCmvnSequence 
			//NOTE: removed gziping the ali.JOBID files because they are zipped and unzipped all the time
			optionsGAC.push_back("ark:" + (dir / "ali.JOBID").string());
			optionsGAC.push_back((dir / (sx + ".JOBID.acc")).string()); //output
			//---------------------------------------------------------------------
			//Start parallel processing
			std::vector<std::thread> _threads;
			_ret.clear();
			for (int JOBID = 1; JOBID <= nj; JOBID++)
			{
				//logfile
				fs::path log(dir / "log" / ("acc." + sx + "." + std::to_string(JOBID) + ".log"));
				//
				_threads.emplace_back(
					LaunchJobGmmAccStatsAli,
					optionsGAC,
					JOBID,
					options_applycmvn, options_adddeltas, 
					sdata,
					log);
			}
			//wait for the threads till they are ready
			for (auto& t : _threads) {
				t.join();
			}
			//check return values from the threads/jobs
			for (int JOBID = 1; JOBID <= nj; JOBID++) {
				if (_ret[JOBID - 1] < 0)
					return -1;
			}
			//---------------------------------------------------------------------

			//Gmm est
			try {
				//log
				fs::ofstream file_log(dir / "log" / ("update." + sx + ".log"), fs::ofstream::binary | fs::ofstream::out);
				if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / ("update." + sx + ".log")).string() << ".";
				//Gmm Sum Accs
				string_vec options1;
				options1.push_back("--print-args=false");
				options1.push_back((dir / (sx+".accsum.temp")).string()); //output => goes to GmmEst
				for (int JOBID = 1; JOBID <= nj; JOBID++) {
					options1.push_back((dir / (sx+"." + std::to_string(JOBID) + ".acc")).string());
				}
				StrVec2Arg args1(options1);
				if (GmmSumAccs(args1.argc(), args1.argv(), file_log) < 0) return -1;

				//options
				string_vec options;
				options.push_back("--print-args=false");
				options.push_back("--mix-up=" + std::to_string(numgauss));
				options.push_back("--power=" + std::to_string(power));
				options.push_back("--write-occs=" + (dir / (std::to_string(x + 1)+".occs")).string());
				options.push_back((dir / (sx+".mdl")).string());
				options.push_back((dir / (sx + ".accsum.temp")).string()); //output from GmmSumAccs
				options.push_back((dir / (std::to_string(x + 1) + ".mdl")).string());
				StrVec2Arg args(options);
				//	
				int ret = GmmEst(args.argc(), args.argv(), file_log);
				if (ret < 0) return -1;
			}
			catch (const std::exception& ex)
			{
				LOGTW_FATALERROR << " in (GmmEst). Reason: " << ex.what();
				return -1;
			}

			//clean up			
			try {
				if (fs::exists(dir / (sx + ".mdl"))) fs::remove(dir / (sx + ".mdl"));
				DeleteAllMatching(dir, boost::regex(".*(\\.acc)$"));
				if (fs::exists(dir / (sx + ".occs"))) fs::remove(dir / (sx + ".occs"));
			} catch (const std::exception&) {}
		}

		if (x <= max_iter_inc) numgauss += incgauss;
		x++;
	}

	//clean up
	try	{
		if (fs::exists(dir / "final.mdl")) fs::remove(dir / "final.mdl");
		if (fs::exists(dir / "final.occs")) fs::remove(dir / "final.occs");
	}catch (const std::exception&){}

	try	{
		fs::rename(dir / (std::to_string(x) + ".mdl"), dir / "final.mdl");
		fs::rename(dir / (std::to_string(x) + ".occs"), dir / "final.occs");
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << "Can not rename model to final.mdl/occs. Reason: " << ex.what();
		return -1;
	}

	//diagnostics:
	if (AnalyzeAlignments(lang, dir) < 0) return -1;

	//Could still summarize warning messages here...

	//Could still extract some info from the logs...

	//clean up some files
	try {
		DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
		for (int JOBID = 1; JOBID <= nj; JOBID++)
			DeleteAllMatching(sdata / std::to_string(JOBID), boost::regex(".*(\\.temp)$"));
	}
	catch (const std::exception&) {}

	LOGTW_INFO << "Done training system with delta+delta-delta features in " << dir.string();
	return 0;
}


//--------------------------------------------------------------------------------------------------------
//		PARALLEL SECTION
//--------------------------------------------------------------------------------------------------------

//NOTE: all input files should be sorted! This is being checked and enforced several times in the data preparation process,
//		therefore no need for further check here.
static int ApplyCmvnSequence(int jobid,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	fs::path sdata, std::string & out,
	bool applyDeltas,
	fs::ofstream & file_log)
{
	string_vec optionscmvn, optionsdelta;
	std::string search("JOBID"), replace(std::to_string(jobid));
	for each(std::string s in options_applycmvn) {
		//replace JOBID with jobid
		ReplaceStringInPlace(s, search, replace);
		optionscmvn.push_back(s);
	}
	if (applyDeltas) {
		for each(std::string s in options_adddeltas) {
			//replace JOBID with jobid
			ReplaceStringInPlace(s, search, replace);
			optionsdelta.push_back(s);
		}
	}
	StrVec2Arg argcmvn(optionscmvn), argdelta(optionsdelta);
	//ApplyCmvn	
	std::string log((sdata / "JOBID" / "apply_cmvn.log").string()); //redirect Kaldi logging to the log file
	ReplaceStringInPlace(log, search, replace);

	try {
		//ApplyCmvn
		if (ApplyCmvn(argcmvn.argc(), argcmvn.argv(), file_log) < 0) {
			LOGTW_ERROR << "Error getting feature dimension while applying CMVN. See file: " << log << ".";
			return -1;
		}
		//AddDeltas
		if (applyDeltas && AddDeltas(argdelta.argc(), argdelta.argv()) < 0) {
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
	if (applyDeltas) {
		out = optionsdelta[optionsdelta.size() - 1];
	}
	else {
		out = optionscmvn[optionscmvn.size() - 1];
	}
	std::string search1("ark:"), replace1("");
	ReplaceStringInPlace(out, search1, replace1);

	return 0;
}

/*
	parallel job for AccTreeStats

	NOTE: the string_vec options must not be passed by reference and make a copy because of the JOBID's!
*/
static void LaunchJobTreeStats(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_acctreestats,
	fs::path sdata,
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log << ".";

	//replace 'JOBID' with the current job ID of the thread
	for (std::string &s : options_acctreestats) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	int ret = 0;
	//NOTE: all options (input/output) are distributed already well, therefore we just need to call the functions
	//		in the right sequence and all parameters will be OK!
	std::string outcmvn;

	//DO: apply-cmvn + add-deltas		
	try {
		ret = ApplyCmvnSequence(JOBID, options_applycmvn, options_adddeltas, sdata, outcmvn, true, file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ApplyCmvnSequence). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//DO: AccTreeStats
	try {
		StrVec2Arg args(options_acctreestats);
		ret = AccTreeStats(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (AccTreeStats). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//all OK
	_ret.push_back(0);
}


/*
	parallel job for ConvertAli()

	NOTE: the string_vec options must not be passed by reference and make a copy because of the JOBID's!
*/
static void LaunchJobConvertAli(
	int JOBID,
	string_vec options,
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log << ".";

	//replace 'JOBID' with the current job ID of the thread
	for (std::string &s : options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	int ret;
	try {
		StrVec2Arg args(options);
		ret = ConvertAli(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ConvertAli). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	_ret.push_back(ret);
}


/*
	parallel job for CompileTrainGraphs()

	NOTE: the string_vec options must not be passed by reference and make a copy because of the JOBID's!
*/
static void LaunchJobCompileTrainGraphs(
	int JOBID,
	string_vec options,
	std::string symtab, std::string input_txt, std::string output_txt, int field_begin, int field_end, std::string oov,	//params for Sym2Int
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log << ".";

	//replace 'JOBID' with the current job ID of the thread
	for (std::string &s : options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	ReplaceStringInPlace(input_txt, "JOBID", std::to_string(JOBID));
	ReplaceStringInPlace(output_txt, "JOBID", std::to_string(JOBID));

	StringTable t_symtab, t_input;
	if (ReadStringTable(symtab, t_symtab) < 0) {
		_ret.push_back(-1);
		return;
	}
	if (ReadStringTable(input_txt, t_input) < 0) {
		_ret.push_back(-1);
		return;
	}
	try {
		if (Sym2Int(t_symtab, t_input, output_txt, field_begin, field_end, oov) < 0) {
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

	int ret;
	try {
		StrVec2Arg args(options);
		ret = CompileTrainGraphs(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (CompileTrainGraphs). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	_ret.push_back(ret);
}


/*
	parallel job for GmmAlignCompiled()

	NOTE: the string_vec options must not be passed by reference and make a copy because of the JOBID's!
*/
static void LaunchJobGmmAlignCompiled(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_gmmalignedcomp,
	fs::path sdata,
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log << ".";

	//replace 'JOBID' with the current job ID of the thread
	for (std::string &s : options_gmmalignedcomp) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	int ret = 0;
	//NOTE: all options (input/output) are distributed already well, therefore we just need to call the functions
	//		in the right sequence and all parameters will be OK!
	std::string outcmvn;
	//DO: apply-cmvn + add-deltas		
	try {
		ret = ApplyCmvnSequence(JOBID, options_applycmvn, options_adddeltas, sdata, outcmvn, true, file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ApplyCmvnSequence). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//DO: GmmAlignCompiled
	try {
		StrVec2Arg args(options_gmmalignedcomp);
		ret = GmmAlignCompiled(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmAlignCompiled). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//all OK
	_ret.push_back(0);
}


/*
	parallel job for GmmAccStatsAli()

	NOTE: the string_vec options must not be passed by reference and make a copy because of the JOBID's!
*/
static void LaunchJobGmmAccStatsAli(
	string_vec optionsGAC,
	int JOBID, 
	string_vec options_applycmvn, string_vec options_adddeltas, 
	fs::path sdata, //params for ApplyCmvnSequence
	fs::path log)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
	for (std::string &s : optionsGAC)
	{//NOTE: accesing by ref for in place editing
		ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	}

	std::string outcmvn;
	int ret1 = 0;

	try {
		ret1 = ApplyCmvnSequence(JOBID, options_applycmvn, options_adddeltas, sdata, outcmvn, true, file_log);
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

	try {
		StrVec2Arg args(optionsGAC);
		ret1 = GmmAccStatsAli(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmAccStatsAli). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret1);
}
