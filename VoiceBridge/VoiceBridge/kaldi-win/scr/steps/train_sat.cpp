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


static void LaunchComposeTransforms(
	int JOBID,
	string_vec options,
	fs::path dir,
	fs::path log
);

static void LaunchJobGmmEstFmllr(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	string_vec options_a2p,
	string_vec options_wsp,
	string_vec options_gef,
	std::string feat_type,
	bool apply_transform_feats,
	fs::path sdata,
	fs::path log
);

static void LaunchJobTreeStats(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	string_vec options_acctreestats,
	std::string feat_type,
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
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	string_vec options_gmmalignedcomp,
	std::string feat_type,
	fs::path sdata,
	fs::path log
);

static void LaunchJobGmmAccStatsAli(
	string_vec optionsGAC,
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	std::string feat_type,
	bool apply_transform_feats,
	fs::path sdata, //params for ApplyCmvnSequence
	fs::path log);

static void LaunchJobGmmAccStatsTwofeats(
	int JOBID,
	string_vec options_a2p,
	string_vec options_gast,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	std::string feat_type,
	fs::path sdata,
	fs::path log
);

//the return values from each thread/job
static std::vector<int> _ret;

/*
	Train a SAT model
	This does Speaker Adapted Training (SAT), i.e. train on fMLLR-adapted features. It can be done on top of either 
	LDA+MLLT, or delta and delta-delta features.  If there are no transforms supplied in the alignment directory, 
	it will estimate transforms itself before building the tree (and in any case, it estimates transforms a number
	of times during training).
*/
int TrainSat(
	fs::path data,			//data directory
	fs::path lang,			//language directory
	fs::path alidir,		//directory with the aligned base model (tri1)
	fs::path dir,			//output directory
	int nj,					//number of threads
	fs::path config,		//config file path with various options //TODO:... document
	double boost_silence,	//factor by which to boost silence likelihoods in alignment
	int numleaves,			//gauss algo param
	int totgauss,			//gauss algo param
	int stage				//stage; can be used to skip some steps done before
)
{
	LOGTW_INFO << "Starting to train SAT...";

	//Read the config file -------->
	//NOTE: the config file here must be parsed manually because the variables in it are
	//		used directly in the code!
	kaldi::ParseOptions po("");
	//Define and Register the options; default values set
	int exit_stage = -100; // you can use this to require it to exit at the beginning of a specific stage.Not all values are supported.
	std::string fmllr_update_type = "full";
	std::vector<std::string> _scale_opts = { "--transition-scale=1.0","--acoustic-scale=0.1","--self-loop-scale=0.1" };
	std::string scale_opts;
	int num_iters = 35;		//Number of iterations of training
	int max_iter_inc = 25;	//Last iter to increase #Gauss on.
	int beam = 10, retry_beam = 40;
	bool careful = false;
	double silence_weight = 0.0; //Weight on silence in fMLLR estimation.
	std::vector<int> _realign_iters = { 10, 20, 30 };
	std::string realign_iters;
	std::vector<int> _fmllr_iters = { 2, 4, 6, 12 };
	std::string fmllr_iters;
	double power = 0.2;			// exponent for number of gaussians according to occurrence counts
	int cluster_thresh = -1;	// for build-tree control final bottom-up clustering of leaves
	bool norm_vars = false;
	std::string phone_map;
	std::string context_opts, tree_stats_opts, cluster_phones_opts, compile_questions_opts;
	po.Register("num-iters", &num_iters, "Number of iterations of training.");
	po.Register("exit-stage", &exit_stage, "You can use this to require it to exit at the beginning of a specific stage.Not all values are supported.");	
	po.Register("fmllr-update-type", &fmllr_update_type, ".");
	po.Register("max-iter-inc", &max_iter_inc, "Last iter to increase #Gauss on.");
	po.Register("totgauss", &totgauss, "Target #Gaussians.");
	po.Register("careful", &careful, ".");
	po.Register("beam", &beam, ".");
	po.Register("retry-beam", &retry_beam, ".");	
	po.Register("silence-weight", &silence_weight, "Weight on silence in fMLLR estimation.");
	po.Register("boost-silence", &boost_silence, "Factor by which to boost silence likelihoods in alignment.");
	po.Register("realign-iters", &realign_iters, ".");
	po.Register("fmllr-iters", &fmllr_iters, ".");
	po.Register("stage", &stage, ".");
	po.Register("power", &power, "Exponent to determine number of gaussians from occurrence counts.");
	po.Register("cluster-thresh", &cluster_thresh, "For build-tree control final bottom-up clustering of leaves.");
	po.Register("norm-vars", &norm_vars, "Deprecated, prefer --cmvn-opts '--norm - vars = false'.");	
	po.Register("phone-map", &phone_map, ".");
	//options
	po.Register("scale-opts", &scale_opts, "Scale options for gmm-align-compiled.");
	po.Register("context-opts", &context_opts, "use '--context-width=5 --central-position=2' for quinphone.");
	po.Register("tree-stats-opts", &tree_stats_opts, "(one line separated by space).");
	po.Register("cluster-phones-opts", &cluster_phones_opts, "(one line separated by space).");
	po.Register("compile-questions-opts", &compile_questions_opts, "(one line separated by space).");
	
	//read config file and replace/overwrite the above default parameters
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

	std::vector<std::string> _context_opts;

	//in case these options are defined in the config file then read them in
	if (GetOptionsVector(scale_opts, _scale_opts) < 0) return -1;
	//make sure that the default values exist (the vector is cleard in GetOptionsVector)
	if (scale_opts.find("--transition-scale=") == std::string::npos) _scale_opts.push_back("--transition-scale=1.0");
	if (scale_opts.find("--acoustic-scale=") == std::string::npos) _scale_opts.push_back("--acoustic-scale=0.1");
	if (scale_opts.find("--self-loop-scale=") == std::string::npos) _scale_opts.push_back("--self-loop-scale=0.1");
	//
	if (GetOptionsVector(context_opts, _context_opts) < 0) return -1;

	//in case the realign-iters or mllt-iters is read from the config file it must be transferred to int vectors
	try {
		if (realign_iters != "") {
			_realign_iters.clear();
			strtk::parse(realign_iters, " ", _realign_iters, strtk::split_options::compress_delimiters);
		}
		if (fmllr_iters != "") {
			_fmllr_iters.clear();
			strtk::parse(fmllr_iters, " ", _fmllr_iters, strtk::split_options::compress_delimiters);
		}
	}
	catch (const std::exception&) {
		LOGTW_ERROR << "Wrong value for parameter 'realign-iters' or 'fmllr-iters' in configuration file.";
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

	if (CheckFileExistsAndNotEmpty(lang / "phones" / "silence.csl", true) < 0) return -1;
	std::string silphonelist = GetFirstLineFromFile((lang / "phones" / "silence.csl").string());

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

	//The following properties are not read from the config file but from the alidir if exist in order to be compatible!
	std::vector<std::string> _cmvn_opts, _delta_opts, _splice_opts;
	std::string cmvn_opts, delta_opts, splice_opts;

	//get options from ali dir
	if (fs::exists(alidir / "splice_opts")) splice_opts = GetFirstLineFromFile((alidir / "splice_opts").string());
	if (fs::exists(alidir / "cmvn_opts")) cmvn_opts = GetFirstLineFromFile((alidir / "cmvn_opts").string());
	if (fs::exists(alidir / "delta_opts")) delta_opts = GetFirstLineFromFile((alidir / "delta_opts").string());

	try	{
		boost::algorithm::trim(splice_opts);
		boost::algorithm::trim(cmvn_opts);
		boost::algorithm::trim(delta_opts);
		strtk::parse(splice_opts, " ", _splice_opts, strtk::split_options::compress_delimiters);
		strtk::parse(cmvn_opts, " ", _cmvn_opts, strtk::split_options::compress_delimiters);
		strtk::parse(delta_opts, " ", _delta_opts, strtk::split_options::compress_delimiters);
	} catch (const std::exception& ex)
	{
		LOGTW_ERROR << ex.what();
		return -1;
	}

	//check if deprecated property is used and add it
	if (norm_vars) _cmvn_opts.push_back("--norm-vars=true");

	//save all options to output directory for further use
	if (SaveOptionsToFile(dir / "context_opts", _context_opts) < 0) return -1;
	if (SaveOptionsToFile(dir / "cmvn_opts", _cmvn_opts) < 0) return -1;
	if (SaveOptionsToFile(dir / "splice_opts", _splice_opts) < 0) return -1;
	if (SaveOptionsToFile(dir / "delta_opts", _delta_opts) < 0) return -1;
	std::string phone_map_opt;
	if (phone_map != "") phone_map_opt = "--phone-map=" + phone_map;

	//feature type:
	std::string feat_type;
	if (fs::exists(alidir / "final.mat")) {
		feat_type = "lda";
		try {
			fs::copy_file(alidir / "final.mat", dir / "final.mat", fs::copy_option::overwrite_if_exists);
			if (fs::exists(alidir / "full.mat")) //do not fail with this
				fs::copy_file(alidir / "full.mat", dir / "full.mat", fs::copy_option::overwrite_if_exists);
		}
		catch (const std::exception& ex) {
			LOGTW_ERROR << ex.what();
			return -1;
		}
	}
	else feat_type = "delta";
	LOGTW_INFO << "feature type is " << feat_type;

	//Decide initial fMLLR transforms directory
	fs::path cur_trans_dir = dir;	
	if (fs::exists(alidir / "trans.1"))
		cur_trans_dir = alidir;
	LOGTW_INFO << "Using transforms from " << cur_trans_dir.string();

	//prepare options ------->
	string_vec options_applycmvn, options_adddeltas, options_splice, options_acctreestats, options_sumtreestats, options_transform_feats, options_transform_sifeats;
	//NOTE: JOBID will need to be replaced later everywhere before calls to functions!

	//apply-cmvn options
	options_applycmvn.push_back("--print-args=false"); //NOTE: do not print arguments
	//parse and add cmvn_opts (space delimited collection of options on one line)
	for each(std::string s in _cmvn_opts) options_applycmvn.push_back(s);	
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

	//splice-feats options
	options_splice.push_back("--print-args=false");
	for each(std::string s in _splice_opts) options_splice.push_back(s);
	options_splice.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //input from apply-cmvn
	//IMPORTANT: the next option must be the last option because it may be read later as output path!
	options_splice.push_back("ark:" + (sdata / "JOBID" / "splice.temp").string()); //output from splice-feats
	
	//transform-feats options
	options_transform_sifeats.push_back("--print-args=false");
	options_transform_sifeats.push_back((alidir / "final.mat").string()); 
	options_transform_sifeats.push_back("ark:" + (sdata / "JOBID" / "splice.temp").string()); //input from apply-cmvn sequence splice-feats
	//IMPORTANT: the next option must be the last option because it may be read later as output path!
	options_transform_sifeats.push_back("ark:" + (sdata / "JOBID" / "transform_sifeats.temp").string()); //output from first transform-feats

	//2nd transform-feats options when Using transforms from alidir!
	//feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
	options_transform_feats.push_back("--print-args=false");
	options_transform_feats.push_back("--utt2spk=ark:" + (sdata/"JOBID"/"utt2spk").string());	
	options_transform_feats.push_back("ark,s,cs:" + (cur_trans_dir / "trans.JOBID").string());
	if(feat_type == "lda")	
		options_transform_feats.push_back("ark:" + (sdata / "JOBID" / "transform_sifeats.temp").string()); //output from first transform-feats
	else
		options_transform_feats.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
	//IMPORTANT: the next option must be the last option because it may be read later as output path!
	options_transform_feats.push_back("ark:" + (sdata / "JOBID" / "transform_feats.temp").string()); //output from second transform-feats

	//prepare options for AccTreeStats (options_acctreestats)
	options_acctreestats.push_back("--print-args=false");
	for each(std::string s in _context_opts) options_acctreestats.push_back(s);
	if(tree_stats_opts != "") options_acctreestats.push_back(tree_stats_opts);
	if (phone_map_opt != "") options_acctreestats.push_back(phone_map_opt);
	if (ciphonelist != "") options_acctreestats.push_back("--ci-phones=" + ciphonelist);
	options_acctreestats.push_back((alidir / "final.mdl").string());
	options_acctreestats.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_feats.temp").string()); //output from transform-feats
	options_acctreestats.push_back("ark:" + (alidir / "ali.JOBID").string());
	options_acctreestats.push_back((dir / "JOBID.treeacc").string()); //output 

	//Get initial fMLLR transforms (possibly from alignment dir)
	if(cur_trans_dir == dir) {
		//5 ----->
		if (stage <= -5) {
			LOGTW_INFO << "Obtaining initial fMLLR transforms since not present in " << alidir.filename().string();

			//In case there is a phone_map defined as input but we are here (no transforms provided) then we must stop here 
			//because otherwise the silphonelist is being incorrect.
			if (phone_map != "") {
				LOGTW_ERROR << "You must provide transforms if you use the phone-map option.";
				return -1;
			}

			string_vec options_a2p, options_wsp, options_gef;
			//a2p
			options_a2p.push_back("--print-args=false");
			options_a2p.push_back("ark:" + (alidir / "ali.JOBID").string());
			options_a2p.push_back("ark:" + (alidir / "ali.JOBID.temp").string());
			//wsp
			options_wsp.push_back("--print-args=false");
			options_wsp.push_back(std::to_string(silence_weight));
			options_wsp.push_back(silphonelist);
			options_wsp.push_back((alidir / "final.mdl").string());
			options_wsp.push_back("ark:" + (alidir / "ali.JOBID.temp").string());
			options_wsp.push_back("ark:" + (alidir / "wsp.JOBID.temp").string());
			//gef
			options_gef.push_back("--print-args=false");
			options_gef.push_back("--fmllr-update-type=" + fmllr_update_type);
			options_gef.push_back("--spk2utt=ark:" + (sdata / "JOBID" / "spk2utt").string());
			options_gef.push_back((alidir / "final.mdl").string());

			if (feat_type == "lda")
				options_gef.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_sifeats.temp").string()); //output from first transform-feats
			else
				options_gef.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add_deltas

			options_gef.push_back("ark:" + (alidir / "wsp.JOBID.temp").string());
			options_gef.push_back("ark:" + (dir / "trans.JOBID").string());
			//---------------------------------------------------------------------
			//Start parallel processing
			std::vector<std::thread> _threads;
			_ret.clear();
			for (int JOBID = 1; JOBID <= nj; JOBID++)
			{
				//logfile
				fs::path log(dir / "log" / ("fmllr.0." + std::to_string(JOBID) + ".log"));
				_threads.emplace_back(
					LaunchJobGmmEstFmllr,
					JOBID,
					options_applycmvn,
					options_adddeltas,
					options_splice,
					options_transform_sifeats,
					options_transform_feats,
					options_a2p,
					options_wsp,
					options_gef,
					feat_type,
					false,	//!
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

			//clean up
			DeleteAllMatching(alidir, boost::regex(".*(\\.temp)$"));
			for (int JOBID = 1; JOBID <= nj; JOBID++)
				DeleteAllMatching(sdata / std::to_string(JOBID), boost::regex(".*(\\.temp)$"));
		}
	}

	//4
	if (stage <= -4) {
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
				options_splice,
				options_transform_sifeats,
				options_transform_feats,
				options_acctreestats,
				feat_type,
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

		//clean up
		for (int JOBID = 1; JOBID <= nj; JOBID++)
			DeleteAllMatching(sdata / std::to_string(JOBID), boost::regex(".*(\\.temp)$"));

		try {
			//options
			options_sumtreestats.push_back("--print-args=false");
			options_sumtreestats.push_back((dir / "treeacc").string()); //output
			for (int JOBID = 1; JOBID <= nj; JOBID++)
				options_sumtreestats.push_back((dir / (std::to_string(JOBID) + ".treeacc")).string()); //input files:

			fs::ofstream file_log(dir / "log" / "sum_tree_acc.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / "sum_tree_acc.log").string() << ".";
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
				fs::path p(dir / (std::to_string(JOBID) + ".treeacc"));
				if (fs::exists(p)) fs::remove(p);
			}
		}
		catch (const std::exception&) {}

	} ///stage 4


	//3
	//preparing questions, roots file...
	if (stage <= -3) {
		LOGTW_INFO << "Getting questions for tree-building, via clustering...";

		//preparing questions, roots file...

		//cluster phones
		try {
			//options
			string_vec options;
			options.push_back("--print-args=false");
			if(cluster_phones_opts!="") options.push_back(cluster_phones_opts);
			for each(std::string s in _context_opts) options.push_back(s);
			options.push_back((dir / "treeacc").string());
			options.push_back((lang / "phones" / "sets.int").string());
			options.push_back((dir / "questions.int.temp").string());
			StrVec2Arg args(options);
			//log
			fs::ofstream file_log(dir / "log" / "questions.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / "questions.log").string() << ".";
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
		try {
			fs::remove(dir / "questions.int.temp");
		}
		catch (const std::exception&) {}

		//Compile questions
		try {
			//options
			string_vec options;
			options.push_back("--print-args=false");
			for each(std::string s in _context_opts) options.push_back(s);
			if(compile_questions_opts!="") options.push_back(compile_questions_opts);
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
			options.push_back("--max-leaves=" + std::to_string(numleaves));
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

	} ///stage 3


	//2
	if (stage <= -2) {

		LOGTW_INFO << "Initializing the model...";

		//Gmm Init Model
		int ret = -1;
		try {
			//options
			string_vec options;
			options.push_back("--print-args=false");
			options.push_back("--write-occs=" + (dir / "1.occs").string());
			options.push_back((dir / "tree").string());
			options.push_back((dir / "treeacc").string());
			options.push_back((lang / "topo").string());
			options.push_back((dir / "1.mdl").string());
			StrVec2Arg args(options);
			//log
			fs::ofstream file_log(dir / "log" / "init_model.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << " log file is not accessible " << (dir / "log" / "init_model.log").string() << ".";
			//	
			ret = GmmInitModel(args.argc(), args.argv(), file_log);
			if (ret < 0) return -1;
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << " in (GmmInitModel). Reason: " << ex.what();
			return -1;
		}

		//NOTE: GmmInitModel returns 2 in case there are warnings about no stats.
		if (ret == 2) {
			LOGTW_INFO << "NOTE: The warnings above about 'no stats' generally mean you have phones (or groups of phones) in your";
			LOGTW_INFO << "phone set that had no corresponding data. You should probably figure out whether something went wrong,";
			LOGTW_INFO << "or whether your data just doesn't happen to have examples of those phones.";
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
		if(phone_map_opt!="") options.push_back(phone_map_opt);
		options.push_back((alidir / "final.mdl").string());
		options.push_back((dir / "1.mdl").string());
		options.push_back((dir / "tree").string());
		options.push_back("ark:" + (alidir / "ali.JOBID").string());
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

	if (exit_stage == 0) {
		LOGTW_INFO << "****** Exiting early as requested with exit-stage option. ******";
		return 0;
	}

	//0
	//Compiling graphs of transcripts
	if (stage <= 0 && _realign_iters.size()>0)
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
		DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
	} ///stage 0


	//training iterations ----------------------------------------------------->
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
				if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / ("boostsilence." + sx + ".log")).string() << ".";
				std::string optional_silence;
				try {
					optional_silence = GetFirstLineFromFile((lang / "phones" / "optional_silence.csl").string());
				}
				catch (const std::exception&) {
					LOGTW_ERROR << "Error can not read " << (lang / "phones" / "optional_silence.csl").string() << ".";
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
				options_gmmalignedcomp.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_feats.temp").string()); //output from transform-feats!
				options_gmmalignedcomp.push_back("ark:" + (dir / "ali.JOBID").string()); //output 

				//---------------------------------------------------------------------
				//Start parallel processing
				std::vector<std::thread> _threads;
				_ret.clear();
				for (int JOBID = 1; JOBID <= nj; JOBID++)
				{
					//logfile
					fs::path log(dir / "log" / ("align." + sx + "." + std::to_string(JOBID) + ".log"));
					//
					_threads.emplace_back(
						LaunchJobGmmAlignCompiled,
						JOBID,
						options_applycmvn,
						options_adddeltas,
						options_splice,
						options_transform_sifeats,
						options_transform_feats,
						options_gmmalignedcomp,
						feat_type,
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
		}

		//Estimating fMLLR transforms
		if (std::find(_fmllr_iters.begin(), _fmllr_iters.end(), x) != _fmllr_iters.end())
		{
			if (stage <= x)
			{
				LOGTW_INFO << "Estimating fMLLR transforms...";
				/* We estimate a transform that's additional to the previous transform; we'll compose them. */

				string_vec options_a2p, options_wsp, options_gef;
				//a2p
				options_a2p.push_back("--print-args=false");
				options_a2p.push_back("ark:" + (dir / "ali.JOBID").string());
				options_a2p.push_back("ark:" + (dir / "ali.JOBID.temp").string());
				//wsp
				options_wsp.push_back("--print-args=false");
				options_wsp.push_back(std::to_string(silence_weight));
				options_wsp.push_back(silphonelist);
				options_wsp.push_back((dir / (sx + ".mdl")).string());
				options_wsp.push_back("ark:" + (dir / "ali.JOBID.temp").string());
				options_wsp.push_back("ark:" + (dir / "wsp.JOBID.temp").string());
				//gef
				options_gef.push_back("--print-args=false");
				options_gef.push_back("--fmllr-update-type=" + fmllr_update_type);
				options_gef.push_back("--spk2utt=ark:" + (sdata / "JOBID" / "spk2utt").string());
				options_gef.push_back((dir / (sx + ".mdl")).string());
				options_gef.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_feats.temp").string()); //output from first transform-feats
				options_gef.push_back("ark:" + (dir / "wsp.JOBID.temp").string());
				options_gef.push_back("ark:" + (dir / "tmp_trans.JOBID").string());
				//---------------------------------------------------------------------
				{
					//Start parallel processing
					std::vector<std::thread> _threads;
					_ret.clear();
					for (int JOBID = 1; JOBID <= nj; JOBID++)
					{
						//logfile
						fs::path log(dir / "log" / ("fmllr." + sx + "." + std::to_string(JOBID) + ".log"));
						_threads.emplace_back(
							LaunchJobGmmEstFmllr,
							JOBID,
							options_applycmvn,
							options_adddeltas,
							options_splice,
							options_transform_sifeats,
							options_transform_feats,
							options_a2p,
							options_wsp,
							options_gef,
							feat_type,
							true,	//!
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
				}
				//---------------------------------------------------------------------

				//clean up
				DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
				for (int JOBID = 1; JOBID <= nj; JOBID++)
					DeleteAllMatching(sdata / std::to_string(JOBID), boost::regex(".*(\\.temp)$"));

				string_vec options;
				options.push_back("--print-args=false");
				options.push_back("--b-is-affine=true");
				options.push_back("ark:" + (dir / "tmp_trans.JOBID").string());
				options.push_back("ark:" + (cur_trans_dir / "trans.JOBID").string());
				options.push_back("ark:" + (dir / "composed_trans.JOBID").string());
				//---------------------------------------------------------------------
				{
					//Start parallel processing
					std::vector<std::thread> _threads;
					_ret.clear();
					for (int JOBID = 1; JOBID <= nj; JOBID++)
					{
						//logfile
						fs::path log(dir / "log" / ("compose_transforms." + sx + "." + std::to_string(JOBID) + ".log"));
						_threads.emplace_back(
							LaunchComposeTransforms,
							JOBID,
							options,
							dir,
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
				}
				//---------------------------------------------------------------------
			}

			//
			cur_trans_dir = dir;
		} ///Estimating MLLT


		if (stage <= x)
		{
			string_vec optionsGAC;
			optionsGAC.push_back("--print-args=false");
			optionsGAC.push_back((dir / (sx + ".mdl")).string());
			optionsGAC.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_feats.temp").string()); //output from ApplyCmvnSequence transform_feats
			optionsGAC.push_back("ark,s,cs:" + (dir / "ali.JOBID").string());
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
					options_applycmvn,
					options_adddeltas,
					options_splice,
					options_transform_sifeats,
					options_transform_feats,
					feat_type,
					true,	//!
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
				options1.push_back((dir / (sx + ".accsum.temp")).string()); //output => goes to GmmEst
				for (int JOBID = 1; JOBID <= nj; JOBID++) {
					options1.push_back((dir / (sx + "." + std::to_string(JOBID) + ".acc")).string());
				}
				StrVec2Arg args1(options1);
				if (GmmSumAccs(args1.argc(), args1.argv(), file_log) < 0) return -1;

				//options
				string_vec options;
				options.push_back("--print-args=false");
				options.push_back("--mix-up=" + std::to_string(numgauss));
				options.push_back("--power=" + std::to_string(power));
				options.push_back("--write-occs=" + (dir / (std::to_string(x + 1) + ".occs")).string());
				options.push_back((dir / (sx + ".mdl")).string());
				options.push_back((dir / (sx + ".accsum.temp")).string()); //output from GmmSumAccs
				options.push_back((dir / (std::to_string(x + 1) + ".mdl")).string());
				StrVec2Arg args(options);
				//	
				int ret = GmmEst(args.argc(), args.argv(), file_log);
				if (ret < 0) return -1;
			}
			catch (const std::exception& ex)
			{
				LOGTW_FATALERROR << "Error in (GmmEst). Reason: " << ex.what();
				return -1;
			}

			//clean up			
			try {
				if (fs::exists(dir / (sx + ".mdl"))) fs::remove(dir / (sx + ".mdl"));
				DeleteAllMatching(dir, boost::regex(".*(\\.acc)$"));
				if (fs::exists(dir / (sx + ".occs"))) fs::remove(dir / (sx + ".occs"));
			}
			catch (const std::exception&) {}
		}

		if (x <= max_iter_inc) numgauss += incgauss;
		x++;
	}

	//Accumulate stats for "alignment model"-- this model is computed with the speaker-independent features, 
	//but matches Gaussian-for-Gaussian with the final speaker-adapted model.
	if (stage <= x)
	{
		//AliToPost
		//GmmAccStatsTwofeats
		string_vec options_a2p, options_gast;
		options_a2p.push_back("--print-args=false");
		options_a2p.push_back("ark:" + (dir / "ali.JOBID").string());
		options_a2p.push_back("ark:" + (dir / "ali.JOBID.temp").string()); //output
		//
		options_gast.push_back("--print-args=false");
		options_gast.push_back((dir / (std::to_string(x) + ".mdl")).string());
		options_gast.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_feats.temp").string()); //output from ApplyCmvnSequence transform_feats
		if (feat_type == "lda")
			options_gast.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_sifeats.temp").string()); //output from first transform-feats
		else
			options_gast.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add_deltas
		options_gast.push_back("ark,s,cs:" + (dir / "ali.JOBID.temp").string());
		options_gast.push_back((dir / (std::to_string(x) + ".JOBID.acc")).string()); //output

		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(dir / "log" / ("acc_alimdl." + std::to_string(JOBID) + ".log"));
			//
			_threads.emplace_back(
				LaunchJobGmmAccStatsTwofeats,
				JOBID,
				options_a2p,
				options_gast,				
				options_applycmvn,
				options_adddeltas,
				options_splice,
				options_transform_sifeats,
				options_transform_feats,
				feat_type,
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

		//clean up
		DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
		for (int JOBID = 1; JOBID <= nj; JOBID++)
			DeleteAllMatching(sdata / std::to_string(JOBID), boost::regex(".*(\\.temp)$"));

		//Update model.

		//Gmm est
		try {
			//log
			fs::ofstream file_log(dir / "log" / "est_alimdl.log", fs::ofstream::binary | fs::ofstream::out);
			if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / "est_alimdl.log").string() << ".";
			//Gmm Sum Accs
			string_vec options1;
			options1.push_back("--print-args=false");
			options1.push_back((dir / (std::to_string(x) + ".accsum.temp")).string()); //output => goes to GmmEst
			for (int JOBID = 1; JOBID <= nj; JOBID++) {
				options1.push_back((dir / (std::to_string(x) + "." + std::to_string(JOBID) + ".acc")).string());
			}
			StrVec2Arg args1(options1);
			if (GmmSumAccs(args1.argc(), args1.argv(), file_log) < 0) return -1;

			//options
			string_vec options;
			options.push_back("--print-args=false");
			options.push_back("--remove-low-count-gaussians=false");
			options.push_back("--power=" + std::to_string(power));
			options.push_back((dir / (std::to_string(x) + ".mdl")).string());
			options.push_back((dir / (std::to_string(x) + ".accsum.temp")).string()); //output from GmmSumAccs
			options.push_back((dir / (std::to_string(x) + ".alimdl")).string());
			StrVec2Arg args(options);
			//	
			int ret = GmmEst(args.argc(), args.argv(), file_log);
			if (ret < 0) return -1;
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (GmmEst). Reason: " << ex.what();
			return -1;
		}

		DeleteAllMatching(dir, boost::regex(".*(\\.acc)$"));

	}

	//final model
	try {
		if (fs::exists(dir / "final.mdl")) fs::remove(dir / "final.mdl");
		if (fs::exists(dir / "final.occs")) fs::remove(dir / "final.occs");
		if (fs::exists(dir / "final.alimdl")) fs::remove(dir / "final.alimdl");
	}
	catch (const std::exception&) {}

	try {
		fs::rename(dir / (std::to_string(x) + ".mdl"), dir / "final.mdl");
		fs::rename(dir / (std::to_string(x) + ".occs"), dir / "final.occs");
		fs::rename(dir / (std::to_string(x) + ".alimdl"), dir / "final.alimdl");
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

	LOGTW_INFO << "Done training SAT system in " << dir.string();
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
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	std::string feat_type,
	bool apply_transform_feats, //2nd transform-feats options when Using transforms from alidir!
	fs::path sdata, std::string & out,
	fs::ofstream & file_log)
{
	string_vec optionscmvn, optionsadddeltas, optionssplice, optionstransformsifeats, optionstransformfeats;
	std::string search("JOBID"), replace(std::to_string(jobid));
	for each(std::string s in options_applycmvn) {
		ReplaceStringInPlace(s, search, replace);
		optionscmvn.push_back(s);
	}
	for each(std::string s in options_splice) {
		ReplaceStringInPlace(s, search, replace);
		optionssplice.push_back(s);
	}
	if (feat_type == "lda") {
		for each(std::string s in options_transform_sifeats) {
			//replace JOBID with jobid
			ReplaceStringInPlace(s, search, replace);
			optionstransformsifeats.push_back(s);
		}
	}
	else {
		for each(std::string s in options_adddeltas) {
			ReplaceStringInPlace(s, search, replace);
			optionsadddeltas.push_back(s);
		}
	}
	if (apply_transform_feats) {
		for each(std::string s in options_transform_feats) {
			//replace JOBID with jobid
			ReplaceStringInPlace(s, search, replace);
			optionstransformfeats.push_back(s);
		}
	}
	
	//ApplyCmvn	
	std::string log((sdata / "JOBID" / "apply_cmvn.log").string());
	ReplaceStringInPlace(log, search, replace);

	try {
		StrVec2Arg argcmvn(optionscmvn);
		//ApplyCmvn
		if (ApplyCmvn(argcmvn.argc(), argcmvn.argv(), file_log) < 0) {
			LOGTW_ERROR << "Error getting feature dimension while applying CMVN. See file: " << log << ".";
			return -1;
		}
		if (feat_type == "lda") {
			StrVec2Arg args(optionssplice), argssifeats(optionstransformsifeats);
			//SpliceFeats
			if (SpliceFeats(args.argc(), args.argv()) < 0) {
				LOGTW_ERROR << "Error while splicing.";
				return -1;
			}
			//TransformFeats
			if (TransformFeats(argssifeats.argc(), argssifeats.argv(), file_log) < 0) {
				LOGTW_ERROR << "Error while Transform Feats.";
				return -1;
			}
		}
		else {
			StrVec2Arg args(optionsadddeltas);
			//AddDeltas
			if (AddDeltas(args.argc(), args.argv()) < 0) {
				LOGTW_ERROR << "Error while adding deltas.";
				return -1;
			}
		}
		//last transformation in case of using transforms from alidir
		if (apply_transform_feats) {
			StrVec2Arg args(optionstransformfeats);
			//TransformFeats
			if (TransformFeats(args.argc(), args.argv(), file_log) < 0) {
				LOGTW_ERROR << "Error while Transform Feats.";
				return -1;
			}
		}
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in ApplyCmvnSequence. Reason: " << ex.what();
		return -1;
	}

	//get the output file path
	if (apply_transform_feats) {
		out = optionstransformfeats[optionstransformfeats.size() - 1];
	}
	else {
		if (feat_type == "lda") {
			out = optionstransformsifeats[optionstransformsifeats.size() - 1];
		}
		else {
			out = optionsadddeltas[optionsadddeltas.size() - 1];
		}
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
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	string_vec options_acctreestats,
	std::string feat_type,
	fs::path sdata,
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << " log file is not accessible " << log << ".";

	//replace 'JOBID' with the current job ID of the thread
	for (std::string &s : options_acctreestats) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	int ret = 0;
	//NOTE: all options (input/output) are distributed already well, therefore we just need to call the functions
	//		in the right sequence and all parameters will be OK!
	std::string outcmvn;

	//DO: ApplyCmvnSequence
	try {
		ret = ApplyCmvnSequence(JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splice,
			options_transform_sifeats,
			options_transform_feats,
			feat_type,
			true,
			sdata, outcmvn, file_log);
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
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	string_vec options_gmmalignedcomp,
	std::string feat_type,
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
		ret = ApplyCmvnSequence(JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splice,
			options_transform_sifeats,
			options_transform_feats,
			feat_type,
			true,
			sdata, outcmvn, file_log);
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
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	std::string feat_type,
	bool apply_transform_feats,
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
		ret1 = ApplyCmvnSequence(JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splice,
			options_transform_sifeats,
			options_transform_feats,
			feat_type,
			apply_transform_feats,
			sdata, outcmvn, file_log);
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


//LaunchJobGmmEstFmllr
static void LaunchJobGmmEstFmllr(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	string_vec options_a2p,
	string_vec options_wsp,
	string_vec options_gef,
	std::string feat_type,
	bool apply_transform_feats,
	fs::path sdata,
	fs::path log
)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
	for (std::string &s : options_a2p) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_wsp) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_gef) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	std::string outcmvn;
	int ret = 0;

	try {
		ret = ApplyCmvnSequence(JOBID, 
			options_applycmvn, 
			options_adddeltas, 
			options_splice, 
			options_transform_sifeats,
			options_transform_feats,
			feat_type, 
			apply_transform_feats,
			sdata, outcmvn, file_log);
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

	//AliToPost
	try {
		StrVec2Arg args(options_a2p);
		ret = AliToPost(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (AliToPost). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//WeightSilencePost
	try {
		StrVec2Arg args(options_wsp);
		ret = WeightSilencePost(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (WeightSilencePost). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	try {
		StrVec2Arg args(options_gef);
		ret = GmmEstFmllr(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmAccMllt). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret);
}


static void LaunchComposeTransforms(
	int JOBID,
	string_vec options,
	fs::path dir,
	fs::path log
)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	for (std::string &s : options) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	StrVec2Arg args(options);

	try {
		int ret = ComposeTransforms(args.argc(), args.argv(), file_log);
		if (ret < 0) {
			//do not proceed if failed
			_ret.push_back(ret);
			return;
		}

		fs::copy_file(dir / ("composed_trans." + std::to_string(JOBID)), dir / ("trans." + std::to_string(JOBID)), fs::copy_option::overwrite_if_exists);
		//cleanup
		if (fs::exists(dir / ("tmp_trans." + std::to_string(JOBID)))) fs::remove(dir / ("tmp_trans." + std::to_string(JOBID)));
		if (fs::exists(dir / ("composed_trans." + std::to_string(JOBID)))) fs::remove(dir / ("composed_trans." + std::to_string(JOBID)));
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ComposeTransforms). Reason: " << ex.what();
		_ret.push_back(-1);
	}

	_ret.push_back(0);
}


//LaunchJobGmmAccStatsTwofeats
static void LaunchJobGmmAccStatsTwofeats(
	int JOBID,
	string_vec options_a2p,
	string_vec options_gast,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splice,
	string_vec options_transform_sifeats,
	string_vec options_transform_feats,
	std::string feat_type,
	fs::path sdata,
	fs::path log
)
{
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
	for (std::string &s : options_a2p) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_gast) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	std::string outcmvn;
	int ret = 0;

	try {
		ret = ApplyCmvnSequence(JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splice,
			options_transform_sifeats,
			options_transform_feats,
			feat_type,
			true,	//!
			sdata, outcmvn, file_log);
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

	//clean up temp file before calling ApplyCmvnSequence for the second time
	//NOTE: only delete transform_sifeats.temp because transform_feats.temp (the output of the first ApplyCmvnSequence()
	//		is needed for GmmAccStatsTwofeats()!
	if (fs::exists(sdata / std::to_string(JOBID) / "transform_sifeats.temp")) fs::remove(sdata / std::to_string(JOBID) / "transform_sifeats.temp");

	try {
		//NOTE: this makes an transform_sifeats.temp
		ret = ApplyCmvnSequence(JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splice,
			options_transform_sifeats,
			options_transform_feats,
			feat_type,
			false,	//!
			sdata, outcmvn, file_log);
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

	//AliToPost
	try {
		StrVec2Arg args(options_a2p);
		ret = AliToPost(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (AliToPost). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//GmmAccStatsTwofeats
	try {
		StrVec2Arg args(options_gast);
		ret = GmmAccStatsTwofeats(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmAccStatsTwofeats). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret);
}
