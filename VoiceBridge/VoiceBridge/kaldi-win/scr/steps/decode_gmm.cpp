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

static void LaunchJobGmmLatgenFaster(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_transformfeats_trans,
	string_vec options_gmmlatgen,
	std::string feat_type,
	fs::path trans_dir,
	fs::path sdata,
	fs::path log
);

//the return values from each thread/job
static std::vector<int> _ret;

/*
	This function works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out what type of features
	you used (assuming it's one of these two)

	IMPORTANT: splice_opts, cmvn_opts, delta_opts only the first line is read!
*/
VOICEBRIDGE_API int Decode(
	fs::path graph_dir,							//graphdir
	fs::path data_dir,							//data
	fs::path decode_dir,						//dir - is assumed to be a sub-directory of the directory where the model is.
	fs::path model,								//which model to use e.g. final.mdl
	fs::path trans_dir,							//dir to find fMLLR transforms; this option won't normally be used, but it can be used if you want to supply existing fMLLR transforms when decoding	
	UMAPSS & wer_ref_filter,					//ref filter NOTE: can be empty but must be defined!
	UMAPSS & wer_hyp_filter,					//hyp filter NOTE: can be empty but must be defined!
	std::string iter,							//Iteration of model to test e.g. 'final'
	int nj,										//default: 4, number of parallel jobs
	float acwt,									//acoustic scale used for lattice generation;  NOTE: only really affects pruning (scoring is on lattices).
	int stage,									//
	int max_active,								//
	double beam,								//Pruning beam [applied after acoustic scaling]
	double lattice_beam,						//
	bool skip_scoring,							//
	//scoring options:
	bool decode_mbr, 							//maximum bayes risk decoding (confusion network).
	bool stats, 								//output statistics
	std::string word_ins_penalty, 				//word insertion penalty
	int min_lmwt, 								//minumum LM-weight for lattice rescoring
	int max_lmwt 								//maximum LM-weight for lattice rescoring
)
{
	fs::path srcdir(decode_dir.parent_path()); //The model directory is one level up from decoding directory.
	fs::path sdata(data_dir / ("split" + std::to_string(nj)));
	if (CreateDir(decode_dir / "log", true) < 0) {
		LOGTW_ERROR << "Failed to create " << (decode_dir / "log").string();
		return -1;
	}
	//
	if (!(fs::exists(sdata) && fs::last_write_time(data_dir / "feats.scp") < fs::last_write_time(sdata))) {
		//split data directory
		if (SplitData(data_dir, nj) < 0) return -1;
	}
	//save num_jobs
	StringTable t_njs;
	string_vec _njs = { std::to_string(nj) };
	t_njs.push_back(_njs);
	if (SaveStringTable((decode_dir / "num_jobs").string(), t_njs) < 0) return -1;
	//decide which model to use
	if (model == "" || !fs::exists(model)) {
		if (iter == "")
			model = srcdir / "final.mdl";
		else
			model = srcdir / (iter + ".mdl");
	}

	if (model.filename().string() != "final.alimdl")
	{
		//Do not use the srcpath but look at the path where the model is
		if (fs::exists(model.parent_path() / "final.alimdl") && (trans_dir == "" || !fs::exists(trans_dir))) {
			LOGTW_WARNING << "Running speaker independent system decoding using a SAT model! This is OK if you know what you are doing...";
		}
	}
	//check if all required files exist
	std::vector<fs::path> required = { sdata / "1" / "feats.scp", sdata / "1" / "cmvn.scp", model, graph_dir / "HCLG.fst" };
	for (fs::path p : required) {
		if(!fs::exists(p)) {
			LOGTW_ERROR << "Failed to find " << p.string();
			return -1;
		}
	}
	//feature type:
	std::string feat_type;
	if(fs::exists(srcdir / "final.mat"))
		feat_type = "lda"; 
	else feat_type = "delta";
	LOGTW_INFO << "Feature type is " << feat_type;

	std::string splice_opts, cmvn_opts, delta_opts;

	try	{
		if (fs::exists(srcdir / "splice_opts")) //frame-splicing options
			splice_opts = GetFirstLineFromFile((srcdir / "splice_opts").string());
		if (fs::exists(srcdir / "cmvn_opts"))
			cmvn_opts = GetFirstLineFromFile((srcdir / "cmvn_opts").string());
		if (fs::exists(srcdir / "delta_opts"))
			delta_opts = GetFirstLineFromFile((srcdir / "delta_opts").string());
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << ex.what();
		return -1;
	}

	int nj_orig = 0;

	//add transforms to features
	if (trans_dir != "" && fs::exists(trans_dir)) 
	{
		LOGTW_INFO << "Using fMLLR transforms from " << trans_dir.string();
		if (!fs::exists(trans_dir / "trans.1")) {
			LOGTW_ERROR << "Expected " << (trans_dir / "trans.1").string() << " to exist.";
			return -1;
		}
		//get the number of jobs (used to designate the number of parallel threads used in the calculations)
		std::string snj("");
		try {
			snj = GetFirstLineFromFile((trans_dir / "num_jobs").string());
		}
		catch (const std::exception&) {}
		nj_orig = StringToNumber<int>(snj, -1);
		if (nj_orig < 1) {
			LOGTW_ERROR << "Could not read number of jobs from file " << (trans_dir / "num_jobs").string() << ".";
			return -1;
		}
		//check if the requested number of jobs corresponds to the number of jobs used to create the trans_dir data files
		//NOTE: this is important because parallel processing is done by subdividin the data files into number of threads peaces
		if (nj != nj_orig)
		{
			//Copy all of the transforms into one archive with an index.
			LOGTW_INFO << "The number of jobs for transforms mismatches, so copying them.";
			//first combine all files:
			StringTable _t, _tcombined;
			for (int n = 1; n <= nj_orig; n++) {
				if (ReadStringTable((trans_dir / ("trans." + std::to_string(n))).string(), _t) < 0) return -1;
				_tcombined.insert(std::end(_tcombined), std::begin(_t), std::end(_t));
			}
			if (SaveStringTable((trans_dir / "trans.temp").string(), _tcombined) < 0) return -1;
			//add all options for copy-feats
			string_vec copy_feats_options;
			copy_feats_options.push_back("--print-args=false"); //NOTE: do not print arguments
			copy_feats_options.push_back("ark:" + (trans_dir / "trans.temp").string());
			copy_feats_options.push_back("ark,scp:" + (decode_dir / ("trans.ark")).string() + "," + (decode_dir / ("trans.scp")).string());
			StrVec2Arg args(copy_feats_options);
			fs::ofstream file_log; //NOTE:... set a path if you need a log, otherwise it goes to the global log
			if(CopyFeats(args.argc(), args.argv(), file_log) < 0) return -1;
		}
		else 
		{
			//number of jobs matches with alignment dir
			//see below 'prepare features' section
		}
	} ///if (trans_dir

	if (stage <= 0)
	{
		if (fs::exists(graph_dir / "num_pdfs"))
		{
			//read the original num_pdfs
			std::string snpdfs("");
			try {
				snpdfs = GetFirstLineFromFile((graph_dir / "num_pdfs").string());
			}
			catch (const std::exception&) {}
			int num_pdfs_orig = StringToNumber<int>(snpdfs, -1);
			if (num_pdfs_orig < 0) {
				LOGTW_ERROR << "Could not read number of pdf's from file " << (graph_dir / "num_pdfs").string() << ".";
				return -1;
			}
			//get number of pfd's from model info
			int nofphones, num_pdfs, noftransitionids, noftransitionstates;
			if (AmInfo(model.string(), nofphones, num_pdfs, noftransitionids, noftransitionstates) < 0) return -1;
			if (num_pdfs_orig != num_pdfs) {
				LOGTW_ERROR << "Mismatch in number of pdfs with model in " << model.string();
				return -1;
			}
		}

		//
		//Prepare all parameters for the funcions which will be called in the parallel processing unit
		//
		string_vec options_applycmvn, options_adddeltas, options_splicefeats, options_transformfeats, options_transformfeats_trans;
		//prepare the options for apply-cmvn			
		options_applycmvn.push_back("--print-args=false"); //NOTE: do not print arguments
		//parse and add cmvn_opts (space delimited collection of options on one line)
		string_vec _cmvn_opts;
		//NOTE: cmvn_opts must be 1 line of options delimited by " "
		strtk::parse(cmvn_opts, " ", _cmvn_opts, strtk::split_options::compress_delimiters);
		for each(std::string s in _cmvn_opts)
			options_applycmvn.push_back(s);
		//NOTE: JOBID will need to be replaced later!
		//cmvn
		options_applycmvn.push_back("--utt2spk=ark:" + (sdata / "JOBID" / "utt2spk").string());
		options_applycmvn.push_back("scp:" + (sdata / "JOBID" / "cmvn.scp").string());
		options_applycmvn.push_back("scp:" + (sdata / "JOBID" / "feats.scp").string());
		//IMPORTANT: the next option must be the last option because it is read later as output path!
		options_applycmvn.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //NOTE: this is the output of apply-cmvn!
		//prepare features
		if (feat_type == "delta") {
			//add-deltas options
			options_adddeltas.push_back("--print-args=false");
			options_adddeltas.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //input from apply-cmvn
			//IMPORTANT: the next option must be the last option because it is read later as output path!
			options_adddeltas.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
		}
		else //if (feat_type == "lda")
		{
			//splice-feats, transform-feats options
			string_vec _splice_opts;
			//NOTE: splice_opts must be 1 line of options delimited by " "
			strtk::parse(splice_opts, " ", _splice_opts, strtk::split_options::compress_delimiters);
			for each(std::string s in _splice_opts)
				options_splicefeats.push_back(s);
			options_splicefeats.push_back("--print-args=false");
			options_splicefeats.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //input from apply-cmvn
			options_splicefeats.push_back("ark:" + (sdata / "JOBID" / "splicefeats.temp").string()); //output from splice-feats
			//
			options_transformfeats.push_back("--print-args=false");
			options_transformfeats.push_back((srcdir / "final.mat").string());
			options_transformfeats.push_back("ark:" + (sdata / "JOBID" / "splicefeats.temp").string()); //input from splice-feats
			options_transformfeats.push_back("ark:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from transform-feats
		}
		//
		if (trans_dir != "" && fs::exists(trans_dir)) {
			if (nj != nj_orig) {
				//options transform-feats 
				options_transformfeats_trans.push_back("--print-args=false");
				options_transformfeats_trans.push_back("--utt2spk=ark:" + (sdata / "JOBID" / "utt2spk").string());
				options_transformfeats_trans.push_back("scp:"+(decode_dir / "trans.scp").string());
				if (feat_type == "delta")
					options_transformfeats_trans.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
				else
					options_transformfeats_trans.push_back("ark:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from transform-feats
				options_transformfeats_trans.push_back("ark:" + (sdata / "JOBID" / "transformfeats_trans.temp").string()); //output
			}
			else 
			{ //number of jobs matches with alignment dir
				//options transform-feats 
				options_transformfeats_trans.push_back("--print-args=false");
				options_transformfeats_trans.push_back("--utt2spk=ark:" + (sdata / "JOBID" / "utt2spk").string());
				options_transformfeats_trans.push_back("ark:" + (trans_dir / "trans.JOBID").string());
				if (feat_type == "delta")
					options_transformfeats_trans.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
				else
					options_transformfeats_trans.push_back("ark:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from transform-feats			
				options_transformfeats_trans.push_back("ark:" + (sdata / "JOBID" / "transformfeats_trans.temp").string()); //output
			}
		}

		//options GmmLatgenFaster
		string_vec options_gmmlatgen;
		options_gmmlatgen.push_back("--print-args=false");
		options_gmmlatgen.push_back("--max-active=" + std::to_string(max_active));
		options_gmmlatgen.push_back("--beam=" + std::to_string(beam));
		options_gmmlatgen.push_back("--lattice-beam=" + std::to_string(lattice_beam));
		options_gmmlatgen.push_back("--acoustic-scale=" + std::to_string(acwt));
		options_gmmlatgen.push_back("--allow-partial=true");
		options_gmmlatgen.push_back("--word-symbol-table=" + (graph_dir / "words.txt").string());
		options_gmmlatgen.push_back(model.string());
		options_gmmlatgen.push_back((graph_dir / "HCLG.fst").string());
		//depending on the former processing the input here can be different 'feats'
		if (trans_dir != "" && fs::exists(trans_dir)) {
			options_gmmlatgen.push_back("ark:" + (sdata / "JOBID" / "transformfeats_trans.temp").string()); //output from trans
		}
		else {
			if (feat_type == "delta") {
				options_gmmlatgen.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
			}
			else {
				options_gmmlatgen.push_back("ark:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from transform-feats
			}
		}
		options_gmmlatgen.push_back("ark:"+(decode_dir / "lat.JOBID").string()); //output

		//make sure that there are no old lat.* files in the output directory beacuse all 'lat.*' will be used later!
		if (DeleteAllMatching(decode_dir, boost::regex("^(lat\\.).*")) < 0) return -1;

		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(decode_dir / "log" / ("decode." + std::to_string(JOBID) + ".log"));
			//
			_threads.emplace_back(
				LaunchJobGmmLatgenFaster,
				JOBID, 
				options_applycmvn, 
				options_adddeltas, 
				options_splicefeats, 
				options_transformfeats, 
				options_transformfeats_trans,
				options_gmmlatgen,
				feat_type,
				trans_dir,
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
	}

	if (stage <= 1)
	{
		if(AnalyzeLats(graph_dir, decode_dir, iter) < 0) return -1;
	}

	if (!skip_scoring)
	{
		/*NOTE: can calculate both error rates:
			ScoreKaldiWER() : Word Error Rate
			ScoreKaldiCER() : Character Error Rate
		*/
		if (ScoreKaldiWER(data_dir, graph_dir, decode_dir, wer_ref_filter, wer_hyp_filter, nj, 
						  stage, decode_mbr, stats, beam, word_ins_penalty, min_lmwt, max_lmwt, iter) < 0) {
			LOGTW_ERROR << "Scoring failed.";
			return -1;
		}

		//NOTE: CER is not implemented yet (does not seem to be important now).

	}
	
	//clean up
	try	{
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			fs::path temppath(sdata / std::to_string(JOBID) / "apply_cmvn.temp");
			if (fs::exists(temppath)) fs::remove(temppath);
			fs::path temppath2(sdata / std::to_string(JOBID) / "add_deltas.temp");
			if (fs::exists(temppath2)) fs::remove(temppath2);
		}
	} catch (const std::exception&) {}

	return 0;
}



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
	parallel job for GmmLatgenFaster()

	NOTE: the string_vec options must not be passed by reference and make a copy because of the JOBID's!
*/
static void LaunchJobGmmLatgenFaster(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_transformfeats_trans,
	string_vec options_gmmlatgen,
	std::string feat_type,
	fs::path trans_dir,
	fs::path sdata,
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log << ".";

	//replace 'JOBID' with the current job ID of the thread
	for (std::string &s : options_splicefeats) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_transformfeats) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_transformfeats_trans) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_gmmlatgen) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	int ret = 0;
	//NOTE: all options (input/output) are distributed already well, therefore we just need to call the functions
	//		in the right sequence and all parameters will be OK!
	std::string outcmvn;
	if (feat_type == "delta") {
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
	}
	else {
		//DO: apply-cmvn + splice-feats + transform-feats
		//apply-cmvn ->
		//NOTE: options_adddeltas will not be used here!
		try {
			ret = ApplyCmvnSequence(JOBID, options_applycmvn, options_adddeltas, sdata, outcmvn, false, file_log);
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
		//splice-feats ->
		try {
			StrVec2Arg args(options_splicefeats);
			ret = SpliceFeats(args.argc(), args.argv());
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (SpliceFeats). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
		if (ret < 0) {
			//do not proceed if failed
			_ret.push_back(ret);
			return;
		}
		//transform-feats ->
		try {
			StrVec2Arg args(options_transformfeats);
			ret = TransformFeats(args.argc(), args.argv(), file_log);
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (TransformFeats). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
		if (ret < 0) {
			//do not proceed if failed
			_ret.push_back(ret);
			return;
		}
	}

	if (trans_dir != "" && fs::exists(trans_dir)) 
	{
		//DO: transform-feats with options_transformfeats_trans
		try {
			StrVec2Arg args(options_transformfeats_trans);
			ret = TransformFeats(args.argc(), args.argv(), file_log);
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (TransformFeats). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
		if (ret < 0) {
			//do not proceed if failed
			_ret.push_back(ret);
			return;
		}
	}

	//DO: gmm-latgen-faster
	try {
		StrVec2Arg args(options_gmmlatgen);
		ret = GmmLatgenFaster(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmLatgenFaster). Reason: " << ex.what();
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
