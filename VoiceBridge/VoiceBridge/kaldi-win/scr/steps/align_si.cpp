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

static void LaunchJobGmmAlignCompiled(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_ctgraphs,
	string_vec options_gmmalignedcomp,
	bool use_graphs,
	std::string feat_type,
	fs::path sdata,
	std::string symtab, std::string input_txt, std::string output_txt, int field_begin, int field_end, std::string oov,	//params for Sym2Int
	fs::path log
);

//the return values from each thread/job
static std::vector<int> _ret;

/*
	Computes training alignments using a model with delta or LDA+MLLT features.
	It checks if the training graphs exist and compatible with the chosen number of jobs (nj),
	if yes then it will use the training graphs from the source directory (where the model is).
	If not then it will generate the training graphs automatically.
*/
VOICEBRIDGE_API int AlignSi(
	fs::path data,		//data directory
	fs::path lang,		//language directory
	fs::path srcdir,	//trained model directory
	fs::path dir,		//output directory
	int nj,				//number of jobs
	double boost_silence,
	double transitionscale, double acousticscale, double selfloopscale,
	int beam, int retry_beam,
	bool careful)
{
	//check if necessary files exist
	std::vector<fs::path> f = {data/"text", lang/"oov.int", srcdir/"tree", srcdir/"final.mdl"};
	for (fs::path p : f) {
		if (!fs::exists(p)) {
			LOGTW_ERROR << "Expected file " << p.string() << " to exist.";
			return -1;
		}
	}

	//get oov
	if (CheckFileExistsAndNotEmpty(lang / "oov.int", true) < 0) return -1;
	StringTable t_oov;
	if (ReadStringTable((lang / "oov.int").string(), t_oov) < 0) return -1;
	std::string oov(t_oov[0][0]);

	//create log directory
	if (CreateDir(dir / "log", true) < 0) return -1;

	//save num_jobs
	StringTable t_njs;
	string_vec _njs = { std::to_string(nj) };
	t_njs.push_back(_njs);
	if (SaveStringTable((dir / "num_jobs").string(), t_njs) < 0) return -1;

	//Get extra options : saved in model directory while training the model (srcdir). 
	//Should use the same extra parameters! May all be empty.
	std::vector<std::string> splice_opts;	//extra options for splice
	std::vector<std::string> cmvn_opts;		//extra options for apply cmvn
	std::vector<std::string> delta_opts;	//extra options for apply delta
	fs::path po_s(srcdir / "splice_opts"), po_c(srcdir / "cmvn_opts"), po_d(srcdir / "delta_opts");
	std::string spliceopts, cmvnopts, deltaopts;
	try {
		if (fs::exists(po_s) && !fs::is_empty(po_s)) spliceopts = GetFirstLineFromFile(po_s.string());
		if (spliceopts != "") strtk::parse(spliceopts, " ", splice_opts, strtk::split_options::compress_delimiters);
	}
	catch (const std::exception&) {}
	try {
		if (fs::exists(po_c) && !fs::is_empty(po_c)) cmvnopts = GetFirstLineFromFile(po_c.string());
		if (cmvnopts != "") strtk::parse(cmvnopts, " ", cmvn_opts, strtk::split_options::compress_delimiters);
	}
	catch (const std::exception&) {}
	try {
		if (fs::exists(po_d) && !fs::is_empty(po_d)) deltaopts = GetFirstLineFromFile(po_d.string());
		if (deltaopts != "") strtk::parse(deltaopts, " ", delta_opts, strtk::split_options::compress_delimiters);
	}
	catch (const std::exception&) {}
	//save all options to output directory for further use
	fs::path poo_s(dir / "splice_opts"), poo_c(dir / "cmvn_opts"), poo_d(dir / "delta_opts");
	fs::ofstream ofs_s(poo_s, fs::ofstream::binary | fs::ofstream::out), ofs_c(poo_c, fs::ofstream::binary | fs::ofstream::out), ofs_d(poo_d, fs::ofstream::binary | fs::ofstream::out);
	if (!ofs_s || !ofs_c || !ofs_d) {
		LOGTW_ERROR << "Failed to save options in " << dir.string();
		return -1;
	}
	for (std::string s : splice_opts) ofs_s << s << " ";
	for (std::string s : cmvn_opts) ofs_c << s << " ";
	for (std::string s : delta_opts) ofs_d << s << " ";
	ofs_s.flush(); ofs_s.close();
	ofs_c.flush(); ofs_c.close();
	ofs_d.flush(); ofs_d.close();

	//extra sanity check
	if (CheckPhonesCompatible(lang / "phones.txt", srcdir / "phones.txt") < 0) return -1;

	try	{
		fs::copy_file(lang / "phones.txt", dir / "phones.txt", fs::copy_option::overwrite_if_exists);
		fs::copy_file(srcdir / "tree", dir / "tree", fs::copy_option::overwrite_if_exists);
		fs::copy_file(srcdir / "final.mdl", dir / "final.mdl", fs::copy_option::overwrite_if_exists);
		fs::copy_file(srcdir / "final.occs", dir / "final.occs", fs::copy_option::overwrite_if_exists);
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << ex.what();
		return -1;
	}

	//feature type:
	std::string feat_type;
	if (fs::exists(srcdir / "final.mat")) {
		feat_type = "lda";
		try	{
			fs::copy_file(srcdir / "final.mat", dir / "final.mat", fs::copy_option::overwrite_if_exists);
			fs::copy_file(srcdir / "full.mat", dir / "full.mat", fs::copy_option::overwrite_if_exists);
		}
		catch (const std::exception& ex) {
			LOGTW_ERROR << ex.what();
			return -1;
		}
	}
	else feat_type = "delta";
	LOGTW_INFO << "feature type is " << feat_type;

	LOGTW_INFO << "Aligning data in " << data.filename().string() << ", using model from " << srcdir.filename().string() 
			   << ", putting alignments in:";
	LOGTW_INFO << dir.string();

	//determine if graphs have to be created; if do not exist then make them automatically from the model
	//read number of jobs from the srcdir (this was used while splitting the data!)
	bool use_graphs = true; //if exists and compatible (nj)
	std::string snj;
	int nj_orig;
	try {
		snj = GetFirstLineFromFile((srcdir / "num_jobs").string());
	}
	catch (const std::exception&) {}
	nj_orig = StringToNumber<int>(snj, -1);
	if (nj_orig < 1) {
		LOGTW_ERROR << " could not read number of jobs from file " << (srcdir / "num_jobs").string() << ".";
		return -1;
	}
	//check compatibility
	if (nj != nj_orig)
	{
		LOGTW_WARNING << "The number of jobs used for the model mismatches alignment destination. Must recreate graphs.";
		use_graphs = false;
	}
	//check if all files exist
	for (int JOBID = 1; JOBID <= nj; JOBID++) {
		if (!fs::exists(srcdir / ("fsts." + std::to_string(JOBID)))) {
			LOGTW_WARNING << (srcdir / ("fsts." + std::to_string(JOBID))).string() << " does not exist. Must recreate graphs.";
			use_graphs = false;
			break;
		}
	}

	//check if the data is aready split into nj parts; if not then split it!
	fs::path sdata(data / ("split" + std::to_string(nj)));
	if (!(fs::exists(sdata) && fs::is_directory(sdata) && fs::last_write_time(data / "feats.scp") < fs::last_write_time(sdata)))
	{
		//split data directory
		if (SplitData(data, nj) < 0) return -1;
	}

	//
	//Make alignment ------------->
	//
	fs::ofstream file_log(dir / "log" / "boostsilence.log", fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << (dir / "log" / "boostsilence.log").string() << ".";

	//1. Boost Silence 
	//get the contents of lang/phones/optional_silence.csl
	std::string optional_silence;
	try {
		optional_silence = GetFirstLineFromFile((lang / "phones" / "optional_silence.csl").string());
	}
	catch (const std::exception&) {
		LOGTW_ERROR << "Can not read " << (lang / "phones" / "optional_silence.csl").string() << ".";
		return -1;
	}
	string_vec options;
	options.push_back("--print-args=false");
	options.push_back("--boost=" + std::to_string(boost_silence));
	options.push_back(optional_silence);	//e.g.: "1:2:3"
	options.push_back((dir / "final.mdl").string());
	options.push_back((dir / "final.bs.temp").string()); //output
	StrVec2Arg args(options);
	if (GmmBoostSilence(args.argc(), args.argv(), file_log) < 0) return -1;

	//2. prepare global options	
	//
	//Prepare all parameters for the funcions which will be called in the parallel processing unit
	//
	string_vec options_applycmvn, options_adddeltas, options_splicefeats, options_transformfeats, options_ctgraphs;
	//prepare the options for apply-cmvn			
	options_applycmvn.push_back("--print-args=false"); //NOTE: do not print arguments
	//parse and add cmvn_opts (space delimited collection of options on one line)
	for each(std::string s in cmvn_opts) options_applycmvn.push_back(s);
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
		for each(std::string s in delta_opts) options_adddeltas.push_back(s);
		options_adddeltas.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //input from apply-cmvn
		//IMPORTANT: the next option must be the last option because it is read later as output path!
		options_adddeltas.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
	}
	else //if (feat_type == "lda")
	{
		//splice-feats 
		options_splicefeats.push_back("--print-args=false");
		for each(std::string s in splice_opts) options_splicefeats.push_back(s);
		options_splicefeats.push_back("ark:" + (sdata / "JOBID" / "apply_cmvn.temp").string()); //input from apply-cmvn
		options_splicefeats.push_back("ark:" + (sdata / "JOBID" / "splicefeats.temp").string()); //output from splice-feats
		//transform-feats 
		options_transformfeats.push_back("--print-args=false");
		options_transformfeats.push_back((srcdir / "final.mat").string());
		options_transformfeats.push_back("ark:" + (sdata / "JOBID" / "splicefeats.temp").string()); //input from splice-feats
		options_transformfeats.push_back("ark:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from transform-feats
	}
			
	//options for compile-train-graphs	
	//Sym2Int
	std::string symtab((lang / "words.txt").string());
	std::string input_txt((sdata / "JOBID" / "text").string());
	std::string output_txt((dir / "trnrspec.JOBID.temp").string()); //output from Sym2Int
	//oov is defined above
	int field_begin = 1; //NOTE: zero based index of fields! 2nd field is index=1
	int field_end = -1;	
	//
	options_ctgraphs.push_back("--print-args=false");
	options_ctgraphs.push_back("--read-disambig-syms=" + (lang / "phones" / "disambig.int").string());
	options_ctgraphs.push_back((dir / "tree").string());
	options_ctgraphs.push_back((dir / "final.mdl").string());
	options_ctgraphs.push_back((lang / "L.fst").string());
	options_ctgraphs.push_back((dir / "trnrspec.JOBID.temp").string()); //output from Sym2Int
	options_ctgraphs.push_back("ark:" + (dir / "fsts.JOBID").string()); //output

	//options GmmAlignCompiled
	string_vec options_gmmalignedcomp;
	options_gmmalignedcomp.push_back("--print-args=false");
	std::string scareful = (careful ? "true" : "false");
	options_gmmalignedcomp.push_back("--careful=" + scareful);
	options_gmmalignedcomp.push_back("--beam=" + std::to_string(beam));
	options_gmmalignedcomp.push_back("--retry-beam=" + std::to_string(retry_beam));
	options_gmmalignedcomp.push_back("--transition-scale=" + std::to_string(transitionscale));
	options_gmmalignedcomp.push_back("--acoustic-scale=" + std::to_string(acousticscale));
	options_gmmalignedcomp.push_back("--self-loop-scale=" + std::to_string(selfloopscale));
	options_gmmalignedcomp.push_back((dir / "final.bs.temp").string()); //output from boost silence!

	//in case we need to make the graphs then make them in 'dir' and use them from there
	if (!use_graphs) {
		//use the output from compile-train-graphs
		options_gmmalignedcomp.push_back("ark:" + (dir / "fsts.JOBID").string());
	}
	else {
		//we have the graphs in the srcdir
		options_gmmalignedcomp.push_back("ark:" + (srcdir / "fsts.JOBID").string());
	}

	if (feat_type == "delta") {
		options_gmmalignedcomp.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
	}
	else {
		options_gmmalignedcomp.push_back("ark:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from transform-feats
	}
	options_gmmalignedcomp.push_back("ark:" + (dir / "ali.JOBID").string()); //output 


	//---------------------------------------------------------------------
	//Start parallel processing
	std::vector<std::thread> _threads;
	_ret.clear();
	for (int JOBID = 1; JOBID <= nj; JOBID++)
	{
		//logfile
		fs::path log(dir / "log" / ("align." + std::to_string(JOBID) + ".log"));
		//
		_threads.emplace_back(
			LaunchJobGmmAlignCompiled,
			JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splicefeats,
			options_transformfeats,
			options_ctgraphs,
			options_gmmalignedcomp,
			use_graphs,
			feat_type,
			sdata,
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

	//diagnostics:
	if (AnalyzeAlignments(lang, dir) < 0) return -1;

	//cleanup
	try {
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			std::string p((sdata / "JOBID").string());
			ReplaceStringInPlace(p, "JOBID", std::to_string(JOBID));
			DeleteAllMatching(p, boost::regex(".*(\\.temp)$"));
		}
		if(fs::exists(dir / "final.bs.temp")) fs::remove(dir / "final.bs.temp");
	}
	catch (const std::exception&) {}


	LOGTW_INFO << "Done aligning data.";
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
	parallel job for GmmAlignCompiled()

	NOTE: the string_vec options must not be passed by reference and make a copy because of the JOBID's!
*/
static void LaunchJobGmmAlignCompiled(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_ctgraphs,
	string_vec options_gmmalignedcomp,
	bool use_graphs,
	std::string feat_type,
	fs::path sdata,
	std::string symtab, std::string input_txt, std::string output_txt, int field_begin, int field_end, std::string oov,	//params for Sym2Int
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log << ".";

	//replace 'JOBID' with the current job ID of the thread
	for (std::string &s : options_splicefeats) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_transformfeats) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_gmmalignedcomp) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	ReplaceStringInPlace(input_txt, "JOBID", std::to_string(JOBID));
	ReplaceStringInPlace(output_txt, "JOBID", std::to_string(JOBID));

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

	//
	if (!use_graphs)
	{ //need first to create graphs //NOTE: in the options_gmmalignedcomp the output is set accordingly!

		//create graphs
		//Sym2Int
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

		//compile train graphs
		try {
			StrVec2Arg args(options_ctgraphs);
			ret = CompileTrainGraphs(args.argc(), args.argv(), file_log);
		}
		catch (const std::exception& ex)
		{
			LOGTW_FATALERROR << "Error in (CompileTrainGraphs). Reason: " << ex.what();
			_ret.push_back(-1);
			return;
		}
		if (ret < 0) {
			//do not proceed if failed
			_ret.push_back(ret);
			return;
		}
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
