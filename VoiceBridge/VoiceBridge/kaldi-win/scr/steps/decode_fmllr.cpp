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

static void LaunchJobGmmEstFmllrGpost(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_l2p,
	string_vec options_wsp,
	string_vec options_gp2g,
	string_vec options_gefg,
	std::string feat_type,
	fs::path sdata,
	fs::path log
);

static void LaunchJobGmmLatgenFaster(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_pass1feats,
	string_vec options_gmmlatgen,
	std::string feat_type,
	fs::path sdata,
	fs::path log
);

static void LaunchJobComposeTransforms(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_pass1feats,
	string_vec options_ldp,
	string_vec options_l2p,
	string_vec options_wsp,
	string_vec options_gef,
	string_vec options_ctr,
	std::string feat_type,
	fs::path sdata,
	fs::path log
);

static void LaunchJobLaticeDetPruned(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_pass1feats,
	string_vec options_ldp,
	string_vec options_grl,
	std::string feat_type,
	fs::path sdata,
	fs::path log
);

//the return values from each thread/job
static std::vector<int> _ret;

/*
	There are 3 models involved potentially in this script, and for a standard, speaker-independent system they will 
	all be the same. The "alignment model" is for the 1st-pass decoding and to get the Gaussian-level alignments for 
	the "adaptation model" the first time we do fMLLR.  The "adaptation model" is used to estimate fMLLR transforms
	and to generate state-level lattices.  The lattices are then rescored with the "final model".
	The following table explains where we get these 3 models from.
	Note: srcdir is one level up from the decoding directory.
	   Model              Default source:
	
	  "alignment model"   srcdir/final.alimdl					--alignment-model <model>
	                  (or srcdir/final.mdl if alimdl absent)
	  "adaptation model"  srcdir/final.mdl						--adapt-model <model>
	  "final model"       srcdir/final.mdl						--final-model <model>
*/
VOICEBRIDGE_API int DecodeFmllr(
	fs::path graphdir,							//graph-dir
	fs::path data,								//data-dir
	fs::path dir,								//decode-dir - is assumed to be a sub-directory of the directory where the model is.
	fs::path alignment_model,					//default:final.mdl Model to get Gaussian-level alignments for 1st pass of transform computation.
	fs::path adapt_model,						//default:final.mdl Model to compute transforms with
	fs::path final_model,						//default:final.mdl Model to finally decode with
	int nj,										//default: 4, number of parallel jobs
	float acwt,									//default 0.08333 ... used to get posteriors;
	int stage,									//
	int max_active,								//
	double beam,								//Pruning beam [applied after acoustic scaling]
	double lattice_beam,						//
	double silence_weight,
	fs::path si_dir,							//use this to skip 1st pass of decoding, caution-- must be with same tree
	std::string fmllr_update_type,	
	//scoring options:
	bool skip_scoring,							//
	bool decode_mbr, 							//maximum bayes risk decoding (confusion network).
	bool stats, 								//output statistics
	std::string word_ins_penalty, 				//word insertion penalty
	int min_lmwt, 								//minumum LM-weight for lattice rescoring
	int max_lmwt 								//maximum LM-weight for lattice rescoring
)
{
	double first_beam = 10.0; // Beam used in initial, speaker - indep.pass
	double first_max_active = 2000; // max - active used in initial pass.

	fs::path srcdir(dir.parent_path()); //The model directory is one level up from decoding directory.
	fs::path sdata(data / ("split" + std::to_string(nj)));
	if (CreateDir(dir / "log", true) < 0) {
		LOGTW_ERROR << "Failed to create " << (dir / "log").string();
		return -1;
	}
	//
	if (!(fs::exists(sdata) && fs::last_write_time(data / "feats.scp") < fs::last_write_time(sdata))) {
		//split data directory
		if (SplitData(data, nj) < 0) return -1;
	}
	//save num_jobs
	StringTable t_njs;
	string_vec _njs = { std::to_string(nj) };
	t_njs.push_back(_njs);
	if (SaveStringTable((dir / "num_jobs").string(), t_njs) < 0) return -1;

	std::string splice_opts, cmvn_opts, delta_opts, silphonelist;

	try	{
		if (fs::exists(srcdir / "splice_opts")) //frame-splicing options
			splice_opts = GetFirstLineFromFile((srcdir / "splice_opts").string());
		if (fs::exists(srcdir / "cmvn_opts"))
			cmvn_opts = GetFirstLineFromFile((srcdir / "cmvn_opts").string());
		if (fs::exists(srcdir / "delta_opts"))
			delta_opts = GetFirstLineFromFile((srcdir / "delta_opts").string());

		boost::algorithm::trim(splice_opts);
		boost::algorithm::trim(cmvn_opts);
		boost::algorithm::trim(delta_opts);

		if (CheckFileExistsAndNotEmpty(graphdir / "phones" / "silence.csl", true) < 0) return -1;
		silphonelist = GetFirstLineFromFile((graphdir / "phones" / "silence.csl").string());
	}
	catch (const std::exception& ex)
	{
		LOGTW_ERROR << ex.what();
		return -1;
	}

	//check if all required files exist
	//NOTE: we don't need srcdir/tree but we expect it should exist. 
	std::vector<fs::path> required = { graphdir / "HCLG.fst", data / "feats.scp", srcdir / "tree" };
	for (fs::path p : required) {
		if (!fs::exists(p)) {
			LOGTW_ERROR << "Failed to find " << p.string();
			return -1;
		}
	}

	//Work out name of alignment model.
	if (alignment_model == "" || !fs::exists(alignment_model) || fs::is_empty(alignment_model)) {
		if (fs::exists(srcdir / "final.alimdl") && !fs::is_empty(srcdir / "final.alimdl"))
			alignment_model = srcdir / "final.alimdl";
		else if(fs::exists(srcdir / "final.mdl") && !fs::is_empty(srcdir / "final.mdl"))
			alignment_model = srcdir / "final.mdl";
		else {
			LOGTW_ERROR << "Failed to find alignment model.";
			return -1;
		}
	}

	//Do the speaker-independent decoding, if si-dir option is not present.
	if (si_dir == "" || !fs::exists(si_dir)) 
	{
		//Name it as our decoding dir, but with suffix ".si".
		si_dir = dir.string() + ".si";
		if (stage <= 0)
		{
			if (fs::exists(graphdir / "num_pdfs"))
			{
				//read the original num_pdfs
				std::string snpdfs("");
				try {
					snpdfs = GetFirstLineFromFile((graphdir / "num_pdfs").string());
				}
				catch (const std::exception&) {}
				int num_pdfs_orig = StringToNumber<int>(snpdfs, -1);
				if (num_pdfs_orig < 0) {
					LOGTW_ERROR << "Could not read number of pdf's from file " << (graphdir / "num_pdfs").string() << ".";
					return -1;
				}
				//get number of pfd's from model info
				int nofphones, num_pdfs, noftransitionids, noftransitionstates;
				if (AmInfo(alignment_model.string(), nofphones, num_pdfs, noftransitionids, noftransitionstates) < 0) return -1;
				if (num_pdfs_orig != num_pdfs) {
					LOGTW_ERROR << "Mismatch in number of pdfs with model in " << alignment_model.string();
					return -1;
				}
			}

			UMAPSS wer_ref_filter;						//ref filter NOTE: can be empty but must be defined!
			UMAPSS wer_hyp_filter; 						//hyp filter NOTE: can be empty but must be defined!
			if (Decode(
				graphdir,								//graph_dir
				data,									//data_dir
				si_dir,									//decode_dir
				alignment_model,						//when not specified "final.mdl" is taken automatically from the folder one above the decode directory
				"",										//trans_dir is the directory to find fMLLR transforms; this option won't normally be used, but it can be used if you want to supply existing fMLLR transforms when decoding			
				wer_ref_filter,							//
				wer_hyp_filter,							//
				"",										//iteration of model to test e.g. 'final', if the model is given then this option is not needed
				nj,										//the number of parallel threads to use in the decoding; must be the same as in the data preparation
				acwt,
				0, 
				first_max_active, 
				first_beam,
				6.0,									//default value 6.0
				//scoring options:
				skip_scoring, decode_mbr, stats, word_ins_penalty,min_lmwt,max_lmwt
			) < 0)
			{
				LOGTW_ERROR << "First pass speaker-independent decoding failed.";
				return -1;
			}
		}
	}

	//Some checks, and setting of defaults for variables
	std::string snj("");
	try {
		snj = GetFirstLineFromFile((si_dir / "num_jobs").string());
	}
	catch (const std::exception&) {}
	int nj_orig = StringToNumber<int>(snj, -1);
	if (nj_orig < 1) {
		LOGTW_ERROR << "Could not read number of jobs from file " << (si_dir / "num_jobs").string() << ".";
		return -1;
	}
	if (nj != nj_orig) {
		LOGTW_ERROR << "Mismatch in number of jobs with si-dir.";
		return -1;
	}
	if (adapt_model == "" || !fs::exists(adapt_model) || fs::is_empty(adapt_model)) adapt_model = srcdir / "final.mdl";
	if (final_model == "" || !fs::exists(final_model) || fs::is_empty(final_model)) final_model = srcdir / "final.mdl";
	//check if all required files exist
	std::vector<fs::path> required2 = { si_dir / "lat.1", adapt_model, final_model };
	for (fs::path p : required2) {
		if (!fs::exists(p)) {
			LOGTW_ERROR << "Failed to find " << p.string();
			return -1;
		}
	}

	//feature type:
	std::string feat_type;
	if (fs::exists(srcdir / "final.mat"))
		feat_type = "lda";
	else feat_type = "delta";
	LOGTW_INFO << "Feature type is " << feat_type;

	//
	//Prepare all parameters for the funcions which will be called in the parallel processing unit
	//
	string_vec options_applycmvn, options_adddeltas, options_splicefeats, options_transformfeats;
	//prepare the options for apply-cmvn			
	options_applycmvn.push_back("--print-args=false"); //NOTE: do not print arguments
	//parse and add cmvn_opts (space delimited collection of options on one line)
	string_vec _cmvn_opts;
	//NOTE: cmvn_opts must be 1 line of options delimited by " "
	strtk::parse(cmvn_opts, " ", _cmvn_opts, strtk::split_options::compress_delimiters);
	for each(std::string s in _cmvn_opts) options_applycmvn.push_back(s);
	//NOTE: JOBID will need to be replaced later!
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

	//Getting first-pass fMLLR transforms
	if (stage <= 1) {

		//Now get the first-pass fMLLR transforms.
		LOGTW_INFO << "Getting first-pass fMLLR transforms...";

		string_vec options_l2p, options_wsp, options_gp2g, options_gefg;
		//l2p : lattice-to-post
		options_l2p.push_back("--print-args=false");
		options_l2p.push_back("--acoustic-scale=" + std::to_string(acwt));
		options_l2p.push_back("ark:" + (si_dir / "lat.JOBID").string());
		options_l2p.push_back("ark:" + (si_dir / "lat.JOBID.temp").string()); //output
		//wsp : weight-silence-post
		options_wsp.push_back("--print-args=false");
		options_wsp.push_back(std::to_string(silence_weight));
		options_wsp.push_back(silphonelist);
		options_wsp.push_back(alignment_model.string());
		options_wsp.push_back("ark:" + (si_dir / "lat.JOBID.temp").string());
		options_wsp.push_back("ark:" + (si_dir / "wsp.JOBID.temp").string()); //output
		//gp2g : gmm-post-to-gpost
		options_gp2g.push_back("--print-args=false");
		options_gp2g.push_back(alignment_model.string());
		if (feat_type == "lda")
			options_gp2g.push_back("ark,s,cs:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from transform-feats
		else
			options_gp2g.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add_deltas	
		options_gp2g.push_back("ark:" + (si_dir / "wsp.JOBID.temp").string());
		options_gp2g.push_back("ark:" + (si_dir / "gp2g.JOBID.temp").string()); //output
		//gefg : gmm-est-fmllr-gpost
		options_gefg.push_back("--print-args=false");
		options_gefg.push_back("--fmllr-update-type=" + fmllr_update_type);
		options_gefg.push_back("--spk2utt=ark:" + (sdata / "JOBID" / "spk2utt").string());
		options_gefg.push_back(adapt_model.string());
		if (feat_type == "lda")
			options_gefg.push_back("ark,s,cs:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from transform-feats
		else
			options_gefg.push_back("ark,s,cs:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add_deltas

		options_gefg.push_back("ark,s,cs:" + (si_dir / "gp2g.JOBID.temp").string());
		options_gefg.push_back("ark:" + (dir / "pre_trans.JOBID").string());

		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(dir / "log" / ("fmllr_pass1." + std::to_string(JOBID) + ".log"));
			_threads.emplace_back(
				LaunchJobGmmEstFmllrGpost,
				JOBID,
				options_applycmvn, options_adddeltas, options_splicefeats, options_transformfeats,
				options_l2p, options_wsp, options_gp2g, options_gefg,
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
		DeleteAllMatching(si_dir, boost::regex(".*(\\.temp)$"));
		for (int JOBID = 1; JOBID <= nj; JOBID++)
			DeleteAllMatching(sdata / std::to_string(JOBID), boost::regex(".*(\\.temp)$"));
	}

	//2nd transform-feats options when Using transforms from alidir!
	string_vec options_pass1feats;
	options_pass1feats.push_back("--print-args=false");
	options_pass1feats.push_back("--utt2spk=ark:" + (sdata / "JOBID" / "utt2spk").string());
	options_pass1feats.push_back("ark:" + (dir / "pre_trans.JOBID").string());
	if (feat_type == "lda")
		options_pass1feats.push_back("ark:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from first transform-feats
	else
		options_pass1feats.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
	//IMPORTANT: the next option must be the last option because it may be read later as output path!
	options_pass1feats.push_back("ark:" + (sdata / "JOBID" / "transform_pass1feats.temp").string()); //output from second transform-feats

	/*
		Do the main lattice generation pass.  Note: we don't determinize the lattices at this stage, 
		as we're going to use them in acoustic rescoring with the larger model, and it's more correct 
		to store the full state-level lattice for this purpose.
	*/
	if (stage <= 2) {
		LOGTW_INFO << "Doing main lattice generation phase...";

		//check compatibility
		if (fs::exists(graphdir / "num_pdfs"))
		{
			//read the original num_pdfs
			std::string snpdfs("");
			try {
				snpdfs = GetFirstLineFromFile((graphdir / "num_pdfs").string());
			}
			catch (const std::exception&) {}
			int num_pdfs_orig = StringToNumber<int>(snpdfs, -1);
			if (num_pdfs_orig < 0) {
				LOGTW_ERROR << "Could not read number of pdf's from file " << (graphdir / "num_pdfs").string() << ".";
				return -1;
			}
			//get number of pfd's from model info
			int nofphones, num_pdfs, noftransitionids, noftransitionstates;
			if (AmInfo(adapt_model.string(), nofphones, num_pdfs, noftransitionids, noftransitionstates) < 0) return -1;
			if (num_pdfs_orig != num_pdfs) {
				LOGTW_ERROR << "Mismatch in number of pdfs with model in " << adapt_model.string();
				return -1;
			}
		}

		//options GmmLatgenFaster
		string_vec options_gmmlatgen;
		options_gmmlatgen.push_back("--print-args=false");
		options_gmmlatgen.push_back("--max-active=" + std::to_string(max_active));
		options_gmmlatgen.push_back("--beam=" + std::to_string(beam));
		options_gmmlatgen.push_back("--lattice-beam=" + std::to_string(lattice_beam));
		options_gmmlatgen.push_back("--acoustic-scale=" + std::to_string(acwt));
		options_gmmlatgen.push_back("--determinize-lattice=false");
		options_gmmlatgen.push_back("--allow-partial=true");
		options_gmmlatgen.push_back("--word-symbol-table=" + (graphdir / "words.txt").string());
		options_gmmlatgen.push_back(adapt_model.string());
		options_gmmlatgen.push_back((graphdir / "HCLG.fst").string());
		options_gmmlatgen.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_pass1feats.temp").string()); //output from pass1feats
		options_gmmlatgen.push_back("ark:"+(dir / "lat.tmp.JOBID").string()); //output

		//make sure that there are no old lat.* files in the output directory beacuse all 'lat.*' will be used later!
		if (DeleteAllMatching(dir, boost::regex("^(lat\\.).*")) < 0) return -1;

		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(dir / "log" / ("decode." + std::to_string(JOBID) + ".log"));
			_threads.emplace_back(
				LaunchJobGmmLatgenFaster,
				JOBID,
				options_applycmvn,
				options_adddeltas,
				options_splicefeats,
				options_transformfeats,
				options_pass1feats,
				options_gmmlatgen,
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

	}

	/*
		Do a second pass of estimating the transform-- this time with the lattices generated from the alignment model.
		Compose the transforms to get dir / trans.1, etc.
	*/
	if (stage <= 3) {
		LOGTW_INFO << "Estimating fMLLR transforms a second time...";

		string_vec options_ldp, options_l2p, options_wsp, options_gef, options_ctr;
		//ldp : lattice-determinize-pruned
		options_ldp.push_back("--print-args=false");
		options_ldp.push_back("--acoustic-scale="+std::to_string(acwt));
		options_ldp.push_back("--beam=" + std::to_string(4.0));
		options_ldp.push_back("ark:" + (dir / "lat.tmp.JOBID").string());
		options_ldp.push_back("ark:" + (dir / "ldp.JOBID.temp").string()); //output
		//l2p : lattice-to-post
		options_l2p.push_back("--print-args=false");
		options_l2p.push_back("--acoustic-scale=" + std::to_string(acwt));
		options_l2p.push_back("ark:" + (dir / "ldp.JOBID.temp").string());
		options_l2p.push_back("ark:" + (dir / "l2p.JOBID.temp").string()); //output
		//wsp : weight-silence-post
		options_wsp.push_back("--print-args=false");
		options_wsp.push_back(std::to_string(silence_weight));
		options_wsp.push_back(silphonelist);
		options_wsp.push_back(adapt_model.string());
		options_wsp.push_back("ark:" + (dir / "l2p.JOBID.temp").string());
		options_wsp.push_back("ark:" + (dir / "wsp.JOBID.temp").string()); //output
		//gef : gmm-est-fmllr 
		options_gef.push_back("--print-args=false");
		options_gef.push_back("--fmllr-update-type=" + fmllr_update_type);
		options_gef.push_back("--spk2utt=ark:" + (sdata / "JOBID" / "spk2utt").string());
		options_gef.push_back(adapt_model.string());
		options_gef.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_pass1feats.temp").string()); //output from pass1feats
		options_gef.push_back("ark,s,cs:" + (dir / "wsp.JOBID.temp").string());
		options_gef.push_back("ark:"+(dir / "trans_tmp.JOBID").string()); //output
		//ctr : compose-transforms 
		options_ctr.push_back("--print-args=false");
		options_ctr.push_back("--b-is-affine=true");
		options_ctr.push_back("ark:" + (dir / "trans_tmp.JOBID").string());
		options_ctr.push_back("ark:" + (dir / "pre_trans.JOBID").string());
		options_ctr.push_back("ark:" + (dir / "trans.JOBID").string()); //output

		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(dir / "log" / ("fmllr_pass2." + std::to_string(JOBID) + ".log"));
			_threads.emplace_back(
				LaunchJobComposeTransforms,
				JOBID,
				options_applycmvn,
				options_adddeltas,
				options_splicefeats,
				options_transformfeats,
				options_pass1feats,
				options_ldp, options_l2p, options_wsp, options_gef, options_ctr,
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
		DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
	}
	
	//recreate the pass1feats options but now with dir/trans.JOB as input instead of dir/pre_trans.JOB
	options_pass1feats.clear();
	options_pass1feats.push_back("--print-args=false");
	options_pass1feats.push_back("--utt2spk=ark:" + (sdata / "JOBID" / "utt2spk").string());
	options_pass1feats.push_back("ark:" + (dir / "trans.JOBID").string());
	if (feat_type == "lda")
		options_pass1feats.push_back("ark:" + (sdata / "JOBID" / "transformfeats.temp").string()); //output from first transform-feats
	else
		options_pass1feats.push_back("ark:" + (sdata / "JOBID" / "add_deltas.temp").string()); //output from add-deltas
	//IMPORTANT: the next option must be the last option because it may be read later as output path!
	options_pass1feats.push_back("ark:" + (sdata / "JOBID" / "transform_pass1feats.temp").string()); //output from second transform-feats

	/*
	Rescore the state-level lattices with the final adapted features, and the final model (which by default is srcdir/final.mdl, 
	but which may be specified on the command line, useful in case of discriminatively trained systems). At this point we prune 
	and determinize the lattices and write them out, ready for language model rescoring.
	*/
	if (stage <= 4) {
		LOGTW_INFO << "Doing a final pass of acoustic rescoring...";

		//GmmRescoreLattice
		string_vec options_ldp, options_grl;
		options_grl.push_back("--print-args=false");
		options_grl.push_back(final_model.string());
		options_grl.push_back("ark:" + (dir / "lat.tmp.JOBID").string());
		options_grl.push_back("ark,s,cs:" + (sdata / "JOBID" / "transform_pass1feats.temp").string()); //output from second transform-feats
		options_grl.push_back("ark:" + (dir / "grl.JOBID.temp").string()); //output

		//ldp : lattice-determinize-pruned
		options_ldp.push_back("--print-args=false");
		options_ldp.push_back("--acoustic-scale=" + std::to_string(acwt));
		options_ldp.push_back("--beam=" + std::to_string(lattice_beam));
		options_ldp.push_back("ark:" + (dir / "grl.JOBID.temp").string()); //<=
		options_ldp.push_back("ark:" + (dir / "lat.JOBID").string());  //output

		//---------------------------------------------------------------------
		//Start parallel processing
		std::vector<std::thread> _threads;
		_ret.clear();
		for (int JOBID = 1; JOBID <= nj; JOBID++)
		{
			//logfile
			fs::path log(dir / "log" / ("acoustic_rescore." + std::to_string(JOBID) + ".log"));
			_threads.emplace_back(
				LaunchJobLaticeDetPruned,
				JOBID,
				options_applycmvn,
				options_adddeltas,
				options_splicefeats,
				options_transformfeats,
				options_pass1feats,
				options_ldp, options_grl,
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

		//cleanup
		for (int JOBID = 1; JOBID <= nj; JOBID++) {
			DeleteAllMatching(sdata / std::to_string(JOBID), boost::regex(".*(\\.temp)$"));
			if (fs::exists(dir / ("lat.tmp." + std::to_string(JOBID))))
				fs::remove(dir / ("lat.tmp." + std::to_string(JOBID)));
		}
		DeleteAllMatching(dir, boost::regex(".*(\\.temp)$"));
	} ///stage <= 4

	if (stage <= 5)
	{
		if (AnalyzeLats(graphdir, dir) < 0) return -1;
	}

	if (!skip_scoring)
	{
		/*NOTE: can calculate both error rates:
		ScoreKaldiWER() : Word Error Rate
		ScoreKaldiCER() : Character Error Rate
		*/
		UMAPSS wer_ref_filter;						//ref filter NOTE: can be empty but must be defined!
		UMAPSS wer_hyp_filter; 						//hyp filter NOTE: can be empty but must be defined!
		//
		if (ScoreKaldiWER(data, graphdir, dir, wer_ref_filter, wer_hyp_filter, nj,
			stage, decode_mbr, stats, beam, word_ins_penalty, min_lmwt, max_lmwt, "") < 0) {
			LOGTW_ERROR << "Scoring failed.";
			return -1;
		}

		//NOTE: CER is not implemented yet (does not seem to be important now).

	}

	//cleanup
	for (int JOBID = 1; JOBID <= nj; JOBID++)
		if (fs::exists(dir / ("trans_tmp." + std::to_string(JOBID))))
			fs::remove(dir / ("trans_tmp." + std::to_string(JOBID)));

	for (int JOBID = 1; JOBID <= nj; JOBID++)
		if (fs::exists(dir / ("pre_trans." + std::to_string(JOBID))))
			fs::remove(dir / ("pre_trans." + std::to_string(JOBID)));

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
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	std::string feat_type,
	fs::path sdata, 
	std::string & out,
	fs::ofstream & file_log)
{
	string_vec optionscmvn, optionsadddeltas, optionssplice, optionstransformfeats;
	std::string search("JOBID"), replace(std::to_string(jobid));

	for each(std::string s in options_applycmvn) {
		ReplaceStringInPlace(s, search, replace);
		optionscmvn.push_back(s);
	}

	if (feat_type == "lda") {
		for each(std::string s in options_transformfeats) {
			//replace JOBID with jobid
			ReplaceStringInPlace(s, search, replace);
			optionstransformfeats.push_back(s);
		}
		for each(std::string s in options_splicefeats) {
			ReplaceStringInPlace(s, search, replace);
			optionssplice.push_back(s);
		}
	}
	else {
		for each(std::string s in options_adddeltas) {
			ReplaceStringInPlace(s, search, replace);
			optionsadddeltas.push_back(s);
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
			StrVec2Arg args(optionssplice), argstranseats(optionstransformfeats);
			//SpliceFeats
			if (SpliceFeats(args.argc(), args.argv()) < 0) {
				LOGTW_ERROR << "Error while splicing.";
				return -1;
			}
			//TransformFeats
			if (TransformFeats(argstranseats.argc(), argstranseats.argv(), file_log) < 0) {
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
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in ApplyCmvnSequence. Reason: " << ex.what();
		return -1;
	}

	//get the output file path
	if (feat_type == "lda") {
		out = optionstransformfeats[optionstransformfeats.size() - 1];
	}
	else {
		out = optionsadddeltas[optionsadddeltas.size() - 1];
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
	string_vec options_pass1feats,
	string_vec options_gmmlatgen,
	std::string feat_type,
	fs::path sdata,
	fs::path log
)
{
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log << ".";

	//replace 'JOBID' with the current job ID of the thread
	for (std::string &s : options_pass1feats) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_gmmlatgen) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	int ret = 0;
	//NOTE: all options (input/output) are distributed already well, therefore we just need to call the functions
	//		in the right sequence and all parameters will be OK!
	std::string outcmvn;
	//DO: apply-cmvn + add-deltas		
	try {
		ret = ApplyCmvnSequence(JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splicefeats,
			options_transformfeats,
			feat_type,
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

	//DO: transform-feats with options_pass1feats
	try {
		StrVec2Arg args(options_pass1feats);
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


static void LaunchJobGmmEstFmllrGpost(
	int JOBID,
	string_vec options_applycmvn, 
	string_vec options_adddeltas, 
	string_vec options_splicefeats, 
	string_vec options_transformfeats, 
	string_vec options_l2p, 
	string_vec options_wsp, 
	string_vec options_gp2g, 
	string_vec options_gefg,
	std::string feat_type,
	fs::path sdata,
	fs::path log
)
{
	/*
		lattice-to-post => weight-silence-post  => gmm-post-to-gpost => gmm-est-fmllr-gpost
	*/
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
	for (std::string &s : options_l2p) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_wsp) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_gp2g) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_gefg) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	std::string outcmvn;
	int ret = 0;

	try {
		ret = ApplyCmvnSequence(JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splicefeats,
			options_transformfeats,
			feat_type,
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

	//lattice-to-post
	try {
		StrVec2Arg args(options_l2p);
		ret = LatticeToPost(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (LatticeToPost). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//weight-silence-post
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

	//gmm-post-to-gpost
	try {
		StrVec2Arg args(options_gp2g);
		ret = GmmPostToGpost(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmPostToGpost). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//gmm-est-fmllr-gpost
	try {
		StrVec2Arg args(options_gefg);
		ret = GmmEstFmllrGpost(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmEstFmllrGpost). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret);
}


static void LaunchJobComposeTransforms(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_pass1feats,
	string_vec options_ldp, 
	string_vec options_l2p, 
	string_vec options_wsp, 
	string_vec options_gef, 
	string_vec options_ctr,
	std::string feat_type,
	fs::path sdata,
	fs::path log
)
{
	/*
		lattice-determinize-pruned => lattice-to-post  => weight-silence-post => gmm-est-fmllr => compose-transforms
	*/
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
	for (std::string &s : options_ldp) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_l2p) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_wsp) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_gef) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_ctr) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_pass1feats) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	std::string outcmvn;
	int ret = 0;

	try {
		ret = ApplyCmvnSequence(JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splicefeats,
			options_transformfeats,
			feat_type,
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

	//DO: transform-feats with options_pass1feats
	try {
		StrVec2Arg args(options_pass1feats);
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

	//lattice-determinize-pruned
	try {
		StrVec2Arg args(options_ldp);
		ret = LatticeDeterminizePruned(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (LatticeDeterminizePruned). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	//lattice-to-post
	try {
		StrVec2Arg args(options_l2p);
		ret = LatticeToPost(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (LatticeToPost). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//weight-silence-post
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

	//gmm-est-fmllr
	try {
		StrVec2Arg args(options_gef);
		ret = GmmEstFmllr(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmEstFmllr). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//compose-transforms
	try {
		StrVec2Arg args(options_ctr);
		ret = ComposeTransforms(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (ComposeTransforms). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret);
}


static void LaunchJobLaticeDetPruned(
	int JOBID,
	string_vec options_applycmvn,
	string_vec options_adddeltas,
	string_vec options_splicefeats,
	string_vec options_transformfeats,
	string_vec options_pass1feats,
	string_vec options_ldp, 
	string_vec options_grl,
	std::string feat_type,
	fs::path sdata,
	fs::path log
)
{
	/*
	gmm-rescore-lattice  => lattice-determinize-pruned
	*/
	//we redirect logging to the log file:
	fs::ofstream file_log(log, fs::ofstream::binary | fs::ofstream::out);
	if (!file_log) LOGTW_WARNING << "Log file is not accessible " << log.string() << ".";

	//replace 'JOBID' with the current job ID of the thread; must do in this way because JOBID is added outside of this loop also!
	for (std::string &s : options_ldp) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_grl) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));
	for (std::string &s : options_pass1feats) ReplaceStringInPlace(s, "JOBID", std::to_string(JOBID));

	std::string outcmvn;
	int ret = 0;

	try {
		ret = ApplyCmvnSequence(JOBID,
			options_applycmvn,
			options_adddeltas,
			options_splicefeats,
			options_transformfeats,
			feat_type,
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

	//DO: transform-feats with options_pass1feats
	try {
		StrVec2Arg args(options_pass1feats);
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

	//gmm-rescore-lattice
	try {
		StrVec2Arg args(options_grl);
		ret = GmmRescoreLattice(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (GmmRescoreLattice). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}
	if (ret < 0) {
		//do not proceed if failed
		_ret.push_back(ret);
		return;
	}

	//lattice-determinize-pruned
	try {
		StrVec2Arg args(options_ldp);
		ret = LatticeDeterminizePruned(args.argc(), args.argv(), file_log);
	}
	catch (const std::exception& ex)
	{
		LOGTW_FATALERROR << "Error in (LatticeDeterminizePruned). Reason: " << ex.what();
		_ret.push_back(-1);
		return;
	}

	_ret.push_back(ret);
}
