/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/

#pragma once
#include <kaldi-win/stdafx.h>
#include "kaldi-win/utility/Utility.h"
#include "kaldi-win/src/fstbin/fst_ext.h"

using string_vec = std::vector<std::string>;
using UMAPSS = std::unordered_map<std::string, std::string>;
using MSYMLINK = std::map<std::string, std::string>;

extern VOICEBRIDGE_API std::vector<std::string> * Get_silencephones();
extern VOICEBRIDGE_API std::vector<std::string> * Get_nonsilencephones();

int ReadSilenceAndNonSilencePhones(fs::path fSilencephones, fs::path fNonSilencephones);

int ValidateDict(fs::path pthdict);
int ValidateData(fs::path datadir,
	bool no_feats = false,
	bool no_wav = false,
	bool no_text = false,
	bool no_spk_sort = false
);

VOICEBRIDGE_API int PrepareLang(bool position_dependent_phones, fs::path unk_fst, fs::path phone_symbol_table, fs::path extra_word_disambig_syms);
VOICEBRIDGE_API int PrepareData(int percentageTrain, std::string transc_ext, int orderngram = 3, int idtype = 1);
VOICEBRIDGE_API int PrepareDict(fs::path refDict,
	const std::map<std::string, std::string> & silphones, //silence phones e.g. !SIL SIL, <UNK> SPN
	const std::map<std::string, std::string> & optsilphones); //optional silence phones e.g. !SIL SIL
VOICEBRIDGE_API int PrepareTestLms(const std::vector<std::string> & lms);

int ValidateDisambigSymFile(fs::path disambig_sym_file, bool bAllowNumeric);
int CheckUTF8AndWhiteSpace(fs::path _path, bool check_last_char_is_nl);
int CheckDuplicates(fs::path _path, bool check_valid_ending, std::vector<std::string> & _v);
int CheckLexicon(fs::path lp, int num_prob_cols, int num_skipped_cols);
int CheckSilprob(fs::path lp);
int ModifyUnkPron(fs::path lexicon_file, std::string unk_word);
int AddLexDisambig(bool pron_probs, bool sil_probs, int first_allowed_disambig, fs::path lexiconp_silprob, fs::path lexiconp_silprob_disambig);

int ApplyMap(StringTable map, StringTable input_txt, fs::path output_txt, int field_begin, int field_end, bool bPermissive);
int Sym2Int(StringTable symtab, StringTable input_txt, fs::path output_txt, int field_begin, int field_end, std::string map_oov);
int Int2Sym(StringTable symtab, StringTable input_txt, fs::path output_txt, int field_begin, int field_end);
int MakeLexiconFstSilprob(StringTable lexfn, StringTable silprobfile, fs::path output_txt, std::string silphone, std::string sildisambig);
int MakeLexiconFst(StringTable lexfn, fs::path output_txt, bool pron_probs, double silprob, std::string silphone, std::string sildisambig);
int GenerateTopology(int num_nonsil_states, int num_sil_states, StringTable nonsil_phones, StringTable sil_phones, fs::path path_topo_output);
int ApplyUnkLM(StringTable input_unk_lm_fst, fs::path lang_dir);

int CheckFileExistsAndNotEmpty(fs::path file, bool bShowError);
int GetSecondFieldFromStringTable(StringTable table, std::string key, std::string path, int * val);

int ValidateLang(fs::path langdir, bool skip_determinization_check, bool skip_disambig_check);

int CheckGProperties(fs::path langdir);

int arpa2fst(std::string arpa_rxfilename, std::string fst_wxfilename,
	// Option flags:
	std::string disambig_symbol = "", //""			Disambiguator. If provided (e.g. #0), used on input side of backoff links, and <s> and </s> are replaced with epsilons
	std::string read_syms_filename = "",	//""	e.g. "data/lang/words.txt"
	std::string write_syms_filename = "", //""		Write generated symbol table to a file
	std::string bos_symbol = "<s>", // = "<s>"		Beginning of sentence symbol
	std::string eos_symbol = "</s>", //= "</s>";	End of sentence symbol
	bool keep_symbols = false, // = false;			Store symbol table with FST. Symbols always saved to FST if symbol tables are neither read or written (otherwise symbols would be lost entirely)
	bool ilabel_sort = true); //= true				Ilabel-sort the output FST

VOICEBRIDGE_API int MakeMfcc(
	fs::path datadir,			//data directory
	fs::path mfcc_config,		//mfcc config file path
	int nj=4,						//default: 4, number of parallel jobs
	bool compress=true,				//default: true, compress mfcc features
	bool write_utt2num_frames=false	//default: false, if true writes utt2num_frames	
);

int spk2utt_to_utt2spk(StringTable & spk2utt, StringTable & utt2spk);
int utt2spk_to_spk2utt(StringTable & spk2utt, StringTable & utt2spk);

//int CreateDataLink(std::vector<fs::path> _fullpaths, MSYMLINK & msymlinkstorage);

int SplitScp(fs::path inscp, std::vector<fs::path> output_segments, int num_jobs=0, int job_id=-1, fs::path utt2spk_file="");
int SplitData(fs::path datadir, int numsplit, bool per_utt=false);

VOICEBRIDGE_API int ComputeCmvnStats(
	fs::path datadir,			//data directory
	bool fake = false,			//default: false, gives you fake cmvn stats that do no normalization.
	bool two_channel = false,	//default: false, is for two-channel telephone data, there must be no segments
								//file and reco2file_and_channel must be present. It will take only frames 
								//that are louder than the other channel.
	std::string fake_dims=""	//Generate stats that won't cause normalization for these dimensions (e.g. "13:14:15")
);

VOICEBRIDGE_API int FixDataDir(fs::path datadir, std::vector<fs::path> spk_extra_files = {}, std::vector<fs::path> utt_extra_files = {});
int FilterScp(fs::path idlist, fs::path in_scp, fs::path out_scp, bool exclude=false, int field=0);
int FilterScps(int jobstart, int jobend, fs::path idlist, fs::path in_scp, fs::path out_scp, bool no_warn=false, int field=0);

/*training of a Gmm monophone model*/
VOICEBRIDGE_API int TrainGmmMono(
	fs::path datadir,			//data directory
	fs::path langdir,			//lang directory
	fs::path traindir,			//training directory
	fs::path config,			//config file path
	int nj,						//number of parallel jobs
	int stage = -4				//stage; can be used to skip some steps done before
);

int AnalyzeAlignments(
	fs::path datadir,			//data directory
	fs::path traindir			//training directory
);

int AnalyzePhoneLengthStats(
	std::vector<fs::path> & _phonestats,	//input phone stats files
	fs::path lang,							//lang directory	
	float frequencyCutoffPercentage=0.5f	//default = 0.5, Cutoff, expressed as a percentage (between 0 and 100), of frequency at which we print stats for a phone. 
);


VOICEBRIDGE_API int MkGraph(fs::path lang_dir, fs::path model_dir, fs::path graph_dir,
	bool remove_oov=false, //If true, any paths containing the OOV symbol (obtained from oov.int in the lang directory) are removed from the G.fst during compilation.
	double tscale=1.0,	 //Scaling factor on transition probabilities.
	double loopscale=0.1 //see: http://kaldi-asr.org/doc/hmm.html#hmm_scale
);

int AnalyzeLats(
	fs::path lang,				//lang directory
	fs::path dir,				//training directory
	std::string iter = "final",	//e.g. 'final' (default: final)
	float acwt = 0.1f				//Acoustic scale for getting best-path (default: 0.1)
);

int AnalyzeLatticeDepthStats(
	std::vector<fs::path> & _phonestats,	//input phone stats files
	fs::path lang,							//lang directory	
	float frequencyCutoffPercentage = 0.5f	//default = 0.5, Cutoff, expressed as a percentage (between 0 and 100), 
											//of frequency at which we print stats for a phone. 
);

VOICEBRIDGE_API int Decode(
	fs::path graph_dir,							//
	fs::path data_dir,							//
	fs::path decode_dir,						//is assumed to be a sub-directory of the directory where the model is.
	fs::path model,								//which model to use e.g. final.mdl
	fs::path trans_dir,							//dir to find fMLLR transforms; this option won't normally be used, 
												//but it can be used if you want to supply existing fMLLR transforms when decoding	
	UMAPSS & wer_ref_filter,					//ref filter NOTE: can be empty but must be defined!
	UMAPSS & wer_hyp_filter,					//hyp filter NOTE: can be empty but must be defined!
	std::string iter = "",						//Iteration of model to test e.g. 'final'
	int nj = 4,									//default: 4, number of parallel jobs
	float acwt = 0.083333f,						//acoustic scale used for lattice generation;  NOTE: only really affects pruning (scoring is on lattices).
	int stage = 0,
	int max_active = 7000,
	double beam = 13.0,
	double lattice_beam = 6.0,
	bool skip_scoring = false,	
	//scoring options:
	bool decode_mbr = false,						//maximum bayes risk decoding (confusion network).
	bool stats = true,								//output statistics
	std::string word_ins_penalty = "0.0,0.5,1.0",	//word insertion penalty
	int min_lmwt = 7,								//minumum LM-weight for lattice rescoring
	int max_lmwt = 17								//maximum LM-weight for lattice rescoring
);


int ScoreKaldiWER(
	fs::path data,
	fs::path lang_or_graph,
	fs::path dir,
	UMAPSS & wer_ref_filter,
	UMAPSS & wer_hyp_filter,
	int nj,
	int stage = 0,
	bool decode_mbr = false,
	bool stats = true,
	double beam = 6.0,
	std::string word_ins_penalty = "0.0,0.5,1.0",
	int min_lmwt = 7,
	int max_lmwt = 17,
	std::string iter = "final"
);

int ScoreKaldiCER(
	fs::path data,
	fs::path lang_or_graph,
	fs::path dir,
	UMAPSS wer_ref_filter,
	UMAPSS wer_hyp_filter,
	int nj,
	int stage = 0,
	bool decode_mbr = false,
	bool stats = true,
	double beam = 6.0,
	std::string word_ins_penalty = "0.0,0.5,1.0",
	int min_lmwt = 7,
	int max_lmwt = 17,
	std::string iter = "final"
);

int BestWer(std::vector<fs::path> _wer, std::string & best_wip, int & best_lmwt);

int WerPerUttDetails(
	fs::path in,
	fs::path out,
	std::string special_symbol = "<eps>",
	std::string separator = ";",
	bool output_hyp = true,
	bool output_ref = true,
	bool output_ops = true,
	bool output_csid = true);

int WerPerSpkDetails(
	fs::path in,
	fs::path utt2spk,
	fs::path out,
	int WIDTH = 10,
	int SPK_WIDTH = 15
);

int WerOpsDetails(fs::path in,
	fs::path out,
	std::string special_symbol = "<eps>");

VOICEBRIDGE_API int AlignSi(
	fs::path data, 
	fs::path lang, 
	fs::path srcdir, 
	fs::path dir, 
	int nj,
	double boost_silence = 1.0,		//Factor by which to boost silence during alignment.
	double transitionscale = 1.0,	//scale options:
	double acousticscale = 0.1, 
	double selfloopscale = 0.1,
	int beam = 10,					//GmmAlignCompiled options:
	int retry_beam = 40,
	bool careful = false			//
	);	

int CheckPhonesCompatible(fs::path table_first, fs::path table_second);

VOICEBRIDGE_API int TrainDeltas(
	fs::path data,				//data directory
	fs::path lang,				//language directory
	fs::path alidir,			//directory with alignments
	fs::path dir,				//output
	int nj,						//number of threads
	fs::path config,			//config file path with various options //TODO:... document
	double boost_silence = 1.0, //
	int numleaves = 2000,		//
	int totgauss = 10000,		//
	int stage = -4				//stage; can be used to skip some steps done before
	);

VOICEBRIDGE_API int TrainLdaMllt(
	fs::path data,				//data directory
	fs::path lang,				//language directory
	fs::path alidir,			//directory with alignments
	fs::path dir,				//output directory
	int nj,						//number of threads
	fs::path config,			//config file path with various options //TODO:... document
	double boost_silence = 1.0,	//factor by which to boost silence likelihoods in alignment
	int numleaves = 2000,		//gauss algo param
	int totgauss = 10000,		//gauss algo param
	int stage = -5				//stage; can be used to skip some steps done before
);

VOICEBRIDGE_API int TrainSat(
	fs::path data,			//data directory
	fs::path lang,			//language directory
	fs::path alidir,		//directory with the aligned base model (tri1)
	fs::path dir,			//output directory
	int nj,					//number of threads
	fs::path config,		//config file path with various options //TODO:... document
	double boost_silence,	//factor by which to boost silence likelihoods in alignment
	int numleaves,			//gauss algo param
	int totgauss,			//gauss algo param
	int stage = -5			//stage; can be used to skip some steps done before
);

VOICEBRIDGE_API int DecodeFmllr(
	fs::path graph_dir,								//graphdir
	fs::path data_dir,								//data
	fs::path decode_dir,							//dir - is assumed to be a sub-directory of the directory where the model is.
	fs::path alignment_model = "",					//Model to get Gaussian-level alignments for 1st pass of transform computation.
	fs::path adapt_model = "",						//Model to compute transforms with
	fs::path final_model="",						//Model to finally decode with
	int nj=4,										//default: 4, number of parallel jobs
	float acwt= 0.083333,							//default 0.08333 ... used to get posteriors;
	int stage=0,									//
	int max_active=7000,							//
	double beam=13.0,								//Pruning beam [applied after acoustic scaling]
	double lattice_beam=6.0,						//
	double silence_weight=0.01,
	fs::path si_dir="",								//use this to skip 1st pass of decoding, caution-- must be with same tree
	std::string fmllr_update_type="full",
	//scoring options:
	bool skip_scoring=false,						//
	bool decode_mbr = false,						//maximum bayes risk decoding (confusion network).
	bool stats = true,								//output statistics
	std::string word_ins_penalty = "0.0,0.5,1.0",	//word insertion penalty
	int min_lmwt = 7,								//minumum LM-weight for lattice rescoring
	int max_lmwt = 17								//maximum LM-weight for lattice rescoring
);

VOICEBRIDGE_API int GetProns(
	fs::path data,	
	fs::path lang,	
	fs::path dir,	
	int stage=1,	
	int lmwt=10);
