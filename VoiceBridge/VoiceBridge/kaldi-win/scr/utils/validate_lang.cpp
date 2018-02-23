/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright  2012   Guoguo Chen, Apache 2.0.
					  2014   Neil Nelson
					  2017   Johns Hopkins University (Jan "Yenda" Trmal <jtrmal@gmail.com>)
*/

//Validation for data/lang

#include "kaldi-win\scr\kaldi_scr.h"

using MSSI = std::map<std::string, std::map<std::string, int > >;
using MAPSS = std::unordered_map<std::string, std::string>;
int CheckWordsPhones(fs::path path_txt, MAPSS * xsymtab, MAPSS * xint2sym, int requiredcols);
int CheckDisjoint(fs::path langdir, bool skip_disambig_check, std::vector<std::string> * sum, std::vector<std::string> * silence, std::vector<std::string> * nonsilence, std::vector<std::string> * disambig);
int CheckSND(StringTable table_, std::vector<std::string>* v_, std::string name);
int CheckSummation(std::vector<std::string> * sum, MAPSS * psymtab);
void intersect(std::vector<std::string> * sum, MAPSS * psymtab, std::vector<std::string> * itset);
int CheckTxtIntCsl(fs::path file, MAPSS map);
int CheckTxtInt(fs::path file, MAPSS map, int sym_check, std::vector<std::string> * silence, std::vector<std::string> * nonsilence);
int CheckTopo(fs::path fileTopo, std::vector<std::string> * silence, std::vector<std::string> * nonsilence, MAPSS * pint2sym);
int CheckWordBoundary(fs::path file, std::vector<std::string> * silence, std::vector<std::string> * nonsilence, std::vector<std::string> * disambig);
int CheckWordBoundaryInt(fs::path file, std::vector<std::string> * disambig, MAPSS * wint2sym, MAPSS * pint2sym);

int ValidateLang(
	fs::path langdir,						// the language directory to validate
	bool skip_determinization_check,		// this flag causes it to skip a time consuming check
	bool skip_disambig_check)				// this flag causes it to skip a disambig check in phone bigram models
{
	bool warning = false;

	// Checking phones.txt ------------------------------
	fs::path phones_txt(langdir / "phones.txt");
	MAPSS psymtab, pint2sym;
	if (CheckWordsPhones(phones_txt, &psymtab, &pint2sym, 2) < 0) return -1;

	// Check words.txt ----------------------------------
	fs::path words_txt(langdir / "words.txt");
	MAPSS wsymtab, wint2sym;
	if (CheckWordsPhones(words_txt, &wsymtab, &wint2sym, 2) < 0) return -1;

	// Checking phones/* -------------------------------
	std::vector<std::string> sum, silence, nonsilence, disambig;
	CheckDisjoint(langdir, skip_disambig_check, &sum, &silence, &nonsilence, &disambig);
	CheckSummation(&sum, &psymtab);
	//
	std::vector<std::string> list1 = { "context_indep", "nonsilence", "silence", "optional_silence" };
	std::vector<std::string> list2 = { "roots", "sets" };
	if (!skip_disambig_check) list1.push_back("disambig");	
	//list1 .txt, .int, .csl
	for each(std::string file in list1) {
		if (CheckTxtIntCsl(langdir / ("phones\\" + file), psymtab) < 0) return -1;
	}
	//list2
	for each(std::string file in list2) {
		if (CheckTxtInt(langdir / ("phones\\" + file), psymtab, 1, &silence, &nonsilence) < 0) return -1;
	}
	//extra_questions.txt, .int
	fs::path eq_txtint(langdir / ("phones\\extra_questions"));
	fs::path eq_txt(langdir / ("phones\\extra_questions.txt"));
	fs::path eq_int(langdir / ("phones\\extra_questions.int"));
	if (!fs::exists(eq_txt)) {
		LOGTW_WARNING << " extra_questions.txt does not exist (may be empty, but should be present). Creating file...";
		//NOTE: ofstream write adds an '\r' in front of the '\n' automatically and some code in Kaldi crashes. To prevent this
		//		add std::ios::binary option to each ofstream to make sure that '\r' is not added!
		fs::ofstream file_(eq_txt, std::ios::binary);
		file_.flush(); file_.close();
	}
	if (!fs::exists(eq_int)) {
		LOGTW_WARNING << " extra_questions.int does not exist (may be empty, but should be present). Creating file...";
		fs::ofstream file_(eq_int, std::ios::binary);
		file_.flush(); file_.close();
	}
	if (CheckFileExistsAndNotEmpty(eq_txt, false) >= 0 && CheckFileExistsAndNotEmpty(eq_int, false) >= 0) { 
		if (CheckTxtInt(eq_txtint, psymtab, 0, &silence, &nonsilence) < 0) return -1;
	}
	//word_boundary.txt, .int
	fs::path wb_txtint(langdir / ("phones\\word_boundary"));
	fs::path wb_txt(langdir / ("phones\\word_boundary.txt"));
	fs::path wb_int(langdir / ("phones\\word_boundary.int"));
	if (CheckFileExistsAndNotEmpty(wb_txt, false) >= 0 && CheckFileExistsAndNotEmpty(wb_int, false) >= 0)
		if (CheckTxtInt(wb_txtint, psymtab, 0, &silence, &nonsilence) < 0)
			return -1;

	// Checking optional_silence.txt----------------------
	LOGTW_INFO << "Checking optional_silence.txt ...";
	fs::path os_txt(langdir / ("phones\\optional_silence.txt"));
	if (CheckFileExistsAndNotEmpty(os_txt, true) < 0) {
		LOGTW_ERROR << " file is empty or does not exist: " << os_txt.string() << ".";
		return -1;
	}
	StringTable table_os;
	if (ReadStringTable(os_txt.string(), table_os) < 0) {
		LOGTW_ERROR << " fail to open: " << os_txt.string() << ".";
		return -1;
	}
	int idx = 1;
	for (StringTable::const_iterator it(table_os.begin()), it_end(table_os.end()); it != it_end; ++it)
	{
		if (idx > 1 || (*it).size() > 1) {
			LOGTW_ERROR << " only 1 phone expected in " << os_txt.string() << ".";
			return -1;
		} else if(std::find(silence.begin(), silence.end(), (*it)[0]) == silence.end()) {
			LOGTW_ERROR << " phone " << (*it)[0] << " not found in phones/silence_phones.txt.";
			return -1;
		}
		idx++;
	}
	LOGTW_INFO << "  --> " << os_txt.string() << " is OK.";

	// Check disambiguation symbols---------------------
	if (!skip_disambig_check) {
		LOGTW_INFO << "Checking disambiguation symbols: #0 and #1";
		if (std::find(disambig.begin(), disambig.end(), "#0") == disambig.end() ||
			std::find(disambig.begin(), disambig.end(), "#1") == disambig.end())
		{
			LOGTW_WARNING << " phones/disambig.txt doesn't have \"#0\" or \"#1\";";
			LOGTW_INFO << "  -->          this would not be OK with a conventional ARPA-type language.";
			LOGTW_INFO << "  -->          model or a conventional lexicon (L.fst).";
			warning = true;
		}
		else {
			LOGTW_INFO << "  --> phones/disambig.txt has \"#0\" and \"#1\".";
			LOGTW_INFO << "  --> phones/disambig.txt is OK.";
		}
	}

	// Check topo---------------------------------------
	LOGTW_INFO << "Checking topo ...";
	fs::path path_topo(langdir / "topo");
	if (CheckTopo(path_topo, &silence, &nonsilence, &pint2sym) < 0) return -1;

	// Check word_boundary------------------------------
	if (CheckFileExistsAndNotEmpty(wb_txt, false) >= 0)
	{
		LOGTW_INFO << "Checking word_boundary.txt: silence.txt, nonsilence.txt, disambig.txt ...";
		if (CheckWordBoundary(wb_txt, &silence, &nonsilence, &disambig) < 0) return -1;
	}

	// Check wdisambig----------------------------------
	LOGTW_INFO << "Checking word-level disambiguation symbols...";
	fs::path wdisambig_txt(langdir / ("phones\\wdisambig.txt"));
	fs::path wdisambig_p_int(langdir / ("phones\\wdisambig_phones.int"));
	fs::path wdisambig_w_int(langdir / ("phones\\wdisambig_words.int"));
	if (CheckFileExistsAndNotEmpty(wdisambig_txt, true) < 0 ||
		CheckFileExistsAndNotEmpty(wdisambig_p_int, true) < 0 ||
		CheckFileExistsAndNotEmpty(wdisambig_w_int, true) < 0)
	{
		/*
		NOTE: for lang diretories prepared by older versions of prepare_lang.sh the symbol '#0' should appear in 
		words.txt and phones.txt. We do not accept older versions here! The files wdisambig.txt, wdisambig_phones.int 
		and wdisambig_words.int must exist and have the expected properties (see below for details)!
		*/
		LOGTW_ERROR << " " << "for lang diretories prepared by older versions of the prepare_lang.sh script the symbol '#0'"
			"appears in words.txt and phones.txt. We do not accept this here anymore! The files wdisambig.txt, "
			"wdisambig_phones.int and wdisambig_words.int must exist and have the expected properties!";
		return -1;
	}

	// Check word_boundary.int & disambig.int ------------
	if (CheckFileExistsAndNotEmpty(wb_int, false) >= 0)
	{
		LOGTW_INFO << "Checking word_boundary.int and disambig.int...";
		if (CheckWordBoundaryInt(langdir, &disambig, &wint2sym, &pint2sym) < 0) return -1;
	}

	// Check oov-----------------------------------------
	fs::path path_oov(langdir / "oov");
	if (CheckTxtInt(path_oov, wsymtab, 0, &silence, &nonsilence) < 0) return -1;
	
	// Check if L.fst is olabel sorted ---------------------
	fs::path path_L_fst(langdir / "L.fst");
	if (CheckFileExistsAndNotEmpty(path_L_fst, false) >= 0) //NOTE: false here makes sure that no error is shown
	{
		//get info
		fst::FstInfo info;
		if (fstinfo(path_L_fst.string(), "any", "auto", true, true, &info) < 0) return -1;
		uint64 properties = info.Properties();
		//each property is set as a bit in properties
		//NOTE: properties are trinary - they are either true, false or unknown. For each such property, 
		//		there are two stored bits; one is set if true, the other is set if false and neither is set if unknown.
		//		Here: kOLabelSorted is set if true, kNotOLabelSorted is set if false, if neither is set then unkonwn!
		bool is_set = IsBitSet(properties, fst::kOLabelSorted);
		if (is_set) {
			LOGTW_INFO << "  --> lang/L.fst is olabel sorted.";
		}
		else {
			LOGTW_ERROR << " lang/L.fst is not olabel sorted.";
			return -1;
		}
	}

	//Check if L_disambig.fst is olabel sorted. --------------
	fs::path path_L_disambig(langdir / "L_disambig.fst");
	if (CheckFileExistsAndNotEmpty(path_L_disambig, false) >= 0) //NOTE: false here makes sure that no error is shown
	{
		//get info
		fst::FstInfo info;
		if (fstinfo(path_L_disambig.string(), "any", "auto", true, true, &info) < 0) return -1;
		uint64 properties = info.Properties();
		bool is_set = IsBitSet(properties, fst::kOLabelSorted);
		if (is_set) {
			LOGTW_INFO << "  --> lang/path_L_disambig.fst is olabel sorted.";
		}
		else {
			LOGTW_ERROR << " lang/path_L_disambig.fst is not olabel sorted.";
			return -1;
		}
	}

	//Check that G.fst is ilabel sorted and nonempty. ----------
	fs::path path_G_fst(langdir / "G.fst");
	if (CheckFileExistsAndNotEmpty(path_G_fst, false) >= 0) //NOTE: false here makes sure that no error is shown
	{
		//get info
		fst::FstInfo info;
		if (fstinfo(path_G_fst.string(), "any", "auto", true, true, &info) < 0) return -1;
		uint64 properties = info.Properties();
		bool is_set = IsBitSet(properties, fst::kILabelSorted);
		if (is_set) {
			LOGTW_INFO << "  --> lang/G.fst is ilabel sorted.";
		}
		else {
			LOGTW_ERROR << " lang/G.fst is not ilabel sorted.";
			return -1;
		}
		//
		int num_states = info.NumStates();
		if (num_states == 0) {
			LOGTW_ERROR << " lang/G.fst is empty.";
		}
		else {
			LOGTW_INFO << "  --> lang/G.fst has " << num_states << " states.";
			return -1;
		}

		// Check that G.fst is determinizable. -----------------------
		if (!skip_determinization_check) {

			/* Check determinizability of G.fst fstdeterminizestar is much faster, and a more relevant test as it's what
			we do in the actual graph creation recipe.*/
			fs::path path_G_fst_tempout(langdir / "G.fst.temp");
			int ret = fstdeterminizestar(path_G_fst.string(), path_G_fst_tempout.string());
			try { //we do not need the output
				fs::remove(path_G_fst_tempout); 
			}
			catch (const std::exception&) {/*do nothing*/ }

			if (ret == 0) 
			{
				LOGTW_INFO << "  --> lang/G.fst is determinizable.";
			}
			else
			{
				LOGTW_ERROR << " fail to determinize lang/G.fst.";
				return -1;
			}

			fs::path path_L_disambig_fst(langdir / "L_disambig.fst");
			if (CheckFileExistsAndNotEmpty(path_L_disambig_fst, false) >= 0) //NOTE: false here makes sure that no error is shown
			{
				LOGTW_INFO << "  --> Testing determinizability of L_disambig . G.";

				if(fsttablecompose(path_L_disambig_fst.string(), path_G_fst.string(), path_G_fst_tempout.string()) < 0) return -1;
				fs::path path_G_fst_tempout1(langdir / "G.fst.temp1");
				if (fstdeterminizestar(path_G_fst_tempout.string(), path_G_fst_tempout1.string()) < 0) return -1;
				fst::FstInfo info;
				if (fstinfo(path_G_fst_tempout1.string(), "any", "auto", true, true, &info) < 0) return -1;
				try { //we do not need the output
					fs::remove(path_G_fst_tempout);
					fs::remove(path_G_fst_tempout1);
				}
				catch (const std::exception&) {/*do nothing*/ }
				//
				if (info.NumStates() > 0)
				{
					LOGTW_INFO << "  --> L_disambig . G is determinizable.";
				}
				else 
				{
					LOGTW_ERROR << " fail to determinize L_disambig . G.";
					return -1;
				}

			}
		} ///!skip_determinization_check

		// Check that G.fst does not have cycles with only disambiguation symbols or epsilons on the input, 
		// or the forbidden symbols <s> and < / s> (and a few related checks
		if (CheckGProperties(langdir) < 0)
		{
			LOGTW_ERROR << " failure while checking G.fst.";
			return -1;
		}
		else {
			LOGTW_INFO << "  --> All checks of G.fst succeeded.";
		}
	} ///path_G_fst exists

	if (warning) {
		LOGTW_WARNING << " (check output above for warnings).";
	}
	else {
		LOGTW_INFO << "  --> SUCCESS [validating lang directory " << langdir.string() << " ].";
	}
	return 0;
}


//NOTE: the MSSI structure map makes it possible to have an assosiative map in the form of
//		transition["begin"]["end"] = 0; 
//		Here we search for two keys and return true if they are found in the same element
bool findInMSSI(MSSI map, std::string s1, std::string s2)
{
	MSSI::const_iterator ci = map.find(s1);
	if (ci == map.end()) return false;
	std::map<std::string, int > m2 = ci->second;
	std::map<std::string, int >::const_iterator ci2 = m2.find(s2);
	if (ci2 == m2.end()) return false;
	return true;
}


int CheckWordBoundaryInt(fs::path langdir, std::vector<std::string> * disambig, MAPSS * wint2sym, MAPSS * pint2sym)
{
	StringTable table_wbi, table_dsi;
	if (ReadStringTable((langdir / ("phones\\word_boundary.int")).string(), table_wbi) < 0) {
		LOGTW_ERROR << " fail to open: " << (langdir / ("phones\\word_boundary.int")).string() << ".";
		return -1;
	}
	MAPSS wbtype;
	for (StringTable::const_iterator it(table_wbi.begin()), it_end(table_wbi.end()); it != it_end; ++it)
	{
		if ((*it).size() != 2) {
			LOGTW_ERROR << " bad line " << join_vector( (*it), " " ) << " in " << (langdir / ("phones\\word_boundary.int")).string() << ".";
			return -1;
		}
		wbtype.emplace((*it)[0], (*it)[1]);
	}

	if (ReadStringTable((langdir / ("phones\\word_boundary.int")).string(), table_wbi) < 0) {
		LOGTW_ERROR << " fail to open: " << (langdir / ("phones\\word_boundary.int")).string() << ".";
		return -1;
	}
	//
	std::vector<std::string> is_disambig;
	if (ReadStringTable((langdir / ("phones\\disambig.int")).string(), table_dsi) < 0) {
		LOGTW_ERROR << " fail to open: " << (langdir / ("phones\\disambig.int")).string() << ".";
		return -1;
	}
	for (StringTable::const_iterator it(table_dsi.begin()), it_end(table_dsi.end()); it != it_end; ++it)
	{
		if ((*it).size() != 1) {
			LOGTW_ERROR << " bad line " << join_vector((*it), " ") << " in " << (langdir / ("phones\\disambig.int")).string() << ".";
			return -1;
		}
		is_disambig.push_back((*it)[0]);
	}

	std::vector<fs::path> fst{ langdir / "L.fst", langdir / "L_disambig.fst" };
	for each(fs::path p in fst)
	{
		int wlen = int(std::rand() % 100) + 1;
		LOGTW_INFO << "  --> generating a " << wlen << " word sequence.";
		std::string wordseq = "";
		int sid = 0;
		std::string wordseq_syms = "";
		for (int i = 1; i <= wlen; i++) 
		{
			int id = int(std::rand() % wint2sym->size());
			// exclude disambiguation symbols, BOS and EOS and epsilon from the word sequence.
			MAPSS::const_iterator ci = wint2sym->find(std::to_string(id));
			while (ci != wint2sym->end() && (ci->second == "<s>" || ci->second == "</s>" || id == 0) ) {
				id = int(rand() % wint2sym->size());
			}					
			wordseq_syms.append(ci->second + " ");
			wordseq.append(std::to_string(sid) + " " + std::to_string(sid + 1) + " " + std::to_string(id) + " " + std::to_string(id) + " 0\n");
			sid++;
		}
		wordseq.append(std::to_string(sid) + " 0");
		//save temporary file
		fs::path path_temp_in(langdir / "wordseq.temp");
		fs::path path_temp_compiled(langdir / "wordseq_compiled.temp");
		fs::ofstream file_(path_temp_in, std::ios::binary);
		if (!file_) {
			LOGTW_ERROR << " can't open output file: " << path_temp_in.string() << ".";
			return -1;
		}		
		file_ << wordseq << "\n";
		file_.flush(); file_.close();
		try
		{
			fstcompile(false, "standard", "vector",
				path_temp_in.string(), path_temp_compiled.string(),
				"", "", "",
				false, false, false, false);

		}
		catch (const std::exception& message)
		{
			LOGTW_ERROR << " error while compiling fst's, check file: " << path_temp_in.string() << "\nDetail: " << message.what() << ".";
			return -1;
		}
				
		//NOTE:... could replace the below code with code which does not use files
		VectorFstClass * pofst = NULL;

		fs::path path_compose_out(langdir / "wordseq_compose.temp");
		fstcompose(p.string(), path_temp_compiled.string(), path_compose_out.string(), pofst);
		fs::path path_project_out(langdir / "wordseq_project.temp");
		fstproject(path_compose_out.string(), path_project_out.string());
		fs::path path_randgen_out(langdir / "wordseq_randgen.temp");
		fstrandgen(path_project_out.string(), path_randgen_out.string());
		fs::path path_rmepsilon_out(langdir / "wordseq_rmepsilon.temp");
		fstrmepsilon(path_randgen_out.string(), path_rmepsilon_out.string());
		fs::path path_topsort_out(langdir / "wordseq_topsort.temp");
		fsttopsort(path_rmepsilon_out.string(), path_topsort_out.string());
		fs::path path_print_out(langdir / "wordseq_print.temp");
		fstprint(path_topsort_out.string(), path_print_out.string());		
		StringTable table_wordseq;
		if (ReadStringTable(path_print_out.string(), table_wordseq) < 0) {
			LOGTW_ERROR << " fail to open: " << path_print_out.string() << ".";
			return -1;
		}
		//delete all temp files
		try {
			fs::remove(path_compose_out);
			fs::remove(path_project_out);
			fs::remove(path_randgen_out);
			fs::remove(path_rmepsilon_out);
			fs::remove(path_topsort_out);
			fs::remove(path_print_out);

		} catch (const std::exception&) {/*do nothing*/}

		std::vector<std::string> _phoneseq;
		for (StringTable::const_iterator it(table_wordseq.begin()), it_end(table_wordseq.end()); it != it_end; ++it)
		{
			if ((*it).size() > 2) {				
				if((*it)[2] != "")
					_phoneseq.push_back((*it)[2]);
			}
		}
		
		MSSI transition;
		for each(std::string x in std::vector<std::string>{"bos", "nonword", "end", "singleton"} )
		{
			transition[x]["nonword"] = 0;
			transition[x]["begin"] = 1;
			transition[x]["singleton"] = 1;
			transition[x]["eos"] = 0;
		}
		transition["begin"]["end"] = 0;
		transition["begin"]["internal"] = 0;
		transition["internal"]["internal"] = 0;
		transition["internal"]["end"] = 0;

		std::string cur_state = "bos";		
		int num_words = 0;
		_phoneseq.push_back("<<eos>>");

		for(std::string phone : _phoneseq) {
			// NOTE: now that we support unk-LMs (see the unk_fst option to prepare_lang), the regular L.fst 
			// may contain some disambiguation symbols.
			if (std::find(is_disambig.begin(), is_disambig.end(), phone) == is_disambig.end())
			{
				std::string state("");
				if (phone == "<<eos>>") {
					state = "eos";
				} else if(phone == "") {
					LOGTW_ERROR << " unexpected phone in sequence = " << phone << ", wordseq = " << wordseq << ".";
					return -1;
				}
				else {
					state = wbtype[phone];
				}
				if (state=="") {
					LOGTW_ERROR << " phone " << phone << " is not specified in phones/word_boundary.int.";
					return -1;
				}
				else if (!findInMSSI(transition, cur_state, state )) {
					LOGTW_ERROR << " transition from state " << cur_state << " to " << state << " indicates error in word_boundary.int or L.fst.";
					return -1;
				}
				else {
					num_words += transition[cur_state][state];
					cur_state = state;
				}
			}
		} ///for each(std::string phone in _phone

		if (num_words != wlen) {
			std::string phoneseq_syms = "";
			//must delete <<eos>> from the list for this step
			_phoneseq.erase(std::remove(_phoneseq.begin(), _phoneseq.end(), "<<eos>>"), _phoneseq.end()); 
			for (std::string phone : _phoneseq) 
			{ 
				phoneseq_syms.append(" " + pint2sym->at(phone));
			}
			LOGTW_ERROR << " number of reconstructed words " << num_words << " does not match real number of words " << wlen << "; indicates problem in " << p.string() << " or word_boundary.int. phoneseq = " << phoneseq_syms << ", wordseq = " << wordseq_syms << ".";
			return -1;
		}
		else {
			LOGTW_INFO << "  --> resulting phone sequence from " << p.string() << " corresponds to the word sequence.";
			LOGTW_INFO << "  --> " << p.string() << " is OK.";
		}
	} ///for each(fs::path p in fst

	//LOGTW_INFO << "\n";

	return 0;
}


int CheckWordBoundary(fs::path file, std::vector<std::string> * silence, std::vector<std::string> * nonsilence, std::vector<std::string> * disambig)
{
	//TODO: this part of the code is not tested!

	int idx = 1;
	std::string nonword = "";
	std::string begin = "";
	std::string end = "";
	std::string internal = "";
	std::string singleton = "";
	std::set<std::string> wb; //NOTE: sorted and unique!
	const boost::regex rem1("^.*nonword$");
	const boost::regex rem2("^.*begin$");
	const boost::regex rem3("^.*end$");
	const boost::regex rem4("^.*internal$");
	const boost::regex rem5("^.*singleton$");
	const boost::regex res1(" nonword$");
	const boost::regex res2(" begin$");
	const boost::regex res3(" end$");
	const boost::regex res4(" internal$");
	const boost::regex res5(" singleton$");
	//
	std::vector<const boost::regex*> _rem;
	std::vector<const boost::regex*> _res;
	std::vector<std::string*> _sw;
	_rem.push_back(&rem1); _rem.push_back(&rem2); _rem.push_back(&rem3); _rem.push_back(&rem4); _rem.push_back(&rem5);
	_res.push_back(&res1); _res.push_back(&res2); _res.push_back(&res3); _res.push_back(&res4); _res.push_back(&res5);
	_sw.push_back(&nonword); _sw.push_back(&begin); _sw.push_back(&end); _sw.push_back(&internal); _sw.push_back(&singleton);

	boost::match_results<std::string::const_iterator> re_res;
	std::ifstream ifs(file.string());
	if (!ifs) {
		LOGTW_ERROR << " Error opening file: " << file.string() << ".";
		return -1;
	}
	std::string line;
	int nLine = 0;
	std::vector<std::string> _words;
	while (std::getline(ifs, line)) {
		nLine++;
		for (int i = 0; i < 5; i++) 
		{ // there are 5 cases (see above)
			//take only lines which are matching the regex
			if (boost::regex_match(line, re_res, *_rem[i]))
			{
				std::string linem;
				linem = boost::regex_replace(line, *_res[i], "");
				_words.clear();
				strtk::parse(linem, " ", _words, strtk::split_options::compress_delimiters);
				if (_words.size() == 1) {
					(_sw[i])->append(_words[0] + " ");
				}
				if (_words.size() != 1) {
					LOGTW_ERROR << " expect 1 column in " << file.string() << " (line " << idx << ").";
					return -1;
				}
				wb.emplace(_words[0]);
				idx++;
			}
		}
	}
	ifs.close();

	bool success1 = true;
	std::vector<std::string> vI_DWB;
	//NOTE: set_intersection expects sorted vector
	std::sort(silence->begin(), silence->end());
	std::set_intersection(disambig->begin(), disambig->end(), wb.begin(), wb.end(), std::inserter(vI_DWB, vI_DWB.end()));
	if (vI_DWB.size() != 0) {
		success1 = false;
		std::stringstream ss;
		for each(std::string s in vI_DWB) 
			ss << s << " ";
		LOGTW_INFO << ss.str();
		LOGTW_ERROR << " phones/word_boundary.txt has disambiguation symbols -- ";
	}
	if(success1) LOGTW_INFO << "  --> phones/word_boundary.txt doesn't include disambiguation symbols.";

	//silence, nonsilence
	bool success2 = true;
	std::vector<std::string> sum;
	sum.insert(sum.end(), silence->begin(), silence->end());
	sum.insert(sum.end(), nonsilence->begin(), nonsilence->end());
	std::vector<std::string> vI_SWB;
	//NOTE: set_intersection expects sorted vector
	std::sort(sum.begin(), sum.end());
	std::set_intersection(sum.begin(), sum.end(), wb.begin(), wb.end(), std::inserter(vI_SWB, vI_SWB.end()));
	if (vI_SWB.size() < sum.size()) {
		success2 = false;
		std::stringstream ss;
		for each(std::string s in sum)
			if (std::find(vI_SWB.begin(), vI_SWB.end(), s) == vI_SWB.end())
				ss << s << " ";
		LOGTW_INFO << ss.str();
		LOGTW_ERROR << " phones in nonsilence.txt and silence.txt but not in word_boundary.txt -- ";
	}
	
	if (vI_SWB.size() < wb.size()) {
		success2 = false;
		std::stringstream ss;
		for each(std::string s in wb)
			if (std::find(vI_SWB.begin(), vI_SWB.end(), s) == vI_SWB.end())
				ss << s << " ";		
		LOGTW_INFO << ss.str();
		LOGTW_ERROR << " phones in word_boundary.txt but not in nonsilence.txt or silence.txt -- ";
	}
	if(success2) LOGTW_INFO << "  --> phones/word_boundary.txt is the union of nonsilence.txt and silence.txt.";
	if(success1 && success2) LOGTW_INFO << "  --> phones/word_boundary.txt is OK.";
	LOGTW_INFO << "\n";

	return 0;
}


int CheckTopo(fs::path fileTopo, std::vector<std::string> * silence, std::vector<std::string> * nonsilence, MAPSS * pint2sym)
{
	if (CheckFileExistsAndNotEmpty(fileTopo, true) < 0) {
		LOGTW_ERROR << " file is empty or does not exist: " << fileTopo.string() << ".";
		return -1;
	}
	bool topo_ok = true;
	int idx = 1;
	std::vector<int> phones_in_topo_int_hash;
	std::vector<std::string> phones_in_topo_hash;

	static const boost::regex re("^<.*>[ ]*$");
	boost::match_results<std::string::const_iterator> re_res;
	std::ifstream ifs(fileTopo.string());
	if (!ifs) {
		LOGTW_ERROR << " Error opening file: " << fileTopo.string() << ".";
		return -1;
	}
	std::string line;
	int nLine = 0;
	while (std::getline(ifs, line)) {
		nLine++;
		//take only lines which are not matching the regex
		if (!boost::regex_match(line, re_res, re))
		{
			std::vector<std::string> _words;
			strtk::parse(line, " ", _words, strtk::split_options::compress_delimiters);
			for (int i = 0; i < _words.size(); i++) 
			{
				if (!is_positive_int(_words[i])) {
					LOGTW_ERROR << " expected an integer but got " << _words[i] << " in " << fileTopo.string() << ".";
					return -1;
				}
				int iw = std::stoi(_words[i]);
				if (std::find(phones_in_topo_int_hash.begin(), phones_in_topo_int_hash.end(), iw) != phones_in_topo_int_hash.end())
				{
					LOGTW_ERROR << " lang/topo has phone " << _words[i] << " twice.";
					return -1;
				}
				MAPSS::const_iterator ci = pint2sym->find(_words[i]);
				if (ci == pint2sym->end()) {
					LOGTW_ERROR << " lang/topo has phone " << _words[i] << " which is not in phones.txt.";
					return -1;
				}
				phones_in_topo_int_hash.push_back(iw);
				phones_in_topo_hash.push_back(ci->second);
			}
		}
	}
	ifs.close();

	std::vector<std::string> phones_that_should_be_in_topo_hash;
	phones_that_should_be_in_topo_hash.insert(phones_that_should_be_in_topo_hash.end(), silence->begin(), silence->end());
	phones_that_should_be_in_topo_hash.insert(phones_that_should_be_in_topo_hash.end(), nonsilence->begin(), nonsilence->end());
	for (int i = 0; i < phones_that_should_be_in_topo_hash.size(); i++)
	{
		if (std::find(phones_in_topo_hash.begin(), phones_in_topo_hash.end(), phones_that_should_be_in_topo_hash[i]) == phones_in_topo_hash.end())
		{
			MAPSS::const_iterator ci = pint2sym->find(phones_that_should_be_in_topo_hash[i]);
			if (ci == pint2sym->end()) {
				LOGTW_ERROR << " lang/topo has phone " << phones_that_should_be_in_topo_hash[i] << " which is not in phones.txt.";
				return -1;
			}
			LOGTW_ERROR << " lang/topo does not cover phone " << phones_that_should_be_in_topo_hash[i] << " (label = " << ci->second << ").";
			return -1;
		}
	}

	for (int i = 0; i < phones_in_topo_int_hash.size(); i++)
	{
		MAPSS::const_iterator ci = pint2sym->find(std::to_string(phones_in_topo_int_hash[i]));
		if (ci == pint2sym->end()) {
			LOGTW_ERROR << " lang/topo has phone " << phones_in_topo_int_hash[i] << " which is not in phones.txt.";
			return -1;
		}
		if (std::find(phones_that_should_be_in_topo_hash.begin(), phones_that_should_be_in_topo_hash.end(), ci->second) == phones_that_should_be_in_topo_hash.end())
		{
			LOGTW_ERROR << " lang/topo covers phone " << ci->second << " (label = " << phones_in_topo_int_hash[i] << " ) which is not a real phone.";
			return -1;
		}
	}

	LOGTW_INFO << "  --> " << fileTopo.filename() << " is OK.";
	return 0;
}


int CheckTxtInt(fs::path file, MAPSS map, int sym_check, std::vector<std::string> * silence, std::vector<std::string> * nonsilence)
{
	LOGTW_INFO << "Checking " << file.string() << "{.txt, .int} ...";
	std::vector<fs::path> sfile{ (file.string() + ".txt"), (file.string() + ".int") };
	int i = 0;
	int idx1 = 1, idx2 = 1, num_lines = 0;
	std::vector<std::string> entry, used_syms;

	for each(fs::path p in sfile) {
		if (CheckFileExistsAndNotEmpty(p, true) < 0) {
			LOGTW_ERROR << " file is empty or does not exist: " << p.string() << ".";
			return -1;
		}
		if (i == 0)
		{ //only check the .txt
			if (CheckUTF8AndWhiteSpace(p, true) < 0) {
				LOGTW_ERROR << " the file is not utf-8 compatible or contains not allowed whitespaces: " << p.string() << ".";
				return -1;
			}
		}
		StringTable table_;
		if (ReadStringTable(p.string(), table_) < 0) {
			LOGTW_ERROR << " fail to open: " << p.string() << ".";
			return -1;
		}
		static const boost::regex re1("(^(shared|not-shared) (split|not-split) )|( nonword$)|( begin$)|( end$)|( internal$)|( singleton$)");
		switch (i) {
		case 0: //txt	
			for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it) 
			{
				std::string s;
				int c = 0;
				for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1) 
				{
					if(c > 0) s.append(" ");
					s.append(*itc1);
					c++;
				}
				if (s != "") {
					entry.push_back(boost::regex_replace(s, re1, ""));
					idx1++;
				}
			}
			idx1--;
			LOGTW_INFO << "  --> " << idx1 << " entry/entries in " << p.filename() << ".";
			break;
		case 1: //int
			for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it)
			{
				std::string s;
				int c = 0;
				for (string_vec::const_iterator itc1(it->begin()), itc1_end(it->end()); itc1 != itc1_end; ++itc1)
				{
					if (c > 0) s.append(" ");
					s.append(*itc1);
					c++;
				}
				if (s != "") {
					std::string ss = boost::regex_replace(s, re1, "");
					std::vector<std::string> col, set;
					strtk::parse(ss, " ", col, strtk::split_options::compress_delimiters);
					//NOTE: entry[] is 0 based but the idx2 starts from 1!
					strtk::parse(entry[idx2-1], " ", set, strtk::split_options::compress_delimiters);
					if (set.size() != col.size()) {
						LOGTW_ERROR << " " << sfile[1].filename() << " doesn't correspond to "
								  << sfile[0].filename() << " (break at line " << idx2 << " ).";
					}
					for (int i = 0; i <= set.size()-1; i++) {
						MAPSS::iterator im = map.find(set[i]);
						if (im == map.end() || im->second != col[i]) {
							LOGTW_ERROR << " " << sfile[1].filename() << " doesn't correspond to "
								<< sfile[0].filename() << " (break at line " << idx2 << ", block " << (i+1) << " ).";
							return -1;
						}
						if (sym_check && std::find(used_syms.begin(), used_syms.end(), set[i]) != used_syms.end()) {
							LOGTW_ERROR << " " << sfile[0].filename() << " doesn't correspond to "
								<< sfile[1].filename() << " (break at line " << idx2 << ", block " << (i + 1) << " ).";
							return -1;
						}
						used_syms.push_back(set[i]);
					}
					idx2++;
				}
			}
			idx2--;
			break;
		}
		i++;
	}

	if (idx1 != idx2) {
		LOGTW_ERROR << " cat.int doesn't correspond to " << sfile[0].filename() << " (break at line " << (idx2 + 1) << ").";
		return -1;
	}
	LOGTW_INFO << "  --> cat.int corresponds to cat.txt.";

	if (sym_check) {
		for each(std::string s in *silence)
		{
			if (std::find(used_syms.begin(), used_syms.end(), s) == used_syms.end()) 
			{
				LOGTW_ERROR << " " << sfile[0].filename() << " and " << sfile[1].filename() << " do not contain all silence phones.";
				return -1;
			}
		}
		for each(std::string s in *nonsilence)
		{
			if (std::find(used_syms.begin(), used_syms.end(), s) == used_syms.end())
			{
				LOGTW_ERROR << " " << sfile[0].filename() << " and " << sfile[1].filename() << " do not contain all non-silence phones.";
				return -1;
			}
		}
	}
	
	LOGTW_INFO << "  --> " << file.string() << "{.txt, .int} are OK!";
	return 0;
}


int CheckTxtIntCsl(fs::path file, MAPSS map)
{
	LOGTW_INFO << "Checking " << file.string() << "{.txt, .int, .csl} ...";
	std::vector<fs::path> sfile{ (file.string() + ".txt"), (file.string() + ".int"), (file.string() + ".csl") };
	int i = 0;
	int idx1 = 1, idx2 = 1, num_lines = 0;
	std::vector<std::string> entry;

	for each(fs::path p in sfile) {
		if (CheckFileExistsAndNotEmpty(p, true) < 0) {
			LOGTW_ERROR << " file is empty or does not exist: " << p.string() << ".";
			return -1;
		}
		if (i == 0) 
		{ //only check the .txt
			if (CheckUTF8AndWhiteSpace(p, true) < 0) {
				LOGTW_ERROR << " the file is not utf-8 compatible or contains not allowed whitespaces: " << p.string() << ".";
				return -1;
			}
		}
		StringTable table_;
		switch (i) {
		case 0: //txt			
			if (ReadStringTable(p.string(), table_) < 0) {
				LOGTW_ERROR << " fail to open: " << p.string() << ".";
				return -1;
			}
			for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it) {
				if ((*it).size() != 1) {
					LOGTW_ERROR << " expect 1 column in " << p.filename() << " (break at line " << idx1 << " ).";
					return -1;
				}
				entry.push_back((*it)[0]);
				idx1++;
			}
			idx1--;
			LOGTW_INFO << "  --> " << idx1 << " entry/entries in " << p.filename() << ".";
			break;
		case 1: //int
			if (ReadStringTable(p.string(), table_) < 0) {
				LOGTW_ERROR << " fail to open: " << p.string() << ".";
				return -1;
			}
			for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it) {
				if ((*it).size() != 1) {
					LOGTW_ERROR << " expect 1 column in " << p.filename() << " (break at line " << idx2 << " ).";
					return -1;
				}
				//NOTE: entry[] is 0 based but the idx2 starts from 1!
				MAPSS::iterator im = map.find(entry[idx2 - 1]);
				if (im == map.end() || im->second != (*it)[0]) {
					LOGTW_ERROR << " " << sfile[1].filename() << " doesn't correspond to "
											   << sfile[0].filename() << " (break at line " << idx2 << " ).";
					return -1;
				}
				idx2++;
			}
			idx2--;
			if (idx1 != idx2) {
				LOGTW_ERROR << " " << sfile[1].filename() << " doesn't correspond to "
								   << sfile[0].filename() << " (break at line " << (idx2+1) << " ).";
				return -1;
			}
			LOGTW_INFO << sfile[1].filename() << " corresponds to " << sfile[0].filename() << ".";
			break;
		case 2: //csl
			if (ReadStringTable(p.string(), table_, ":") < 0) {
				LOGTW_ERROR << " fail to open: " << p.string() << ".";
				return -1;
			}
			for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it) {
				num_lines++;
				if ((*it).size() != idx1) {
					LOGTW_ERROR << " expect " << idx1 << " block/blocks in " << p.filename() << " (break at line " << num_lines << ").";
					return -1;
				}
				for (int i = 1; i <= idx1; i++) {
					//NOTE: entry[] is 0 based but the idx1 starts from 1!
					MAPSS::iterator im = map.find(entry[i - 1]);
					if (im == map.end() || im->second != (*it)[i-1]) 
					{
						//DEBUG:
						//LOGTW_INFO << "###### " << im->second << " <> " << (*it)[i-1] << "\n";

						LOGTW_ERROR << " " << sfile[2].filename() << " doesn't correspond to "
								  << sfile[0].filename() << " (break at line " << num_lines << ", block " << i << " ).";
						return -1;
					}
				}
			}
			break;
		}
		i++;
	}

	if (idx1 != 0) {
		// nonempty.txt, .int files
		if (num_lines != 1) {
			LOGTW_ERROR << " expect 1 line in " << sfile[2].filename() << ".";
			return -1;
		}
	}
	else {
		if (num_lines != 1 && num_lines != 0) {
			LOGTW_ERROR << " expect 0 or 1 line in " << sfile[2].filename() << ", since empty .txt,int.";
			return -1;
		}
	}
	LOGTW_INFO << "  --> " << sfile[2].filename() << " corresponds to " << sfile[0].filename() << ".";
	LOGTW_INFO << "  --> " << file.string() << "{.txt, .int, .csl} are OK!";

	return 0;
}


int CheckWordsPhones(fs::path path_txt, MAPSS * xsymtab, MAPSS * xint2sym, int requiredcols)
{
	LOGTW_INFO << "Checking " << path_txt.string() << "...";
	if (CheckFileExistsAndNotEmpty(path_txt, true) < 0) {
		LOGTW_ERROR << " file is empty or does not exist: " << path_txt.string() << ".";
		return -1;
	}
	if (CheckUTF8AndWhiteSpace(path_txt, true) < 0) {
		LOGTW_ERROR << " the file is not utf-8 compatible or contains not allowed whitespaces: " << path_txt.string() << ".";
		return -1;
	}
	StringTable table_;
	std::vector<std::string> _ids; //for duplicates checking
	if (ReadStringTable(path_txt.string(), table_) < 0) {
		LOGTW_ERROR << " fail to open: " << path_txt.string() << ".";
		return -1;
	}
	int idx = 1;
	for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it)
	{
		if ((*it).size() != requiredcols) {
			LOGTW_ERROR << " expect " << requiredcols << " columns in " << path_txt.string() << " (break at line " << idx << " ).";
			return -1;
		}
		xsymtab->emplace((*it)[0], (*it)[1]);
		xint2sym->emplace((*it)[1], (*it)[0]);

		//save for duplicate check of IDs (*it)[1]
		_ids.push_back((*it)[1]);

		idx++;
	}
	//check duplicates
	_ids.erase(std::unique(_ids.begin(), _ids.end()), _ids.end()); //delete duplicates
	if (_ids.size() != xsymtab->size()) {
		LOGTW_ERROR << " duplicate IDs in " << path_txt.string() << ".";
		return -1;
	}

	LOGTW_INFO << "  --> " << path_txt.filename() << " is OK.";

	return 0;
}


int CheckDisjoint(fs::path langdir, bool skip_disambig_check, 
	std::vector<std::string> * sum, std::vector<std::string> * silence, std::vector<std::string> * nonsilence, 
	std::vector<std::string> * disambig)
{
	LOGTW_INFO << "Checking disjoint: silence.txt, nonsilence.txt, disambig.txt ...";
	if (CheckFileExistsAndNotEmpty(langdir / "phones/silence.txt", true) < 0) {
		LOGTW_ERROR << " file is empty or does not exist: " << (langdir / "phones/silence.txt").string() << ".";
		return -1;
	}
	if (CheckFileExistsAndNotEmpty(langdir / "phones/nonsilence.txt", true) < 0) {
		LOGTW_ERROR << " file is empty or does not exist: " << (langdir / "phones/nonsilence.txt").string() << ".";
		return -1;
	}
	if (!skip_disambig_check && CheckFileExistsAndNotEmpty(langdir / "phones/disambig.txt", true) < 0) {
		LOGTW_ERROR << " file is empty or does not exist: " << (langdir / "phones/disambig.txt").string() << ".";
		return -1;
	}
	StringTable table_silence, table_nonsilence, table_disambig;
	if (ReadStringTable((langdir / "phones/silence.txt").string(), table_silence) < 0) {
		LOGTW_ERROR << " fail to open: " << (langdir / "phones/silence.txt").string() << ".";
		return -1;
	}
	if (ReadStringTable((langdir / "phones/nonsilence.txt").string(), table_nonsilence) < 0) {
		LOGTW_ERROR << " fail to open: " << (langdir / "phones/nonsilence.txt").string() << ".";
		return -1;
	}
	if (ReadStringTable((langdir / "phones/disambig.txt").string(), table_disambig) < 0) {
		LOGTW_ERROR << " fail to open: " << (langdir / "phones/disambig.txt").string() << ".";
		return -1;
	}

	if (CheckSND(table_silence, silence, "silence.txt") < 0) return -1;
	if (CheckSND(table_nonsilence, nonsilence, "nonsilence.txt") < 0) return -1;
	if (CheckSND(table_disambig, disambig, "disambig.txt") < 0) return -1;

	sum->insert(sum->end(), silence->begin(), silence->end());
	sum->insert(sum->end(), nonsilence->begin(), nonsilence->end());
	sum->insert(sum->end(), disambig->begin(), disambig->end());

	std::vector<std::string> vI_SN, vI_SD, vI_DN;
	std::sort(silence->begin(), silence->end());
	std::sort(nonsilence->begin(), nonsilence->end());
	std::sort(disambig->begin(), disambig->end());
	std::set_intersection(silence->begin(), silence->end(), nonsilence->begin(), nonsilence->end(), std::inserter(vI_SN, vI_SN.end()));
	std::set_intersection(silence->begin(), silence->end(), disambig->begin(), disambig->end(), std::inserter(vI_SD, vI_SD.end()));
	std::set_intersection(disambig->begin(), disambig->end(), nonsilence->begin(), nonsilence->end(), std::inserter(vI_DN, vI_DN.end()));
	if (vI_SN.size() > 0) {
		LOGTW_ERROR << " silence.txt and nonsilence.txt have intersection -- ";
		std::stringstream ss;
		for each(std::string s in vI_SN)
			ss << " " << s;
		LOGTW_INFO << ss.str();
	}
	else {
		LOGTW_INFO << "  --> silence.txt and nonsilence.txt are disjoint.";
	}
	if (vI_SD.size() > 0) {
		LOGTW_ERROR << " silence.txt and disambig.txt have intersection -- ";
		std::stringstream ss;
		for each(std::string s in vI_SD)
			ss << " " << s;
		LOGTW_INFO << ss.str();
	}
	else {
		LOGTW_INFO << "  --> silence.txt and disambig.txt are disjoint.";
	}
	if (vI_DN.size() > 0) {
		LOGTW_ERROR << " disambig.txt and nonsilence.txt have intersection -- ";
		std::stringstream ss;
		for each(std::string s in vI_DN)
			ss << " " << s;
		LOGTW_INFO << ss.str();
	}
	else {
		LOGTW_INFO << "  --> disambig.txt and nonsilence.txt are disjoint.";
	}

	if (vI_SN.size()==0 && vI_SD.size()==0 && vI_DN.size()==0)
		LOGTW_INFO << "  --> disjoint property is OK.";
	return 0;
}


int CheckSND(StringTable table_, std::vector<std::string>* v_, std::string name)
{
	int idx = 1;
	for (StringTable::const_iterator it(table_.begin()), it_end(table_.end()); it != it_end; ++it)
	{
		if (std::find(v_->begin(), v_->end(), (*it)[0]) == v_->end())
			v_->push_back((*it)[0]);
		else {
			LOGTW_ERROR << " phone " << (*it)[0] << " duplicates in " << name << " (line " << idx << " )";
			return -1;
		}
		idx++;
	}
	return 0;
}


int CheckSummation(std::vector<std::string> * sum, MAPSS * psymtab)
{
	LOGTW_INFO << "Checking summation: silence.txt, nonsilence.txt, disambig.txt ...";
	sum->push_back("<eps>");
	std::vector<std::string> itset;
	intersect(sum, psymtab, & itset);

	if (itset.size() < sum->size()) {
		LOGTW_ERROR << " phones in silence.txt, nonsilence.txt, disambig.txt but not in phones.txt -- ";
		std::stringstream ss;
		for each(std::string s in  *sum) {
			if (std::find(itset.begin(), itset.end(), s) == itset.end()) {
				ss << " " << s;
			}
		}
		LOGTW_INFO << ss.str();
		return -1;
	}

	if (itset.size() < psymtab->size()) {
		LOGTW_ERROR << " phones in phones.txt but not in silence.txt, nonsilence.txt, disambig.txt -- ";
		std::stringstream ss;
		for (const auto& pair : *psymtab) {
			if (std::find(itset.begin(), itset.end(), pair.first) == itset.end()) {
				ss << " " << pair.first;
			}
		}
		LOGTW_INFO << ss.str();
		return -1;
	}

	if (itset.size() == sum->size() && itset.size() == psymtab->size()) {
		LOGTW_INFO << "  --> summation property is OK.";
	}

	return 0;
}


void intersect(
	std::vector<std::string> * sum,			// input
	MAPSS * psymtab,						// input (the keys used)
	std::vector<std::string> * itset)		// output the intersection
{
	for each(std::string s in *sum)
	{
		if (psymtab->find(s) != psymtab->end() &&
			std::find(itset->begin(), itset->end(), s) == itset->end())
			itset->push_back(s);
	}
}


