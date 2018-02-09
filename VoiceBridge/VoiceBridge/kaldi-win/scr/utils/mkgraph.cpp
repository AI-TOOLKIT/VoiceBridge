/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2010-2012 Microsoft Corporation, Apache 2.0
					 2012-2013 Johns Hopkins University (Author: Daniel Povey)

	This function creates a fully expanded decoding graph (HCLG) that represents all the language-model, 
	pronunciation dictionary (lexicon), context-dependency, and HMM structure in our model. 
	The output is a Finite State Transducer that has word-ids on the output, and pdf-ids on the input 
	(these are indexes that resolve to Gaussian Mixture Models).
	See  http://kaldi-asr.org/doc/graph_recipe_test.html
*/

#include "kaldi-win\scr\kaldi_scr.h"
#include "kaldi-win\src\kaldi_src.h"

VOICEBRIDGE_API int MkGraph(fs::path lang_dir, fs::path model_dir, fs::path graph_dir,
	bool remove_oov, //If true, any paths containing the OOV symbol (obtained from oov.int in the lang directory) are removed from the G.fst during compilation.
	double tscale,	 //Scaling factor on transition probabilities.
	double loopscale //see: http://kaldi-asr.org/doc/hmm.html#hmm_scale
	)
{
	fs::path lang(lang_dir);
	fs::path tree(model_dir / "tree" );
	fs::path model(model_dir / "final.mdl");
	fs::path dir(graph_dir);
	fs::path f_HCLG_fst(dir / "HCLG.fst");

	if (CreateDir(graph_dir, true) < 0) return -1;

	// If lang/tmp/LG.fst does not exist or is older than its sources, make it... 
	std::vector<fs::path> required = { lang_dir /"L.fst", lang_dir / "G.fst", lang_dir / "phones.txt", lang_dir /"words.txt", lang_dir /"phones" / "silence.csl", lang_dir / "phones" / "disambig.int", model, tree };
	for (fs::path p : required)	{
		if (!fs::exists(p)) {
			LOGTW_ERROR << p.string() << "expected to exist.";
			return -1;
		}
	}

	if (fs::exists(f_HCLG_fst))
	{// detect when the result already exists, and avoid overwriting it.
		bool must_rebuild = false;
		for (fs::path p : required) {
			if (fs::last_write_time(p) > fs::last_write_time(f_HCLG_fst))
				must_rebuild = true;
		}
		if ( !must_rebuild ) {
			LOGTW_INFO << f_HCLG_fst.string() << " is up to date.";
			return 0;
		}
	}

	int numpdfs, context_width, central_position;
	string_vec options;
	options.push_back("--print-args=false");
	options.push_back((tree).string());
	StrVec2Arg args(options);
	if (TreeInfo(args.argc(), args.argv(), numpdfs, context_width, central_position) < 0) {
		LOGTW_ERROR << "Error when getting context-width and central-position.";
		return -1;
	}

	if (fs::exists(model_dir / "frame_subsampling_factor") && loopscale == 0.1)
	{
		LOGTW_WARNING << "chain models need 'self-loop-scale = 1.0' but have " << loopscale;
	}

	if (CreateDir(lang / "tmp", true) < 0) return -1;

	//lang / "tmp" / "LG.fst" - create decoding graph
	if (CheckFileExistsAndNotEmpty(lang / "tmp" / "LG.fst", false) < 0 ||
		fs::last_write_time(lang / "tmp" / "LG.fst") < fs::last_write_time(lang / "G.fst") ||
		fs::last_write_time(lang / "tmp" / "LG.fst") < fs::last_write_time(lang / "L_disambig.fst")	)
	{
		try {
			fs::path f_temp1(lang / "tmp" / "LGTC.temp");
			fsttablecompose((lang / "L_disambig.fst").string(), (lang / "G.fst").string(), f_temp1.string());

			fs::path f_temp2(lang / "tmp" / "LGDS.temp");
			fstdeterminizestar(f_temp1.string(), f_temp2.string(), true);

			fs::path f_temp3(lang / "tmp" / "LGME.temp");
			fstminimizeencoded(f_temp2.string(), f_temp3.string());

			fs::path f_temp4(lang / "tmp" / "LGPS.temp");
			fstpushspecial(f_temp3.string(), f_temp4.string());

			fs::path f_LG_FST(lang / "tmp" / "LG.fst");
			if (fs::exists(f_LG_FST)) fs::remove(f_LG_FST);
			fstarcsort("ilabel", f_temp4.string(), f_LG_FST.string());

			if (fstisstochastic(f_LG_FST.string()) < 0)
			{
				LOGTW_INFO << "LG is not stochastic.";
			}
		}
		catch (const std::exception&)
		{
			LOGTW_ERROR << "failed to create decoding graph.";
			return -1;
		}
	}
	
	//CLG - Context FST creation 
	int N = context_width;
	int P = central_position;

	fs::path clg(lang / "tmp" / ("CLG_" + std::to_string(N) +"_"+ std::to_string(P) + ".fst"));
	fs::path ilabels(lang / "tmp" / ("ilabels_" + std::to_string(N) + "_" + std::to_string(P)));

	if ( CheckFileExistsAndNotEmpty(clg, false) < 0 || fs::last_write_time(clg) < fs::last_write_time(lang / "tmp" / "LG.fst") ||
		 CheckFileExistsAndNotEmpty(ilabels, false) < 0 || fs::last_write_time(ilabels) < fs::last_write_time(lang / "tmp" / "LG.fst") )
	{
		try {
			if (fs::exists(clg)) fs::remove(clg);
			if (fs::exists(ilabels)) fs::remove(ilabels);
		}
		catch (const std::exception& ex) {
			LOGTW_ERROR << "failed to delete file. " << ex.what();
			return -1;
		}

		string_vec optcc;
		optcc.push_back("--print-args=false");
		optcc.push_back("--context-size="+ std::to_string(N));
		optcc.push_back("--central-position=" + std::to_string(P));
		optcc.push_back("--read-disambig-syms=" + (lang / "phones" / "disambig.int").string());
		optcc.push_back("--write-disambig-syms=" + (lang / "tmp" / ("disambig_ilabels_" + std::to_string(N) + "_" + std::to_string(P) + ".int")).string());
		optcc.push_back(ilabels.string());
		optcc.push_back((lang / "tmp" / "LG.fst").string());
		optcc.push_back((lang / "tmp" / "LGCC.temp").string()); //output
		StrVec2Arg argscc(optcc);

		try	{
			if (fstcomposecontext(argscc.argc(), argscc.argv()) < 0) {
				LOGTW_ERROR << "Context FST creation failed.";
				return -1;
			}
			fstarcsort("ilabel", (lang / "tmp" / "LGCC.temp").string(), clg.string());
			if (fstisstochastic(clg.string()) < 0)
			{
				LOGTW_INFO << "CLG is not stochastic.";
			}
		}
		catch (const std::exception&)
		{
			LOGTW_ERROR << "Context FST creation failed.";
			return -1;
		}
	}

	//dir / "Ha.fst" - H transducer creation
	if (CheckFileExistsAndNotEmpty(dir / "Ha.fst", false) < 0 || fs::last_write_time(dir / "Ha.fst") < fs::last_write_time(model) ||
		fs::last_write_time(dir / "Ha.fst") < fs::last_write_time(ilabels))
	{
		string_vec optmhtd;
		optmhtd.push_back("--print-args=false");
		optmhtd.push_back("--disambig-syms-out=" + (dir / "disambig_tid.int").string());
		optmhtd.push_back("--transition-scale=" + std::to_string(tscale));
		optmhtd.push_back((lang / "tmp" / ("ilabels_" + std::to_string(N) + "_" + std::to_string(P))).string());
		optmhtd.push_back(tree.string());
		optmhtd.push_back(model.string());
		optmhtd.push_back((dir / "Ha.fst").string()); //output
		StrVec2Arg argsmhtd(optmhtd);

		try	{
			if (fs::exists(dir / "Ha.fst")) fs::remove(dir / "Ha.fst");

			if (MakeHTransducer(argsmhtd.argc(), argsmhtd.argv()) < 0) {
				LOGTW_ERROR << "H transducer creation failed.";
				return -1;
			}
		}
		catch (const std::exception&)
		{
			LOGTW_ERROR << "H transducer creation failed.";
			return -1;
		}
	}

	//dir / "HCLGa.fst" - initialize Final HCLG
	if (CheckFileExistsAndNotEmpty(dir / "HCLGa.fst", false) < 0 || fs::last_write_time(dir / "HCLGa.fst") < fs::last_write_time(dir / "Ha.fst") ||
		fs::last_write_time(dir / "HCLGa.fst") < fs::last_write_time(clg))
	{
		if (remove_oov) {
			if (!fs::exists(lang / "oov.int")) {
				LOGTW_WARNING << "remove-oov option is specified but there is no file " << (lang / "oov.int").string();
				remove_oov = false;
			}
			else {
				//modify clg : remove symbols
				fs::path clg_temp(clg.string() + ".temp");
				string_vec optrms;
				optrms.push_back("--print-args=false");
				optrms.push_back("--remove-arcs=true");
				optrms.push_back("--apply-to-output=true");
				optrms.push_back((lang / "oov.int").string());
				optrms.push_back(clg.string());
				optrms.push_back(clg_temp.string()); //output
				StrVec2Arg argsrms(optrms);

				try {
					if (fstrmsymbols(argsrms.argc(), argsrms.argv()) < 0) {
						LOGTW_ERROR << "Symbols replacement failed.";
						return -1;
					}
					//delete clg and rename clg_temp to clg
					fs::remove(clg);
					fs::rename(clg_temp, clg);
				}
				catch (const std::exception&)
				{
					LOGTW_ERROR << "Symbols replacement failed.";
					return -1;
				}
			}
		}

		fs::path f_temp1(dir / "Ha.temp");
		fsttablecompose((dir / "Ha.fst").string(), clg.string(), f_temp1.string());
		fs::path f_temp2(dir / "HaDS.temp");
		fstdeterminizestar(f_temp1.string(), f_temp2.string(), true);

		fs::path f_temp3(dir / "HaRMS.temp");
		string_vec optrms;
		optrms.push_back("--print-args=false");
		optrms.push_back((dir / "disambig_tid.int").string());
		optrms.push_back(f_temp2.string()); //input from fstdeterminizestar	
		optrms.push_back(f_temp3.string()); //output
		StrVec2Arg argsrms(optrms);
		try {
			if (fstrmsymbols(argsrms.argc(), argsrms.argv()) < 0) {
				LOGTW_ERROR << "Symbols replacement failed.";
				return -1;
			}
		}
		catch (const std::exception&)
		{
			LOGTW_ERROR << "Symbols replacement failed.";
			return -1;
		}

		fs::path f_temp4(dir / "HaRMEL.temp");
		string_vec optrmel;
		optrmel.push_back("--print-args=false");
		optrmel.push_back(f_temp3.string());
		optrmel.push_back(f_temp4.string()); //output
		StrVec2Arg argsrmel(optrmel);
		try {
			if (fstrmepslocal(argsrmel.argc(), argsrmel.argv()) < 0) {
				LOGTW_ERROR << "Symbols replacement failed.";
				return -1;
			}			
		}
		catch (const std::exception&)
		{
			LOGTW_ERROR << "Epsilon removal failed.";
			return -1;
		}

		try	{
			if (fs::exists(dir / "HCLGa.fst")) fs::remove(dir / "HCLGa.fst");
		}
		catch (const std::exception&)
		{
			LOGTW_ERROR << "Could not delete " << (dir / "HCLGa.fst").string();
			return -1;
		}

		fstminimizeencoded(f_temp4.string(), (dir / "HCLGa.fst").string());

		if (fstisstochastic((dir / "HCLGa.fst").string()) < 0)
		{
			LOGTW_INFO << "HCLGa is not stochastic.";
		}
	}

	//dir / "HCLG.fst" - create Final HCLG
	if (CheckFileExistsAndNotEmpty(f_HCLG_fst, false) < 0 || fs::last_write_time(f_HCLG_fst) < fs::last_write_time(dir / "HCLGa.fst"))
	{
		//AddSelfLoops
		fs::path f_tempsl(dir / "HCLGa.temp");
		string_vec optasl;
		optasl.push_back("--print-args=false");
		optasl.push_back("--self-loop-scale=" + std::to_string(loopscale));
		optasl.push_back("--reorder=true");
		optasl.push_back(model.string());
		optasl.push_back((dir / "HCLGa.fst").string());
		optasl.push_back(f_tempsl.string()); //output
		StrVec2Arg argsasl(optasl);
		try {
			if (AddSelfLoops(argsasl.argc(), argsasl.argv()) < 0) {
				LOGTW_ERROR << "Adding self-loops failed.";
				return -1;
			}
		}
		catch (const std::exception&)
		{
			LOGTW_ERROR << "Adding self-loops failed.";
			return -1;
		}	

		try	{
			if (fs::exists(f_HCLG_fst)) fs::remove(f_HCLG_fst);
			if (fstconvert(f_tempsl.string(), f_HCLG_fst.string(), "const") < 0) return -1;
		}
		catch (const std::exception&) {
			LOGTW_ERROR << "Failed to convert HCLGa.";
			return -1;
		}
		if (tscale == 1.0 && loopscale == 1.0) {
			// No point doing this test if transition - scale not 1, as it is bound to fail.				
			if (fstisstochastic(f_HCLG_fst.string()) < 0)
			{
				LOGTW_INFO << "Final HCLG is not stochastic.";
			}
		}
	}

	//NOTE: the empty FST has 66 bytes. This check is for whether the final FST is the empty file or is the empty FST.
	FILE *f;
	f = fopen(f_HCLG_fst.string().c_str(), "rb");
	if (f == NULL) {
		LOGTW_ERROR << "Error opening file. " << f_HCLG_fst.string();
		return -1;
	}
	int count = 0;
	//count one more than 66 and stop. If there is more than 66 then it is OK.
	while (fgetc(f) != EOF)	{
		count++;
		if (count == 67) break;
	}
	fclose(f);
	if (count != 67) {
		LOGTW_ERROR << "It looks like the result in " << f_HCLG_fst.string() << " is empty.";
		return -1;
	}

	//remove all temp files and intermediate files 
	std::vector<fs::path> _temps = { (lang / "tmp" / "LGTC.temp"),
									(lang / "tmp" / "LGDS.temp"),
									(lang / "tmp" / "LGME.temp"),
									(lang / "tmp" / "LGPS.temp"),
									(lang / "tmp" / "LGCC.temp"),
									(clg.string() + ".temp"),
									(dir / "Ha.temp"),
									(dir / "HaDS.temp"),
									(dir / "HaRMS.temp"),
									(dir / "HaRMEL.temp"),
									(dir / "HCLGa.temp"),
									(dir / "HCLGa.temp"),
									(dir / "HCLGa.fst"),
									(dir / "Ha.fst") };
	try	{
		for (fs::path p : _temps) if (fs::exists(p)) fs::remove(p);
	}
	catch (const std::exception&) {
		LOGTW_WARNING << "Could not remove temporary files.";
	}
	
	//keep a copy of the lexicon and a list of silence phones with HCLG. This means we can decode without reference to the 'lang' directory.
	try	{
		fs::copy_file(lang / "words.txt", dir / "words.txt");
		if (CreateDir(dir / "phones", true) < 0) return -1;
	}
	catch (const std::exception& ex){
		LOGTW_ERROR << ex.what();
		return -1;
	}
	//the next will not crash if the files do not exist:
	//NOTE "^(word_boundary\\.).*"   =>   any file starting with 'word_boundary.' ( = word_boundary.* )
	CopyAllMatching(lang / "phones", dir / "phones", boost::regex("^(word_boundary\\.).*")); //might be needed for ctm scoring
	CopyAllMatching(lang / "phones", dir / "phones", boost::regex("^(align_lexicon\\.).*")); //might be needed for ctm scoring
	CopyAllMatching(lang / "phones", dir / "phones", boost::regex("^(optional_silence\\.).*")); //might be needed for analyzing alignments

	if(fs::exists(lang / "phones" / "disambig.txt"))
		fs::copy_file(lang / "phones" / "disambig.txt", dir / "phones" / "disambig.txt");
	if (fs::exists(lang / "phones" / "disambig.int"))
		fs::copy_file(lang / "phones" / "disambig.int", dir / "phones" / "disambig.int");
	if (fs::exists(lang / "phones" / "silence.csl"))
		fs::copy_file(lang / "phones" / "silence.csl", dir / "phones" / "silence.csl");
	if (fs::exists(lang / "phones.txt"))
		fs::copy_file(lang / "phones.txt", dir / "phones.txt");

	//get model info
	int nofphones, nofpdfs, noftransitionids, noftransitionstates;
	if(AmInfo(model.string(), nofphones, nofpdfs, noftransitionids, noftransitionstates) < 0) return -1;

	fs::ofstream file_num_pdfs(dir / "num_pdfs", std::ios::binary);
	if (!file_num_pdfs) {
		LOGTW_ERROR << " can't open output file: " << (dir / "num_pdfs").string();
		return -1;
	}
	file_num_pdfs << nofpdfs << "\n";
	file_num_pdfs.flush(); file_num_pdfs.close();

	return 0;
}
