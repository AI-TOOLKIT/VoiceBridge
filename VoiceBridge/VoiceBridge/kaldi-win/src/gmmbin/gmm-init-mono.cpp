/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2009-2011  Microsoft Corporation, Apache 2.0
		   See ../../COPYING for clarification regarding multiple authors
*/

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"

#include "kaldi-win/src/kaldi_src.h"

namespace kaldi {
// This function reads a file like:
// 1 2 3
// 4 5
// 6 7 8
// where each line is a list of integer id's of phones (that should have their pdfs shared).
int ReadSharedPhonesList(std::string rxfilename, std::vector<std::vector<int32> > *list_out) {
  list_out->clear();
  Input input(rxfilename);
  std::istream &is = input.Stream();
  std::string line;
  while (std::getline(is, line)) {
    list_out->push_back(std::vector<int32>());
	if (!SplitStringToIntegers(line, " \t\r", true, &(list_out->back()))) {
		KALDI_ERR << "Bad line in shared phones list: " << line << " (reading "
			<< PrintableRxfilename(rxfilename) << ")";
		return -1; //VB
	}
    std::sort(list_out->rbegin()->begin(), list_out->rbegin()->end());
	if (!IsSortedAndUniq(*(list_out->rbegin()))) {
		KALDI_ERR << "Bad line in shared phones list (repeated phone): " << line
			<< " (reading " << PrintableRxfilename(rxfilename) << ")";
		return -1; //VB
	}
  }

  return 0; //VB
}

} // end namespace kaldi

int GmmInitMono(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize monophone GMM.\n"
        "Usage:  gmm-init-mono <topology-in> <dim> <model-out> <tree-out> \n"
        "e.g.: \n"
        " gmm-init-mono topo 39 mono.mdl mono.tree\n";

    bool binary = true;
    std::string train_feats;
    std::string shared_phones_rxfilename;
    BaseFloat perturb_factor = 0.0;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("train-feats", &train_feats,
                "rspecifier for training features [used to set mean and variance]");
    po.Register("shared-phones", &shared_phones_rxfilename,
                "rxfilename containing, on each line, a list of phones whose pdfs should be shared.");
    po.Register("perturb-factor", &perturb_factor,
                "Perturb the means using this fraction of standard deviation.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
		//po.PrintUsage();
		//exit(1);
		KALDI_ERR << "wrong arguments.";
		return -1;
    }

    std::string topo_filename = po.GetArg(1);
    int dim = atoi(po.GetArg(2).c_str());
    KALDI_ASSERT(dim> 0 && dim < 10000);
    std::string model_filename = po.GetArg(3);
    std::string tree_filename = po.GetArg(4);

    Vector<BaseFloat> glob_inv_var(dim);
    glob_inv_var.Set(1.0);
    Vector<BaseFloat> glob_mean(dim);
    glob_mean.Set(1.0);

    if (train_feats != "") {
      double count = 0.0;
      Vector<double> var_stats(dim);
      Vector<double> mean_stats(dim);
      SequentialDoubleMatrixReader feat_reader(train_feats);
      for (; !feat_reader.Done(); feat_reader.Next()) {
        const Matrix<double> &mat = feat_reader.Value();
        for (int32 i = 0; i < mat.NumRows(); i++) {
          count += 1.0;
          var_stats.AddVec2(1.0, mat.Row(i));
          mean_stats.AddVec(1.0, mat.Row(i));
        }
      }
      if (count == 0) { 
		  KALDI_ERR << "No features were seen."; 
		  return -1; //VB
	  }
      var_stats.Scale(1.0/count);
      mean_stats.Scale(1.0/count);
      var_stats.AddVec2(-1.0, mean_stats);
	  if (var_stats.Min() <= 0.0) {
		  KALDI_ERR << "Bad variance.";
		  return -1; //VB
	  }
      var_stats.InvertElements();
      glob_inv_var.CopyFromVec(var_stats);
      glob_mean.CopyFromVec(mean_stats);
    }

    HmmTopology topo;
    bool binary_in;
    Input ki(topo_filename, &binary_in);
    topo.Read(ki.Stream(), binary_in);

    const std::vector<int32> &phones = topo.GetPhones();

    std::vector<int32> phone2num_pdf_classes (1+phones.back());
    for (size_t i = 0; i < phones.size(); i++)
      phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);

    // Now the tree [not really a tree at this point]:
    ContextDependency *ctx_dep = NULL;
    if (shared_phones_rxfilename == "") {  // No sharing of phones: standard approach.
      ctx_dep = MonophoneContextDependency(phones, phone2num_pdf_classes);
    } else {
      std::vector<std::vector<int32> > shared_phones;
      int ret = ReadSharedPhonesList(shared_phones_rxfilename, &shared_phones);
      // ReadSharedPhonesList crashes on error.
	  if(ret<0) return -1; //VB
      ctx_dep = MonophoneContextDependencyShared(shared_phones, phone2num_pdf_classes);
    }

    int32 num_pdfs = ctx_dep->NumPdfs();

    AmDiagGmm am_gmm;
    DiagGmm gmm;
    gmm.Resize(1, dim);
    {  // Initialize the gmm.
      Matrix<BaseFloat> inv_var(1, dim);
      inv_var.Row(0).CopyFromVec(glob_inv_var);
      Matrix<BaseFloat> mu(1, dim);
      mu.Row(0).CopyFromVec(glob_mean);
      Vector<BaseFloat> weights(1);
      weights.Set(1.0);
      gmm.SetInvVarsAndMeans(inv_var, mu);
      gmm.SetWeights(weights);
      gmm.ComputeGconsts();
    }

    for (int i = 0; i < num_pdfs; i++)
      am_gmm.AddPdf(gmm);

    if (perturb_factor != 0.0) {
      for (int i = 0; i < num_pdfs; i++)
        am_gmm.GetPdf(i).Perturb(perturb_factor);
    }

    // Now the transition model:
    TransitionModel trans_model(*ctx_dep, topo);

    {
      Output ko(model_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }

    // Now write the tree.
    ctx_dep->Write(Output(tree_filename, binary).Stream(),
                   binary);

    delete ctx_dep;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

