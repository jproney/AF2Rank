# AF2Rank

Code for the paper "State-of-the-Art Estimation of Protein
Model Accuracy using AlphaFold." Experiments were run using code cloned from https://github.com/deepmind/alphafold on commit`1d43aaff941c84dc56311076b58795797e49107b`. More documentation and scripts coming soon.

# Data Availability

The raw data from the analyses in the paper can be found here: https://drive.google.com/drive/folders/1hsLs-Ul1ZpsFWrfpAgeWOJfUjN77h0dO?usp=sharing

This folder contains the following files with ranking results from the Rosetta decoy dataset:

* `rosetta_alanine.csv` -- Results with the decoy sequence set to a sequence of all alanines
* `rosetta_targetseq.csv` -- Results with the decoy sequence set to the target sequence
* `rosetta_sidechains.csv` -- Results with sidechains included in the decoy structures

The fields in these files are as follows:

* `target` -- The PDB ID of the target sequnce
* `decoy_name` -- The identifier of the decoy structure. "none" means no decoy was used, and "native" means the native structure was used as a decoy
* `rmsd_in` -- The RMSD of the decoy to the native structure
* `gdtts_in` -- The GDT_TS Score of the decoy to the native structure
* `tmscore_in` -- the TM Score of the decoy to the native structure
* `plddt` -- The predicted LDDT Score from AlphaFold
* `ptm` -- The predicted TM Score from AlphaFold
* `tmscore_diff` -- The TM Score between the decoy structure and AlphaFold's output structure
* `tmscore_out` -- The TM Score of AlphaFold's output structure to the native structure
* `rmsd_out` -- The RMSD for the AlphaFold's output structure to the native structure
* `rosetta_score` -- The Rosetta energy of the decoy structure
* `dan_score` -- The DeepAccNet score of the decoy structure
* `is_native` -- Boolean indicator of whether the decoy is the native structure.
* `no_template` -- Boolean indicator of whether the prediction was made without a template.

In addition, the following files are included from the CASP14 evalutation:

* `casp_alanine.csv` -- Results with the decoy sequence set to a sequence of all alanines
* `casp_targetseq.csv` -- Results with the decoy sequence set to the target sequence
* `casp_sidechains.csv` -- Results with sidechains included in the decoy structures

The fields in these files are as follows:

* `target` -- The CASP14 target identifier
* `decoy_name` -- The name of the decoy server submission from CASP14.
* `gddts_in` -- The GDT_TS score of the template to the native structure
* `plddt` -- The predicted LDDT Score from AlphaFold
* `ptm` -- The predicted TM Score from AlphaFold
* `tmscore_diff` -- The TM Score between the decoy structure and AlphaFold's output structure
* `gdtts_out` -- The GDT_TS Score of AlphaFold's output structure to the native structure
* `no_template` -- Boolean indicator of whether the prediction was made without a template.

Note that, for the CASP data, we were unable to access native structures for targets T1085 and T1086, so output accuracies are unavailable for these targets. In general, numeric values are set to -1 when they are not applicable (for instance, the input TM Score for a line representing AlphaFold's behavior with no template input).  


