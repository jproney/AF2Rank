# AF2Rank

Code for the paper "State-of-the-Art Estimation of Protein
Model Accuracy using AlphaFold" (https://www.biorxiv.org/content/10.1101/2022.03.11.484043v3). Experiments were run using the latest AlphaFold github commit as of 5/16/2022 (https://github.com/deepmind/alphafold on commit`b85ffe10799ca08cc62146f1dabb4e4ee6c0a580`).

The script `test_templates.py` was used to run the evaluations in the paper for both the Rosetta decoy set and CASP. At a high level, this script takes a series of decoy structures, and uses AlphaFold to rank them by predicted accuracy. Its command line arguments and behavior are as follows:

* `name` -- the name of the directory containing the results of the evaulation (directory will be created by the script if it doesn't exist).
* `target_list` -- a list of names corresponding to the protein targets to evaluate. Each name corresponds to a target in the decoy dataset. Example usage: `--target_list 1agy 1t2p 1mjc`
* `targets_file` -- the path to a file containing a list of targets to be evaluated. Each line contains one target name.
* `output_dir` -- the prefix of the results directory. The full results path is `output_dir + name`
* `decoy_dir` -- the directory containing the decoy dataset to be evaluated. The structure of this directory will be described shortly.
* `af2_dir` -- the directory containing the AlphFold2 code and weights, as cloned from https://github.com/deepmind/alphafold. To set up the AlphaFold directory, clone the AlphaFold repository and run `./scripts/download_alphafold_params.sh`.
* `tm_exec` -- path to a TMScore executable compiled from the code at https://zhanggroup.org/TM-score/.
* `model_num` -- The AlphaFold model weights to use. Only models 1 and 2 work, since the others don't use templates (defaults to model 1).
* `recycles` -- the number of recycles to run AlphaFold with (desults to 1). The total number of iterations through the model is `args.recycles + 1`.
* `seed` -- the random seed to use (defaults to 0)
* `deterministic` -- run the model in a deterministic way (no sequence masking, etc).
* `use_native` -- add the native structure as a decoy and score it like the others.
* `verbose` -- print extra information.
* `seq_relacement` -- the character to use for the sequnece of the decoy structures. Defaults to "", which uses the target sequence.
* `mask_sidechains` -- mask out all of the sidechain atoms aside from CB.
* `mask_sidechains_add_cb` -- mask out all of the sidechain atoms aside from CB, and add CB atoms to the glycines so that all residues have the same atomic structure (use with the gap sequence so that we don't leak information about which residues are glycines).

## Example Usage:

```
python main.py [name] --targets_file [list of targets] --seed 1 --recycles 1 --decoy_dir [decoy directory] --seq_replacement - --mask_sidechains_add_cb --use_native
```

This will run the script with the gap token for the template sequence and the side chains masked, which was the configuration used for the results in the paper.

# Directory Structure

## The decoy dataset directory specified by `decoy_dir` should have the following structure:
* `decoy_pdbs` -- directory containing the decoy structures. Should have a subdirectory corresponding to each target in the dataset. For example, the rosetta decoy set would have subdirectories like `1a32` and `1tud`. Each subdirectory contains a set of `.pdb` files, each of which is a decoy corresponding to the current target. The name of each decoy pdb file is aribtrary unique identifier.
* `native_pdbs` -- directory containing native structures for each target in the dataset. The name of each pdb file should correspond to the name of a subdirectory in the `decoy_pdbs` directory. For example, the rosetta decoy set would have native files like `1a32.pdb` and `1tud.pdb`.
* `decoy_list.txt` -- a file with a list of every decoy in the dataset. Each line has the form `target decoy_id.pdb`. For example, a line from the Rosetta dataset is `1a32 empty_tag_10067_0001_0001.pdb`, which corresponded to a decoy for the protein `1a32`. The corresponding decoy pdb file is `decoy_pdbs/1a32/empty_tag_10067_0001_0001.pdb`
* `gdt_ts.txt` -- a file with a list of GDT_TS scores for each decoy structure. An example line is `1a32 empty_tag_10067_0001_0001.pdb gdt_ref 0.865`. The last field is the actual GDT_TS score, and the second to last field simply indicates that the metric being reported is GDT_TS. If this file is not included, the input GDT_TS values will be reported as `-1` in the output files.
* `tmscore.txt` -- same as `gdt_ts.txt`, except containing TM Scores. Example line: `1a32 empty_tag_10067_0001_0001.pdb tm_ref 0.8339`. If this file is not included, the input TM Score values will be reported as `-1` in the output files.
* `rosettascore.txt` -- same as `gdt_ts.txt`, except containing Rosetta energies. Example line: `1a32 empty_tag_10067_0001_0001.pdb rosetta_ref -179.614`. If this file is not included, the input Roseeta energies will be reported as `-1` in the output files.
* `rmsd.txt` -- same sat `gdt_ts.txt`, except containing aglined RMSDs. Example Line: `1a32 empty_tag_10067_0001_0001.pdb rms_ref 2.105`. If this file is not included, the input RMSD values will be reported as `-1` in the output files.
* `danscore.txt` -- same sa `gdt_ts.txt`, except DeepAccNet scores each decoy. Example Line: `1a32 empty_tag_10067_0001_0001.pdb danscore_ref 0.807`. If this file is not included, the input DeepAccNet values will be reported as `-1` in the output files.

## The output directory specified by `output_dir + name` will have the following structure:
* `pdbs` -- directory containing all of the output PDBs from AlphaFold. These files will have the form `[target]_[decoy_id].pdb`. Example: `1a32_empty_tag_10067_0001_0001.pdb`
* `results` -- directory containing result csv files for each target (e.g. `results_ia32.csv`). The results for each target are written seperately to prevent race conditions if the targets are processed in parallel. Each results file has the following fields:
  - `target` -- The unique identifier of the target (PDB ID or CASP14 Target ID in the example datasets)
  - `decoy_id` -- The identifier of the decoy structure. "none" means no decoy was used, and "native" means the native structure was used as a decoy
  - `decoy_path` -- The path to the PDB file of the decoy structure.
  - `rmsd` -- The RMSD of the decoy to the native structure
  - `rosettascore` -- The Rosetta energy of the decoy structure
  - `gdt_ts` -- The GDT_TS Score of the decoy to the native structure
  - `tmscore` -- the TM Score of the decoy to the native structure
  - `danscore` -- The DeepAccNet score of the decoy structure
  - `output_path` -- The path to the PDB file of the AlphaFold output structure
  - `rmsd_out` -- The RMSD for the AlphaFold's output structure to the native structure
  - `tm_diff` -- The TM Score between the decoy structure and AlphaFold's output structure
  - `tm_out` -- The TM Score of AlphaFold's output structure to the native structure
  - `plddt` -- The predicted LDDT Score from AlphaFold
  - `ptm` -- The predicted TM Score from AlphaFold
* `finished_targets.txt` -- directory containing a list of all targets that have been completed. Used for resuming jobs that are partially finished.
 
# Data Availability

The raw data from the analyses in the paper can be found here: https://drive.google.com/drive/folders/1Q0aCR_lk4R67XlX9IHl6Jk0-dUI19rhA?usp=sharing

This folder contains the following files with ranking results from the Rosetta decoy dataset:

* `rosetta_gapseq.csv` -- Results with the decoy sequence set to a sequence of all gap tokens and sidechains masked
* `rosetta_targetseq.csv` -- Results with the decoy sequence set to the target sequence and sidechains masked

These files have the following subset of the fields described above:
* `target`, `decoy_id`, `rmsd`, `rosettascore`, `gdt_ts`, `tmscore`, `danscore`, `rmsd_out`, `tm_diff`, `tm_out`, `plddt`, `ptm`


In addition, the following files are included from the CASP14 evalutation:

* `casp_gapseq.csv` -- Results with the decoy sequence set to a sequence of all gap tokens and sidechains masked

The fields in these files are as follows:

* `target`, `decoy_id`, `gdt_ts`, `rmsd_out`, `tm_diff`, `gdt_ts_out`, `plddt`, `ptm`

Where all fields are as described above, and `gdt_ts_out` is the GDT_TS of the AlphaFold output structure to the native structure. Note that, for the CASP data, we were unable to access native structures for targets T1085 and T1086, so output accuracies are unavailable for these targets. In general, numeric values are set to -1 when they are not applicable (for instance, the input TM Score for a line representing AlphaFold's behavior with no template input).  

The rosetta decoy set can be found here:
https://files.ipd.uw.edu/pub/decoyset/decoys.zip

Some extra `.txt` files have been added to this dataset. The full set of `.txt` files can be found here: https://drive.google.com/drive/folders/1ew0Y8N55U--2m9fIWm9gJTuKC1LzN9K2?usp=sharing

# Notebook

To run AF2Rank in Google Colab, take a look at this notebook: 

https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/af/examples/AF2Rank.ipynb#scrollTo=UCUZxJdbBjZt

# Update to previous version (5/23/2022)

The previous version of this repository contained a slightly different set of data obtained from setting the decoy sequence to either the target sequence and a sequence of all alanines. This earlier version of the results contained a bug which caused the target sequence to be incorrectly encoded. Specifically, the old code used the amino acid encoding specified by `residue_constants.restypes`, while it should have used the encoding given by `residue_constants.HHBLITS_AA_TO_ID`. This bug caused significant changes to the results using the target sequence, which are discussed in the paper. The old code and data with the erroneous sequence encoding can be found here: https://github.com/jproney/AF2Rank/tree/d7c9ec1fda03604b95f05132a9c2b4b2739a77a5. These results are described in an earlier version of the preprint from before the error was corrected: https://www.biorxiv.org/content/10.1101/2022.03.11.484043v2 
