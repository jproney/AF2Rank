import sys

sys.path.append("/n/home01/jroney/alphafold")

import os
import argparse
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import re
import subprocess

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data

from alphafold.data import parsers
from alphafold.data import pipeline

from alphafold.common import protein
from alphafold.common import residue_constants

parser = argparse.ArgumentParser()
parser.add_argument("name", help="name to save everything under")
parser.add_argument("--target_list", nargs='*', help="List of PDB codes to run")
parser.add_argument("--targets_file", default="", help="File with list of PDB codes to run")
parser.add_argument("--recycles", type=int, default=3, help="Number of recycles when predicting")
parser.add_argument("--model_num", type=int, default=1, help="Which AF2 model to use")
parser.add_argument("--seed", type=int, default=0, help="RNG Seed")
parser.add_argument("--ptm", action='store_true', help="Use models with ptm heads")
parser.add_argument("--max_decoys", type=int, default=-1, help="Max number of decoys to run per target (-1 = do them all)")
parser.add_argument("--verbose", action='store_true', help="print extra")
parser.add_argument("--none2x", action='store_true', help="pass the template-free prediction back as a template")
parser.add_argument("--alanine", action='store_true', help="switch all of the template residues to alanine")
parser.add_argument("--nosc", action='store_true', help="keep the template sequence but get rid of the sidechains")
args = parser.parse_args()

# helper functions

def pdb_to_string(pdb_file):
    lines = []
    for line in open(pdb_file,"r"):
      if line[:6] == "HETATM" and line[17:20] == "MSE":
        line = "ATOM  "+line[6:17]+"MET"+line[20:]
      if line[:4] == "ATOM":
        lines.append(line)
    return "".join(lines)

def jnp_rmsd(true, pred):
    def kabsch(P, Q):
      V, S, W = jnp.linalg.svd(P.T @ Q, full_matrices=False)
      flip = jax.nn.sigmoid(-10 * jnp.linalg.det(V) * jnp.linalg.det(W))
      S = flip * S.at[-1].set(-S[-1]) + (1-flip) * S
      V = flip * V.at[:,-1].set(-V[:,-1]) + (1-flip) * V
      return V@W
    p = true - true.mean(0,keepdims=True)
    q = pred - pred.mean(0,keepdims=True)
    p = p @ kabsch(p,q)
    loss = jnp.sqrt(jnp.square(p-q).sum(-1).mean() + 1e-8)
    return float(loss)


def make_model_runner(name, cfg_name, recycles):
    #print(cfg_name)
    cfg = config.model_config(cfg_name)      

    cfg.data.common.num_recycle = recycles
    cfg.model.num_recycle = recycles
    cfg.data.eval.num_ensemble = 1
    params = data.get_model_haiku_params(name,'/n/home01/jroney/alphafold/data/')

    return model.RunModel(cfg, params)

def make_processed_feature_dict(runner, sequence, name="test", templates=None, seed=0):
    feature_dict = {}
    feature_dict.update(pipeline.make_sequence_features(sequence, name, len(sequence)))

    msa = [[sequence]]
    deletion_matrices = [[[0]*len(sequence)]]

    feature_dict.update(pipeline.make_msa_features(msa, deletion_matrices=deletion_matrices))

    if templates is not None:
        #print("Updating templates!")
        feature_dict.update(templates)

    processed_feature_dict = runner.process_features(feature_dict, random_seed=seed)
    return processed_feature_dict

def parse_results(prediction_result, processed_feature_dict):
    b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']  
    dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
    dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
    contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,dist_bins < 8].sum(-1)

    out = {"unrelaxed_protein": protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors),
          "plddt": prediction_result['plddt'],
          "pLDDT": prediction_result['plddt'].mean(),
          "dists": dist_mtx,
          "adj": contact_mtx}

    if "ptm" in prediction_result:
      out.update({"pae": prediction_result['predicted_aligned_error'],
                  "pTMscore": prediction_result['ptm']})
    return out


base_dir = "/n/home01/jroney/template_injection/"
tm_re = re.compile(r'TM-score[\s]*=[\s]*(\d.\d+)')
ref_len_re = re.compile(r'Length=[\s]*(\d+)[\s]*\(by which all scores are normalized\)')
common_re = re.compile(r'Number of residues in common=[\s]*(\d+)')
super_re = re.compile(r'\(":" denotes the residue pairs of distance < 5\.0 Angstrom\)\\n([A-Z\-]+)\\n[" ", :]+\\n([A-Z\-]+)\\n')

def make_tmscore(pdb_path, pdb_native):
    cmd = " ".join(["/n/home01/jroney/tmscore/TMscore", pdb_path, pdb_native])
    tmscore_output = str(subprocess.check_output(["/n/home01/jroney/tmscore/TMscore", pdb_path, pdb_native]))
    try:
      tm_out = float(tm_re.search(tmscore_output).group(1))
      reflen = int(ref_len_re.search(tmscore_output).group(1))
      common = int(common_re.search(tmscore_output).group(1))
      
      seq1 = super_re.search(tmscore_output).group(1)
      seq2 = super_re.search(tmscore_output).group(1)
    except Exception as e:
      print("Failed on: " + cmd)
      raise e

    assert reflen == common, cmd
    assert seq1 == seq2, cmd
    assert len(seq1) == reflen, cmd

    return tm_out

def write_results(target_name, targets_hash, result, prot_ref, rms_in, gdt_in, rscore_in, tm_in, ema_in, pdb_in, pdb_out, pdb_native, args, prot_decoy=None):
    plddt = float(result['pLDDT'])
    if args.ptm:
      ptm = float(result["pTMscore"])
    rms_out = jnp_rmsd(prot_ref.atom_positions[:,1,:], result['unrelaxed_protein'].atom_positions[:,1,:])

    if prot_decoy is not None:
      rms_change = jnp_rmsd(prot_decoy.atom_positions[:,1,:], result['unrelaxed_protein'].atom_positions[:,1,:])
    else:
      rms_change = -1

    pdb_lines = protein.to_pdb(result["unrelaxed_protein"])
    pdb_path = base_dir + "outputs/" + args.name + "/pdbs/" + pdb_out
    with open(pdb_path, 'w') as f:
      f.write(pdb_lines)


    tm_out = make_tmscore(pdb_path, pdb_native)
    if pdb_in != "_":
      tm_diff = make_tmscore(pdb_in, pdb_path)
    else: tm_diff = -1

    with open(base_dir + "outputs/" + args.name + "/results/results_{}.csv".format(targets_hash), "a") as f:
      # name of target, input pdb, output pdb, input rmsd, input gdt, input rosetta, rms_out, plddt, ptm
      if args.ptm:
        f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(target_name, pdb_in, pdb_path, rms_in, gdt_in, rscore_in, rms_out, rms_change, plddt, ptm, tm_in, tm_diff, tm_out, ema_in))
      else:
        f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(target_name, pdb_in, pdb_path, rms_in, gdt_in, rscore_in, rms_out, rms_change, plddt, tm_in, tm_diff, tm_out, ema_in))

      if args.verbose:
        if args.ptm:
          print("taget={}, decoy={}, out={}, rms_in={}, gdt_in={}, rscore_in={}, rms_out={}, rms_change={}, plddt={}, ptm={}, tm_in={}, tm_diff={}, tm_out={}, ema_in={}".format(target_name, pdb_in, pdb_path, rms_in, gdt_in, rscore_in, rms_out, rms_change, plddt, ptm, tm_in, tm_diff, tm_out, ema_in))
        else:
          print("taget={}, decoy={}, out={}, rms_in={}, gdt_in={}, rscore_in={}, rms_out={}, rms_change={}, plddt={}, tm_in={}, tm_diff={}, tm_out={}, ema_in={}".format(target_name, pdb_in, pdb_path, rms_in, gdt_in, rscore_in, rms_out, rms_change, plddt, tm_in, tm_diff, tm_out, ema_in))

def extend(a,b,c, L,A,D):
  '''
  input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
  output: 4th coord
  '''
  N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
  bc = N(b-c)
  n = N(np.cross(b-a, bc))
  m = [bc,np.cross(n,bc),n]
  d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
  return c + sum([m*d for m,d in zip(m,d)])

# parse all of the decoy files
os.makedirs(base_dir + "outputs/" + args.name, exist_ok=True)
os.makedirs(base_dir + "outputs/" + args.name + "/pdbs", exist_ok=True)
os.makedirs(base_dir + "outputs/" + args.name + "/results", exist_ok=True)

rms_file = open(base_dir + "decoy_set/rms_ref.txt", 'r').read().split("\n")[:-1]
gdt_file = open(base_dir + "decoy_set/gdtmm_ref.txt", 'r').read().split("\n")[:-1]
rosetta_file = open(base_dir + "decoy_set/score.txt", 'r').read().split("\n")[:-1]
tm_file = open(base_dir + "decoy_set/tm_ref.txt", 'r').read().split("\n")[:-1]
ema_file = open(base_dir + "decoy_set/ema_sorted_ref.txt", 'r').read().split("\n")[:-1]

if len(args.targets_file) > 0:
  natives_list = open(args.targets_file, 'r').read().split("\n")[:-1]
else:
  natives_list = args.target_list

if len(natives_list) > 1:
  targets_hash = hashlib.md5("".join(natives_list).encode('utf-8')).hexdigest()
else:
  targets_hash = natives_list[0]

if os.path.exists(base_dir + "outputs/" + args.name + "/results/results_{}.csv".format(targets_hash)):
  finished_decoys = set([x.split(",")[2].split("/")[-1] for x in open(base_dir + "outputs/" + args.name + "/results/results_{}.csv".format(targets_hash), "r").readlines()])
else:
  finished_decoys = []
  
if os.path.exists(base_dir + "outputs/" + args.name + "/finished_targets.txt"):
  finished_targets = set(open(base_dir + "outputs/" + args.name + "/finished_targets.txt", 'r').read().split("\n")[:-1])
else:
  finished_targets = []

decoy_dict = {n : [] for n in natives_list if n not in finished_targets} # key = pdb code, value = list of decoys and their data

for rms, gdt, rscore, tmscore, ema in zip(rms_file, gdt_file, rosetta_file, tm_file, ema_file):
    rms_fields = rms.split()
    gdt_fields = gdt.split()
    tm_fields = tmscore.split()
    rscore_fields = rscore.split()
    ema_fields = ema.split()

    # make sure proteins and names are the same
    assert rms_fields[0] == gdt_fields[0]
    assert rms_fields[0] == rscore_fields[0]
    assert rms_fields[0] == tm_fields[0]
    assert rms_fields[0] == ema_fields[0]
    assert rms_fields[1] == gdt_fields[1]
    assert rms_fields[1] == rscore_fields[1]
    assert rms_fields[1] == tm_fields[1]
    assert rms_fields[1] == ema_fields[1]

    if rms_fields[0] in decoy_dict and rms_fields[0] + "_" + rms_fields[1] not in finished_decoys and (len(decoy_dict[rms_fields[0]]) < args.max_decoys or args.max_decoys < 0):
      decoy_dict[rms_fields[0]].append((rms_fields[1], rms_fields[-1], gdt_fields[-1], rscore_fields[-1], tm_fields[-1], ema_fields[-1]))

if args.verbose:
  print(finished_decoys)
  print(decoy_dict)

ptm_suffix = ("_ptm" if args.ptm else "")
model_name = "model_{}".format(args.model_num) + ptm_suffix
results_key = model_name + "_seed_{}".format(args.seed)
for n in natives_list:

    pdb_native = base_dir + "decoy_set/native_pdbs/" + n + "_renum.pdb"
    prot_native = protein.from_pdb_string(pdb_to_string(pdb_native))
    seq_native = "".join([residue_constants.restypes[x] for x in prot_native.aatype])

    if n + "_none.pdb" not in finished_decoys:

      # run the model with no templates

      runner = make_model_runner(model_name, "model_5" + ptm_suffix, args.recycles) # model 5 = no templates
      features = make_processed_feature_dict(runner, seq_native, name=n + "_none", seed=args.seed)
      result = parse_results(runner.predict(features), features)

      write_results(n, targets_hash, result, prot_native, -1, -1, -1, -1, -1, "_", n + "_none.pdb", pdb_native, args)


    runner = make_model_runner(model_name, "model_1" + ptm_suffix, args.recycles) # model 1 = yes templates

    # run the model with the native structure passed as a template
    if n + "_native.pdb" not in finished_decoys:
      if args.alanine:
        pos = np.zeros([1,len(seq_native), 37, 3])
        atom_mask = np.zeros([1, len(seq_native), 37])
        gly_idx = [i for i,a in enumerate(seq_native) if a == "G"]

        pos[0, :, :5] = prot_native.atom_positions[:,:5]
        cbs = np.array([extend(c,n,ca, 1.522, 1.927, -2.143) for c, n ,ca in zip(pos[0,:,2], pos[0,:,0], pos[0,:,1])])
        pos[0, gly_idx, 3] = cbs[gly_idx]
        atom_mask[0, :, :5] = 1

        template_native = {"template_aatype":np.array([jax.nn.one_hot(residue_constants.restype_order.get("A", residue_constants.restype_num),22)]*len(seq_native))[None],
                           "template_all_atom_masks": atom_mask,
                           "template_all_atom_positions":pos,
                           "template_domain_names":np.asarray(["None"])}
      elif args.nosc:
        pos = np.zeros([1,len(seq_native), 37, 3])
        atom_mask = np.zeros([1, len(seq_native), 37])

        pos[0, :, :5] = prot_native.atom_positions[:,:5]
        atom_mask[0, :, :5] = prot_native.atom_mask[:,:5]

        template_native = {"template_aatype":np.array(jax.nn.one_hot(prot_native.aatype,22))[None],
                            "template_all_atom_masks": atom_mask,
                            "template_all_atom_positions":pos,
                            "template_domain_names":np.asarray(["None"])}
      else:
        template_native = {"template_aatype":np.array(jax.nn.one_hot(prot_native.aatype,22))[None],
                            "template_all_atom_masks":prot_native.atom_mask[None],
                            "template_all_atom_positions":prot_native.atom_positions[None],
                            "template_domain_names":np.asarray(["None"])}

      features = make_processed_feature_dict(runner, seq_native, name=n + "_native", templates=template_native, seed=args.seed)
      result = parse_results(runner.predict(features), features)

      write_results(n, targets_hash, result, prot_native, 0.0, 1.0, -1, 1.0, -1, pdb_native, n + "_native.pdb", pdb_native, args)

    # run the model with the template-free structure passed as a template
    if n + "_none2x.pdb" not in finished_decoys and args.none2x:
      pdb_none = base_dir + "outputs/" + args.name + "/pdbs/" + n + "_none.pdb"
      prot_none = protein.from_pdb_string(pdb_to_string(pdb_none))

      if args.alanine:
        pos = np.zeros([1,len(seq_native), 37, 3])
        atom_mask = np.zeros([1, len(seq_native), 37])
        gly_idx = [i for i,a in enumerate(seq_native) if a == "G"]

        pos[0, :, :5] = prot_none.atom_positions[:,:5]
        cbs = np.array([extend(c,n,ca, 1.522, 1.927, -2.143) for c, n ,ca in zip(pos[0,:,2], pos[0,:,0], pos[0,:,1])])
        pos[0, gly_idx, 3] = cbs[gly_idx]
        atom_mask[0, :, :5] = 1

        template_none = {"template_aatype":np.array([jax.nn.one_hot(residue_constants.restype_order.get("A", residue_constants.restype_num),22)]*len(seq_native))[None],
                          "template_all_atom_masks": atom_mask,
                          "template_all_atom_positions":pos,
                          "template_domain_names":np.asarray(["None"])}
      elif args.nosc:
        pos = np.zeros([1,len(seq_native), 37, 3])
        atom_mask = np.zeros([1, len(seq_native), 37])

        pos[0, :, :5] = prot_none.atom_positions[:,:5]
        atom_mask[0, :, :5] = prot_none.atom_mask[:,:5]

        template_none = {"template_aatype":np.array(jax.nn.one_hot(prot_none.aatype,22))[None],
                          "template_all_atom_masks": atom_mask,
                          "template_all_atom_positions":pos,
                          "template_domain_names":np.asarray(["None"])}
      else:
        template_none = {"template_aatype":np.array(jax.nn.one_hot(prot_none.aatype,22))[None],
                          "template_all_atom_masks":prot_none.atom_mask[None],
                          "template_all_atom_positions":prot_none.atom_positions[None],
                          "template_domain_names":np.asarray(["None"])}

      features = make_processed_feature_dict(runner, seq_native, name=n + "_none2x", templates=template_none, seed=args.seed)
      result = parse_results(runner.predict(features), features)

      write_results(n, targets_hash, result, prot_native, -2, -2, -2, -2, -2, pdb_none, n + "_none2x.pdb", pdb_native, args)


    for d in decoy_dict[n]:
        pdb_in = base_dir + "decoy_set/decoy_pdbs/" + n + "/" + d[0]
        prot = protein.from_pdb_string(pdb_to_string(pdb_in))
        seq = "".join([residue_constants.restypes[x] for x in prot.aatype])
        assert seq == seq_native, base_dir + "decoy_set/decoy_pdbs/" + n + "/" + d[0]

        # run the model with a decoy passed as a template
        if args.alanine:
          pos = np.zeros([1,len(seq), 37, 3])
          atom_mask = np.zeros([1, len(seq), 37])
          gly_idx = [i for i,a in enumerate(seq) if a == "G"]

          pos[0, :, :5] = prot.atom_positions[:,:5]
          cbs = np.array([extend(c,n,ca, 1.522, 1.927, -2.143) for c, n ,ca in zip(pos[0,:,2], pos[0,:,0], pos[0,:,1])])
          pos[0, gly_idx, 3] = cbs[gly_idx]
          atom_mask[0, :, :5] = 1

          template = {"template_aatype":np.array([jax.nn.one_hot(residue_constants.restype_order.get("A", residue_constants.restype_num),22)]*len(seq))[None],
                      "template_all_atom_masks": atom_mask,
                      "template_all_atom_positions":pos,
                      "template_domain_names":np.asarray(["None"])}
        elif args.nosc:
          pos = np.zeros([1,len(seq), 37, 3])
          atom_mask = np.zeros([1, len(seq), 37])

          pos[0, :, :5] = prot.atom_positions[:,:5]
          atom_mask[0, :, :5] = prot.atom_mask[:,:5]

          template = {"template_aatype":np.array(jax.nn.one_hot(prot.aatype,22))[None],
                      "template_all_atom_masks": atom_mask,
                      "template_all_atom_positions":pos,
                      "template_domain_names":np.asarray(["None"])}
        else:
          template = {"template_aatype":np.array(jax.nn.one_hot(prot.aatype,22))[None],
                      "template_all_atom_masks":prot.atom_mask[None],
                      "template_all_atom_positions":prot.atom_positions[None],
                      "template_domain_names":np.asarray(["None"])}


        features = make_processed_feature_dict(runner, seq, name=n + "_" + d[0], templates=template, seed=args.seed)
        result = parse_results(runner.predict(features), features)

        write_results(n, targets_hash, result, prot_native, d[1], d[2], d[3], d[4], d[5], pdb_in, n + "_" + d[0], pdb_native, args, prot_decoy=prot)


with open(base_dir + "outputs/" + args.name + "/finished_targets.txt", 'a') as f:
  f.write(n + "\n")
