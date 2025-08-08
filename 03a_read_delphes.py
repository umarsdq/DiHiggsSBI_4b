#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
import math
import os

from madminer.delphes import DelphesReader
import argparse

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.DEBUG)

        
import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
    
parser = argparse.ArgumentParser()
parser.add_argument("-p","--process_code",help="Choose from signal_sm, signal_supp, or background")
parser.add_argument("-b","--batch_index",help="batch_index")
parser.add_argument("-supp_id","--supp_id",help="Index of non_SM benchmark that events were generated at")
parser.add_argument("-dr","--delphes_run",action="store_true",help="Whether Delphes has been run on the events or not")
parser.add_argument("-start","--start",help="MadGraph run start index")
parser.add_argument("-stop","--stop",help="Madgraph run stop index")
args = parser.parse_args()
    
mg_dir = workflow["madgraph"]["dir"]
delphes = DelphesReader(workflow["morphing_setup"])
  
if "background" in args.process_code:
    is_background = True
elif "signal" in args.process_code:
    is_background = False
    
if args.process_code != "signal_supp":
    path_to_events_dir = "{input_dir_prefix}/{process_code}/batch_{batch_index}/".format(input_dir_prefix=workflow["delphes"]["input_dir_prefix"], process_code = args.process_code, batch_index=args.batch_index)
    sampled_from_benchmark = "sm"
else:
    path_to_events_dir = "{input_dir_prefix}/{process_code}/mb_vector_{supp_id}/batch_{batch_index}/".format(input_dir_prefix=workflow["delphes"]["input_dir_prefix"], process_code = args.process_code, batch_index=args.batch_index, supp_id = args.supp_id)
    sampled_from_benchmark = f"morphing_basis_vector_{args.supp_id}"
    
    
for run_id in range(int(args.start), int(args.stop)+1):   
    
    # background events have not gone through MadSpin
    if "background" in args.process_code:
        loc_dir = f"{path_to_events_dir}/run_{str(run_id).zfill(2)}"
    else:
        loc_dir = f"{path_to_events_dir}/run_{str(run_id).zfill(2)}_decayed_1"
    
    print(loc_dir)

    if args.delphes_run: 
        delphes.add_sample(lhe_filename=f"{loc_dir}/unweighted_events.lhe.gz",
                            hepmc_filename=f"{loc_dir}/tag_1_pythia8_events.hepmc.gz",
                               delphes_filename=f"{loc_dir}/tag_1_pythia8_events_delphes.root",
                               weights="lhe",
                            sampled_from_benchmark=sampled_from_benchmark,
                            is_background=is_background,
                            k_factor= 1.0,)
    else: 
        delphes.add_sample(lhe_filename=f"{loc_dir}/unweighted_events.lhe.gz",
                            hepmc_filename=f"{loc_dir}/tag_1_pythia8_events.hepmc.gz",
                               weights="lhe",
                            sampled_from_benchmark=sampled_from_benchmark,
                            is_background=is_background,
                            k_factor= 1.0,)

                                                                       
# Now we run Delphes on these samples (you can also do this externally and then add the keyword `delphes_filename` when calling `DelphesReader.add_sample()`):


if args.delphes_run: 
    print("Delphes has already been run.")
else:
    delphes.run_delphes(
        delphes_directory=mg_dir + "/HEPTools/Delphes-3.5.0/", # For latest madgraph version.
        delphes_card="cards/delphes_card_HLLHC.tcl",
        log_file=f"delpheslogs/delphes_{args.process_code}_batch{args.batch_index}{'_mb' + str(args.supp_id) if args.supp_id else ''}.log",
    )

"""
CUSTOM FUNCTIONS TO ISOLATE THE BJETS (HH → 4b)
and build H→bb candidates following the paper selections
"""

def get_four_bjets(j):
    # Return the four leading b-tagged jets
    bjets = [jet for jet in j if jet.b_tag == 1]
    if len(bjets) >= 4:
        return True, bjets[:4]
    return False, None

def get_four_jets(j):
    # Return the four leading AK4 jets (no b-tagging requirement)
    if len(j) >= 4:
        return True, j[:4]
    return False, None

def count_bjets(l,a,j,met):
    return sum(1 for jet in j if jet.b_tag == 1)

def count_electrons_veto(l,a,j,met):
    # electrons: pT > 10 GeV, |eta| < 2.5
    return sum(1 for lep in l if abs(int(lep.pdgid)) == 11 and lep.pt > 10.0 and abs(lep.eta) < 2.5)

def count_muons_veto(l,a,j,met):
    # muons: pT > 10 GeV, |eta| < 2.4
    return sum(1 for lep in l if abs(int(lep.pdgid)) == 13 and lep.pt > 10.0 and abs(lep.eta) < 2.4)

def count_additional_jets(l,a,j,met):
    # additional AK4 jets with pT > 25 GeV and |eta| < 4.7
    # applying special rule: if 2.5 < |eta| < 3.0 then require pT > 50 GeV
    found_b, bjets = get_four_bjets(j)
    used = set(bjets) if found_b else set()
    count = 0
    for jet in j:
        if jet in used:
            continue
        if abs(jet.eta) > 4.7:
            continue
        pt_threshold = 50.0 if 2.5 < abs(jet.eta) < 3.0 else 25.0
        if jet.pt > pt_threshold:
            count += 1
    return count

def get_b_pt(l,a,j,met,index):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[index].pt
    return np.nan

def get_b_eta(l,a,j,met,index):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[index].eta
    return np.nan

def get_bb_deltaR(l,a,j,met,i1,i2):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[i1].deltaR(bjets[i2])
    return np.nan

def get_mbb(l,a,j,met,i1,i2):
    found, bjets = get_four_bjets(j)
    if found:
        return (bjets[i1] + bjets[i2]).m
    return np.nan

def get_ptbb(l,a,j,met,i1,i2):
    found, bjets = get_four_bjets(j)
    if found:
        return (bjets[i1] + bjets[i2]).pt
    return np.nan

def get_mtot_4b(l,a,j,met):
    found, bjets = get_four_bjets(j)
    if found:
        return sum(bjets, start=bjets[0].__class__(0)).m  # vector sum
    return np.nan

# ------------------------
# H→bb candidate building
# ------------------------

def _pair_bjets_by_min_dhh(bjets):
    # Implements min-dHH pairing with center (125,120) GeV and H1 = higher-pT candidate
    if len(bjets) < 4:
        return None
    idx = [0,1,2,3]
    pairing_options = [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]

    results = []
    for (i1,j1), (i2,j2) in pairing_options:
        # compute candidate masses and pts
        cand1 = bjets[i1] + bjets[j1]
        cand2 = bjets[i2] + bjets[j2]
        # order so that H1 has larger pT
        if cand2.pt > cand1.pt:
            m1, m2 = cand2.m, cand1.m
            pt1, pt2 = cand2.pt, cand1.pt
            pair = (i2, j2, i1, j1)
        else:
            m1, m2 = cand1.m, cand2.m
            pt1, pt2 = cand1.pt, cand2.pt
            pair = (i1, j1, i2, j2)
        # dHH with center (125,120)
        pair_factor = 125.0 / 120.0
        dHH = abs(m1 - pair_factor * m2) / math.sqrt(1.0 + pair_factor**2)
        # RHH
        RHH = math.sqrt((m1 - 125.0)**2 + (m2 - 120.0)**2)
        results.append((dHH, RHH, pt1, pair))

    # sort by dHH
    results.sort(key=lambda x: x[0])

    # tie-breaker: if within 30 GeV in dHH, choose highest pT(H1)
    best = results[0]
    if len(results) > 1 and (results[1][0] - results[0][0]) < 30.0:
        best = max(results[:2], key=lambda x: x[2])

    return best  # (dHH, RHH, pt1, (H1_i, H1_j, H2_i, H2_j))

def _get_selected_pair(j):
    found, bjets = get_four_bjets(j)
    if not found:
        return None
    return _pair_bjets_by_min_dhh(bjets)

def get_h1_mass(l,a,j,met):
    sel = _get_selected_pair(j)
    if sel is None:
        return np.nan
    _, _, _, (i1, j1, _, _) = sel
    found, bjets = get_four_bjets(j)
    return (bjets[i1] + bjets[j1]).m

def get_h2_mass(l,a,j,met):
    sel = _get_selected_pair(j)
    if sel is None:
        return np.nan
    _, _, _, (_, _, i2, j2) = sel
    found, bjets = get_four_bjets(j)
    return (bjets[i2] + bjets[j2]).m

def get_h1_pt(l,a,j,met):
    sel = _get_selected_pair(j)
    if sel is None:
        return np.nan
    _, _, _, (i1, j1, _, _) = sel
    found, bjets = get_four_bjets(j)
    return (bjets[i1] + bjets[j1]).pt

def get_h2_pt(l,a,j,met):
    sel = _get_selected_pair(j)
    if sel is None:
        return np.nan
    _, _, _, (_, _, i2, j2) = sel
    found, bjets = get_four_bjets(j)
    return (bjets[i2] + bjets[j2]).pt

def get_h1_deltaR(l,a,j,met):
    sel = _get_selected_pair(j)
    if sel is None:
        return np.nan
    _, _, _, (i1, j1, _, _) = sel
    found, bjets = get_four_bjets(j)
    return bjets[i1].deltaR(bjets[j1])

def get_h2_deltaR(l,a,j,met):
    sel = _get_selected_pair(j)
    if sel is None:
        return np.nan
    _, _, _, (_, _, i2, j2) = sel
    found, bjets = get_four_bjets(j)
    return bjets[i2].deltaR(bjets[j2])

def get_RHH(l,a,j,met):
    sel = _get_selected_pair(j)
    if sel is None:
        return np.nan
    _, RHH, _, _ = sel
    return RHH

# ------------------------
# utilities for leading jets
# ------------------------

def get_j_pt(l,a,j,met,index):
    found, jets = get_four_jets(j)
    if found:
        return jets[index].pt
    return np.nan

def get_j_eta(l,a,j,met,index):
    found, jets = get_four_jets(j)
    if found:
        return jets[index].eta
    return np.nan


"""
MAIN ANALYSIS
"""

def add_observables(delphes):

    # Number of b-jets
    delphes.add_observable_from_function("num_bjets", count_bjets, required=True)
    # Lepton veto counts
    delphes.add_observable_from_function("n_electrons_veto", count_electrons_veto, required=True)
    delphes.add_observable_from_function("n_muons_veto", count_muons_veto, required=True)

    # Leading jet pT and eta (no b-tag requirement) for trigger-like selection
    for i in range(4):
        delphes.add_observable_from_function(f"j{i}_pt", lambda l,a,j,met,ii=i: get_j_pt(l,a,j,met,ii), required=True)
        delphes.add_observable_from_function(f"j{i}_eta", lambda l,a,j,met,ii=i: get_j_eta(l,a,j,met,ii), required=True)

    # b-jet pT and eta (for diagnostics)
    for i in range(4):
        delphes.add_observable_from_function(f"b{i}_pt", lambda l,a,j,met,ii=i: get_b_pt(l,a,j,met,ii), required=True)
        delphes.add_observable_from_function(f"b{i}_eta", lambda l,a,j,met,ii=i: get_b_eta(l,a,j,met,ii), required=True)

    # ΔR between Higgs candidate jets
    delphes.add_observable_from_function("bb1_deltaR", get_h1_deltaR, required=True)
    delphes.add_observable_from_function("bb2_deltaR", get_h2_deltaR, required=True)

    # Invariant masses for Higgs candidates
    delphes.add_observable_from_function("m_bb1", get_h1_mass, required=True)
    delphes.add_observable_from_function("m_bb2", get_h2_mass, required=True)

    # pT of Higgs candidates
    delphes.add_observable_from_function("pt_bb1", get_h1_pt, required=True)
    delphes.add_observable_from_function("pt_bb2", get_h2_pt, required=True)

    # Total invariant mass of all 4 b-jets
    delphes.add_observable_from_function("m_tot_4b", get_mtot_4b, required=True)

    # RHH selection variable
    delphes.add_observable_from_function("RHH", get_RHH, required=True)

    # Additional jets and VBF helpers
    delphes.add_observable_from_function("n_additional_jets", count_additional_jets, required=True)

def add_cuts_and_efficiencies(delphes, region=None):

    # Require ≥4 b-jets
    delphes.add_cut('num_bjets>=4')

    # Lepton veto (no e or mu passing loose kinematics)
    delphes.add_cut('n_electrons_veto==0')
    delphes.add_cut('n_muons_veto==0')

    # Jet pT cuts (trigger plateau, 2023 parkingHH: 35/35/35/30 GeV)
    delphes.add_cut('j0_pt>35')
    delphes.add_cut('j1_pt>35')
    delphes.add_cut('j2_pt>35')
    delphes.add_cut('j3_pt>30')

    # |eta| cuts for leading jets
    delphes.add_cut('abs(j0_eta)<2.5')
    delphes.add_cut('abs(j1_eta)<2.5')
    delphes.add_cut('abs(j2_eta)<2.5')
    delphes.add_cut('abs(j3_eta)<2.5')

    # RHH selection around (125,120) GeV
    delphes.add_cut('RHH<55')

    # Higgs mass windows
    delphes.add_cut('abs(m_bb1-125)<25')
    delphes.add_cut('abs(m_bb2-125)<25')

add_observables(delphes)
add_cuts_and_efficiencies(delphes)
    
# 4. Run analysis
delphes.analyse_delphes_samples()

# 5. Save results into new .h5 file

if args.process_code != "signal_supp":
    delphes.save("{delphes_output_data}_{process_code}_batch_{batch_index}.h5".format(delphes_output_data=workflow["delphes"]["output_file"], process_code=args.process_code, batch_index=args.batch_index))
else: 
    delphes.save("{delphes_output_data}_{process_code}_{supp_id}_batch_{batch_index}.h5".format(delphes_output_data=workflow["delphes"]["output_file"], process_code=args.process_code, batch_index=args.batch_index, supp_id = args.supp_id))
