import numpy as np
import awkward as ak
import pickle
import time  

from File_loading import data, file_config
from Helper_functions import (compute_candidate_kinematics, get_displaced_jets)

start_time = time.time()

# -----------------------------------------------------------------------------
# Loading dataset with weights
# -----------------------------------------------------------------------------

print("\nCreating dataset...\n")

scale_factor = 1  # Temporary scaling
hlt_btag_thresholds = [0.38, 0.375, 0.31, 0.275]
btag_threshold = hlt_btag_thresholds[1] # defaults to 'tightest' threshold = 0.38

test_variables = [
    # Standard L1 jets
    "nL1Jet", "L1Jet_pt", "L1Jet_eta", "L1Jet_phi", "L1Jet_eta",

    # Displaced jets
    "nL1DisplacedJet", "L1DisplacedJet_pt", "L1DisplacedJet_eta", 
    "L1DisplacedJet_phi", "L1DisplacedJet_btagScore", "L1DisplacedJet_llpTagScore",

    # Offline jets
    "nJet", "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepFlavB", 

    # Generator-level variables for validation
    "nGenPart", "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_pdgId", "GenPart_genPartIdxMother",

    # L1 trigger 
    "L1_pPuppiHT400_pQuadJet70_55_40_40_final",
    
    # HLT triggers 
    "HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepCSV_2p4",
    "HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4",
    "HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepCSV_2p4",
    "HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4"
]

config = file_config(scale_factor=scale_factor) 

data_loader = data(
    signal_directories=[config["signal_directories"][0]],
    signal_file_ranges=[config["signal_file_ranges"][0]],
    signal_processes=[config["signal_processes"][0]],
    qcd_directories=config["qcd_directories"],
    qcd_file_ranges=config["qcd_file_ranges"],
    qcd_processes=config["qcd_processes"],
    cross_sections=config["cross_sections"], 
    luminosity=config["luminosity"],
    variables=test_variables,
    scale_factor=scale_factor
)

signal_df = data_loader.load_signal()
background_df = data_loader.load_background()
full_dataset = data_loader.load_full()

# -----------------------------------------------------------------------------
# Higgs candidate selection
# -----------------------------------------------------------------------------

print("\nPerforming Higgs candidate selection...\n")

# Efficiency counters
total_events = len(full_dataset)
total_signal = ak.sum(full_dataset["label"] == 1)
total_background = ak.sum(full_dataset["label"] == 0)

events_passing_4jets = 0
events_passing_leading_pt_cuts = 0
events_passing_trigger = 0
events_passing_btag_2b = 0

signal_events_4jets = 0
signal_events_pt = 0 
signal_events_trigger = 0
signal_events = 0

background_events_4jets = 0
background_events_pt = 0
background_events_trigger = 0
background_events = 0

events_passing_signal_region = 0
events_passing_control_region = 0
events_passing_region_selection = 0

# Store selected events
selected_events = []

# =================================== Event selection loop ===================================

for row in full_dataset:
    # == Jet veto map ==

    # Load in offline jets 
    pts, etas, phis, btag_scores = get_displaced_jets(row)

    valid_jets = (pts > 25) & (np.abs(etas) < 2.5)

    # keep only jets that satisfy the acceptance cuts
    pts      = pts [valid_jets]
    etas     = etas[valid_jets]
    phis     = phis[valid_jets]
    btag_scores = btag_scores[valid_jets]

    # Require at least 4 jets, otherwise skip this event
    if len(pts) < 4:
        continue
        
    events_passing_4jets += 1

    if row["label"] == 1:
        signal_events_4jets += 1

    if row["label"] == 0:
        background_events_4jets += 1 

    leading_pt_cuts = (35, 35, 35, 30) 
    sort_pt_idx = np.argsort(pts)[::-1]  # descending order
    if any(pts[sort_pt_idx[i]] < thr for i, thr in enumerate(leading_pt_cuts)):
        continue

    events_passing_leading_pt_cuts += 1

    if row["label"] == 1:
        signal_events_pt += 1

    if row["label"] == 0:
        background_events_pt += 1 
    
    trigger_mode = 'off'  # 'loose', 'tight', 'direct' or 'off'

    # Check if the event passes the HLT trigger requirements
    hlt_triggers_passed = [
        row["HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepCSV_2p4"],
        row["HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4"],
        row["HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepCSV_2p4"],
        row["HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4"]
    ]  

    num_hlt_passed = sum(hlt_triggers_passed)
    
    # Emulated trigger logic for both signal and background events
    sorted_pts = np.sort(pts)[::-1] 

    emulated_trigger_passed = False
    if trigger_mode == 'loose':
        if sorted_pts[0] > 70 and sorted_pts[1] > 40 and sorted_pts[2] > 30 and sorted_pts[3] > 30:
            emulated_trigger_passed = True

    elif trigger_mode == 'tight':
        if sorted_pts[0] > 75 and sorted_pts[1] > 60 and sorted_pts[2] > 45 and sorted_pts[3] > 40:
            emulated_trigger_passed = True

    elif trigger_mode == 'direct':
        # Determine thresholds based on which HLT passed
        if num_hlt_passed == 1:
            idx = hlt_triggers_passed.index(True)
            # first two triggers correspond to 200 HT and 70-40-30-30 jet pts
            if idx < 2:
                pt_thr = [70, 40, 30, 30]
                trigger_ht_threshold = 200
            else:
                pt_thr = [75, 60, 45, 40]
                trigger_ht_threshold = 330
        else:
            # default to the tighter thresholds if multiple triggers passed
            pt_thr = [75, 60, 45, 40]
            trigger_ht_threshold = 330
        
        if (sorted_pts[0] > pt_thr[0] and sorted_pts[1] > pt_thr[1]
            and sorted_pts[2] > pt_thr[2] and sorted_pts[3] > pt_thr[3]):
            emulated_trigger_passed = True
    
    elif trigger_mode == 'off':
        emulated_trigger_passed = True
    
    if not emulated_trigger_passed:
        continue

    # HT trigger logic 
    event_ht = np.sum(pts[pts > 30]) 

    emulated_ht_passed = False
    if trigger_mode == 'loose':
        if event_ht > 200:
            emulated_ht_passed = True

    elif trigger_mode == 'tight':
        if event_ht > 330:
            emulated_ht_passed = True
            
    elif trigger_mode == 'direct':
        if event_ht > trigger_ht_threshold:
            emulated_ht_passed = True

    elif trigger_mode == 'off':
        emulated_ht_passed = True
    
    if not emulated_ht_passed:
        continue
        
    # Set btag threshold based on trigger mode and HLT triggers passed
    if trigger_mode == 'direct':
        if num_hlt_passed == 1:
            # use the complementary b-tag threshold for trigger
            idx = hlt_triggers_passed.index(True)
            btag_threshold = hlt_btag_thresholds[idx]
        else:
            # default threshold when more than one trigger passed
            btag_threshold = hlt_btag_thresholds[1]
    
    events_passing_trigger += 1

    if row["label"] == 1:
        signal_events_trigger += 1

    if row["label"] == 0:
        background_events_trigger += 1 
    
    # == B-tag selection (2bs) ==

    # Skip events less than the b-tag threshold on the mean of the two highest btag jets
    top_two_btag_mean = np.mean(btag_scores[np.argsort(pts)[-2:]])
    if top_two_btag_mean < btag_threshold:
        continue
    
    events_passing_btag_2b += 1

    if row["label"] == 1:
        signal_events += 1

    if row["label"] == 0:
        background_events += 1 

    # =================================== Candidate selection ===================================

    # Select the four jets with highest b tag scores
    jet_indices = np.argsort(btag_scores)[::-1][:4]

    jets = [(pts[i], etas[i], phis[i]) for i in jet_indices]

    # Define possible dijet pairings for HH
    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    pairing_results = []

    # Loop through pairings to calcualte kinematics for each 
    for (pair1_jet1, pair1_jet2), (pair2_jet1, pair2_jet2) in pairings:
        m_pair1, pt_pair1, _, _, _, _, _ = compute_candidate_kinematics(jets[pair1_jet1][0], jets[pair1_jet2][0], jets[pair1_jet1][1], jets[pair1_jet2][1], jets[pair1_jet1][2], jets[pair1_jet2][2])
        m_pair2, pt_pair2, _, _, _, _, _ = compute_candidate_kinematics(jets[pair2_jet1][0], jets[pair2_jet2][0], jets[pair2_jet1][1], jets[pair2_jet2][1], jets[pair2_jet1][2], jets[pair2_jet2][2])

        if pt_pair2 > pt_pair1:
            # Swap the Higgs candidate information so the higher pt one is always H1
            m1, m2 = m_pair2, m_pair1
            pt1, pt2 = pt_pair2, pt_pair1
            H1_jet1, H1_jet2, H2_jet1, H2_jet2 = pair2_jet1, pair2_jet2, pair1_jet1, pair1_jet2
        else:
            m1, m2 = m_pair1, m_pair2
            pt1, pt2 = pt_pair1, pt_pair2
            H1_jet1, H1_jet2, H2_jet1, H2_jet2 = pair1_jet1, pair1_jet2, pair2_jet1, pair2_jet2

        # Use dHH metric
        pair_factor = 125 / 120
        dHH = abs(m1 - pair_factor * m2) / np.sqrt(1 + pair_factor**2)

        # Calculate RHH metric
        RHH = np.sqrt((m1-125)**2 + (m2-120)**2) 

        pairing_results.append((dHH, RHH, (H1_jet1, H1_jet2, H2_jet1, H2_jet2)))

    # Sort pairings by dHH metric ascending
    pairing_results.sort(key=lambda x: x[0])
    _, RHH, best_indices = pairing_results[0] # best pairing

    # # =================================== Signal and Control regions (SR4b, CR2b) ===================================
    # top_four_btag_mean = np.mean(np.sort(btag_scores)[-4:])
    
    # # Define region flags to track which region this event belongs to
    # is_in_signal_region = False
    # is_in_control_region = False
    
    # # Signal region (SR4b): RHH < 30 and good b-tagging (4 b-jets), only for signal events (label == 1)
    # if row["label"] == 1 and RHH < 30 and top_four_btag_mean > btag_threshold:
    #     is_in_signal_region = True
    #     events_passing_signal_region += 1
        
    # # Control region (CR2b): 30 <= RHH < 55, only for background events (label == 0)
    # elif row["label"] == 0 and RHH >= 30 and RHH < 55:
    #     is_in_control_region = True
    #     events_passing_control_region += 1
    
    # # If event doesn't fall into either region, skip it
    # if not (is_in_signal_region or is_in_control_region):
    #     continue
    
    # events_passing_region_selection += 1
    
    # Store the selected event with jet information
    selected_events.append({
        'event_data': row,
        'jets': jets,
        'best_indices': best_indices,
        'RHH': RHH
    })

# Print efficiency statistics    
print(f"\nEfficiency statistics:")

print(f"\nTotal efficiency\n")
print(f"  Total events: {total_events}")
print(f"  Total signal events: {total_signal} ({total_signal / total_events:.2%})")
print(f"  Total background events: {total_background} ({total_background / total_events:.2%})")
print(f"  Events passing 4 jets: {events_passing_4jets} ({events_passing_4jets / total_events:.2%})")
print(f"  Events passing leading pT cuts 35, 35, 30, 30: {events_passing_leading_pt_cuts} ({events_passing_leading_pt_cuts / events_passing_4jets:.2%})")
print(f"  Events passing HLT triggers: {events_passing_trigger} ({events_passing_trigger / events_passing_leading_pt_cuts:.2%})")
print(f"  Events passing btag (2b): {events_passing_btag_2b} ({events_passing_btag_2b / events_passing_trigger:.2%})")
print(f"  Total events passing selection: {events_passing_btag_2b} ({events_passing_btag_2b / total_events:.2%}, total events)")

print(f"\nSignal efficiency:\n")
print(f"  signal events passing 4 jets: {signal_events_4jets} ({signal_events_4jets / total_signal:.2%})")
print(f"  signal events passing leading pT cuts: {signal_events_pt} ({signal_events_pt / signal_events_4jets:.2%})")
print(f"  signal events passing HLT triggers: {signal_events_trigger} ({signal_events_trigger / signal_events_pt:.2%})")
print(f"  Total events passing selection: {signal_events} ({signal_events / total_events:.2%}, total events)")

print(f"\nBackground efficiency:\n")
print(f"  background events passing 4 jets: {background_events_4jets} ({background_events_4jets / total_background:.2%})")
print(f"  background events passing leading pT cuts: {background_events_pt} ({background_events_pt / background_events_4jets:.2%})")
print(f"  background events passing HLT triggers: {background_events_trigger} ({background_events_trigger / background_events_pt:.2%})")
print(f"  Total events passing selection: {background_events} ({background_events / total_events:.2%}, total events)")

# print(f"\nPost candidate selection:\n")
# print(f"  Events passing signal region SR4b: {events_passing_signal_region} ({events_passing_signal_region / events_passing_btag_2b:.2%})")
# print(f"  Events passing control region CR2b: {events_passing_control_region} ({events_passing_control_region / events_passing_btag_2b:.2%})")
# print(f"  Events passing region selection: {events_passing_region_selection} ({events_passing_region_selection / events_passing_btag_2b:.2%})")

# -----------------------------------------------------------------------------
# Save dataset
# -----------------------------------------------------------------------------

print("\nSaving selected candidates to pickle file...\n")

with open("higgs_candidates.pkl", "wb") as f:
    pickle.dump(selected_events, f)
    
print(f"Selected candidates saved to higgs_candidates.pkl")

# Calculate total runtime
end_time = time.time()
total_runtime = end_time - start_time
minutes = total_runtime / 60
seconds = total_runtime % 60

print(f"\nRun time: {minutes:.2f} minutes ({int(minutes)}m {seconds:.2f}s)")