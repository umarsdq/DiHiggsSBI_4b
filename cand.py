import numpy as np
import awkward as ak  # Now needed for reading the data
import uproot
import matplotlib.pyplot as plt

FILE_PATH = "/vols/cms/us322/events_4b/02b_final_events_4b/background/batch_1/run_01/tag_1_pythia8_events_delphes.root" # Bkgd

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def compute_candidate_kinematics(pt1, pt2, eta1, eta2, phi1, phi2):
    """
    Computes candidate four-vector kinematics from two jet inputs.
    """
    E1  = pt1 * np.cosh(eta1)
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    
    E2  = pt2 * np.cosh(eta2)
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    
    E  = E1 + E2
    px, py, pz = px1 + px2, py1 + py2, pz1 + pz2

    eps = 1e-12
    candidate_pt  = np.sqrt(px**2 + py**2)
    candidate_p = np.sqrt(px**2 + py**2 + pz**2)
    candidate_phi = np.arctan2(py, px)
    candidate_eta = 0.5 * np.log((candidate_p+pz + eps)/(candidate_p-pz + eps))
    candidate_mass = np.sqrt(max(E**2 - (px**2+py**2+pz**2), 0))
    candidate_rapidity = 0.5 * np.log((E+pz)/(E-pz)) if (E-pz)!=0 else 0.0

    return candidate_mass, candidate_pt, (E,px,py,pz), candidate_eta, candidate_phi, candidate_rapidity, candidate_p

# -----------------------------------------------------------------------------
# Loading dataset with weights
# -----------------------------------------------------------------------------

hlt_btag_thresholds = [0.38, 0.375, 0.31, 0.275]
btag_threshold = hlt_btag_thresholds[1] # defaults to 'tightest' threshold = 0.38

# Read only the GenJet branches we need to avoid the data corruption error
branches_to_read = [
    "GenJet/GenJet.PT",
    "GenJet/GenJet.Eta", 
    "GenJet/GenJet.Phi",
    "GenJet/GenJet.BTag"
]

with uproot.open(FILE_PATH) as file:
    arrays = file["Delphes"].arrays(branches_to_read, library="ak")
    
# Convert to a more convenient format for iteration
# Note: Each event can have variable number of jets
print(f"Number of events: {len(arrays)}")
print(f"Available branches: {list(arrays.fields)}")

# Let's examine the first few events
for i in range(min(3, len(arrays))):
    print(f"\nEvent {i}:")
    print(f"  Number of GenJets: {len(arrays['GenJet/GenJet.PT'][i])}")
    if len(arrays['GenJet/GenJet.PT'][i]) > 0:
        print(f"  First jet PT: {arrays['GenJet/GenJet.PT'][i][0]}")
        print(f"  First jet BTag: {arrays['GenJet/GenJet.BTag'][i][0]}")


# -----------------------------------------------------------------------------
# Higgs candidate selection
# -----------------------------------------------------------------------------

# Store selected events
selected_events = []

# Store candidate masses for plotting
cand1_masses = []
cand2_masses = []

# =================================== Event selection loop ===================================

for event_idx in range(len(arrays)):
    # Load in offline jets from the arrays
    pts = arrays["GenJet/GenJet.PT"][event_idx]
    etas = arrays["GenJet/GenJet.Eta"][event_idx] 
    phis = arrays["GenJet/GenJet.Phi"][event_idx]
    btag_scores = arrays["GenJet/GenJet.BTag"][event_idx]
    
    # Convert awkward arrays to numpy arrays for easier manipulation
    pts = np.array(pts)
    etas = np.array(etas)
    phis = np.array(phis)
    btag_scores = np.array(btag_scores)
    
    # # == B-tag selection (2bs) ==

    # # Skip events less than the b-tag threshold on the mean of the two highest btag jets
    # if len(btag_scores) >= 2:
    #     top_two_btag_mean = np.mean(btag_scores[np.argsort(pts)[-2:]])
    #     if top_two_btag_mean < btag_threshold:
    #         continue 

    # =================================== Candidate selection ===================================

    # Ensure we have at least 4 jets
    if len(pts) < 4:
        continue

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

    # =================================== Create candidate features ===================================

    # Use the best pairing to define the two Higgs candidates:
    cand1_mass, cand1_pt, cand1_fourvec, cand1_eta, cand1_phi, _, _ = compute_candidate_kinematics(
        jets[best_indices[0]][0], jets[best_indices[1]][0], 
        jets[best_indices[0]][1], jets[best_indices[1]][1],
        jets[best_indices[0]][2], jets[best_indices[1]][2]
    )
    
    cand2_mass, cand2_pt, cand2_fourvec, cand2_eta, cand2_phi, _, _ = compute_candidate_kinematics(
        jets[best_indices[2]][0], jets[best_indices[3]][0],
        jets[best_indices[2]][1], jets[best_indices[3]][1],
        jets[best_indices[2]][2], jets[best_indices[3]][2]
    )
    
    # Store the masses for plotting
    cand1_masses.append(cand1_mass)
    cand2_masses.append(cand2_mass)

# =================================== Plot 2D Histogram ===================================

plt.figure(figsize=(10, 8))
plt.hist2d(cand1_masses, cand2_masses, bins=50, cmap='viridis')
plt.colorbar(label='Number of Events')
plt.xlim(0, 240)
plt.ylim(0, 240)
plt.xlabel('Candidate 1 Mass (GeV)')
plt.ylabel('Candidate 2 Mass (GeV)')
plt.title('2D Histogram of Higgs Candidate Masses')
plt.grid(True, alpha=0.3)

# Add reference lines for Higgs mass
plt.axvline(x=125, color='red', linestyle='--', alpha=0.7, label='Higgs mass (125 GeV)')
plt.axhline(y=125, color='red', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('higgs_candidate_masses_2d_histogram.pdf')

print(f"Processed {len(cand1_masses)} events")
print(f"Mean candidate 1 mass: {np.mean(cand1_masses):.2f} GeV")
print(f"Mean candidate 2 mass: {np.mean(cand2_masses):.2f} GeV") 