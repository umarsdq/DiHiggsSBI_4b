import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

# -----------------------------------------------------------------------------
# AxE Gains
# -----------------------------------------------------------------------------
def b_quarks_from_higgs(row, pt_cut=20, eta_cut=2.5):
    """
    For a given event, counts the number of b quarks originating from a Higgs
    and returns their count, pt, eta and phi values.
    """
    pdgId, genPartIdxMother, pt, eta, phi = get_gen_part(row)

    count = 0
    selected_pts, selected_etas, selected_phis = [], [], []

    for i in range(len(pdgId)):
        if abs(pdgId[i]) == 5 and pt[i] > pt_cut and abs(eta[i]) < eta_cut:
            mother_idx = int(genPartIdxMother[i])
            
            if mother_idx >= 0 and abs(pdgId[mother_idx]) == 25:
                count += 1
                selected_pts.append(pt[i])
                selected_etas.append(eta[i])
                selected_phis.append(phi[i])

    return count, selected_pts, selected_etas, selected_phis

def jet_gen_matching(df, dR_cut, get_jets_func, pt_cut, eta_cut):
    """
    Matches generated b-quarks from a Higgs to jets and counts events with
    at least 4 matches within a delta R cut.
    """
    passing_events = 0
    total_considered = 0

    for row in df:
        count, _, gen_etas, gen_phis = b_quarks_from_higgs(row, pt_cut, eta_cut)

        if count < 4:
            continue

        total_considered += 1
        jet_data = get_jets_func(row)

        if len(jet_data[0]) == 0:
            continue

        good_matches = 0
        for i in range(4):
            best_dR = float('inf')

            for j in range(len(jet_data[1])):
                dphi = abs(gen_phis[i] - jet_data[2][j])

                if dphi > np.pi:
                    dphi = 2 * np.pi - dphi

                deta = abs(gen_etas[i] - jet_data[1][j])
                dR = np.sqrt(deta**2 + dphi**2)

                if dR < best_dR:
                    best_dR = dR

            if best_dR < dR_cut:
                good_matches += 1

        if good_matches == 4:
            passing_events += 1

    return passing_events, total_considered

def jet_gen_match_pass(row, dR_cut, jets_tuple, pt_cut, eta_cut):
    """
    Matches generated b-quarks from a Higgs to jets and checks if the event has
    at least 4 matches within a delta R cut.
    Uses the top 4 pT b-quarks for matching.
    Prevents double counting by ensuring each jet is matched to at most one gen particle.
    
    Args:
        row: Single event row
        dR_cut: Delta R matching threshold
        eta_col: Column name for jet eta values
        phi_col: Column name for jet phi values
        pt_cut: pT threshold for gen b-quarks
        eta_cut: eta threshold for gen b-quarks
        
    Returns:
        Boolean: True if all 4 b-quarks are matched to jets, False otherwise
    """
    count, gen_pts, gen_etas, gen_phis = b_quarks_from_higgs(row, pt_cut, eta_cut)

    # Sort gen particles by descending pT
    gen_pts = np.array(gen_pts)
    gen_etas = np.array(gen_etas)
    gen_phis = np.array(gen_phis)
    
    # Return False if we don't have enough gen particles
    if count < 4:
        return False
        
    sorted_indices = np.argsort(-gen_pts)
    gen_pts = gen_pts[sorted_indices]
    gen_etas = gen_etas[sorted_indices]
    gen_phis = gen_phis[sorted_indices]

    jets = jets_tuple
    eta, phi = jets[1], jets[2]

    # Return False if no jets are found
    if len(eta) == 0:
        return False

    good_matches = 0
    matched_jets = []  # Keep track of which jets have been matched already
    
    for i in range(4):
        best_dR = float('inf')
        best_jet_idx = -1  # Store the index of the best matching jet

        for j in range(len(eta)):
            # Skip jets that have already been matched
            if j in matched_jets:
                continue
                
            dphi = abs(gen_phis[i] - phi[j])

            if dphi > np.pi:
                dphi = 2 * np.pi - dphi

            deta = abs(gen_etas[i] - eta[j])
            dR = np.sqrt(deta**2 + dphi**2)

            if dR < best_dR:
                best_dR = dR
                best_jet_idx = j

        if best_dR < dR_cut and best_jet_idx != -1:
            good_matches += 1
            matched_jets.append(best_jet_idx)  # Mark this jet as matched

    return good_matches == 4

def get_gen_part(row):
    """
    Returns generator particle PDG IDs, mother indices, pt, eta, and phi values.
    """
    return np.array(row["GenPart_pdgId"]), np.array(row["GenPart_genPartIdxMother"]), np.array(row["GenPart_pt"]), np.array(row["GenPart_eta"]), np.array(row["GenPart_phi"])

def get_L1_jets(row):
    """
    Returns L1 jet pt, eta, and phi values.
    """
    return np.array(row["L1Jet_pt"]), np.array(row["L1Jet_eta"]), np.array(row["L1Jet_phi"])

def get_displaced_jets(row):
    """
    Returns displaced L1 jet pt, eta, phi values and b-tag scores.
    """
    return (np.array(row["L1DisplacedJet_pt"]), np.array(row["L1DisplacedJet_eta"]),
            np.array(row["L1DisplacedJet_phi"]), np.array(row["L1DisplacedJet_btagScore"]))

def get_offline_jets(row):
    """
    Returns offline jet pt, eta, phi values and b-tag scores.
    """
    return (np.array(row["Jet_pt"]), np.array(row["Jet_eta"]),
            np.array(row["Jet_phi"]), np.array(row["Jet_btagDeepFlavB"]))

def n_jets_sort(row, n_jets, get_jets_func):
    """
    Returns the first n_jets jet pt, eta, phi values sorted by descending pt.
    """
    pts, etas, phis = get_jets_func(row)[:3]
    sorted_indices = np.argsort(-pts)
    n = min(n_jets, len(pts))
    return pts[sorted_indices][:n], etas[sorted_indices][:n], phis[sorted_indices][:n]

def compute_HT(row, level='offline', pt_min=40, eta_max=3, return_count=False, n_jets_exact=None, n_jets_min=None):
    """
    Compute HT (scalar sum of jet pTs) for offline or generator level jets.
    Parameters:
        row: Event data row
        level: 'offline' or 'gen' to specify jet collection
        pt_min: Minimum pT threshold for jets
        eta_max: Maximum |η| for jets
        return_count: If True, returns (HT, count) tuple for gen level
        n_jets_exact: If specified, requires exactly this many jets to pass cuts
        n_jets_min: If specified, requires at least this many jets to pass cuts
    """
    if level == 'gen':
        # Generator-level HT including only b quarks from Higgs decays
        ht = 0
        count = 0
        
        try:
            pts = np.array(row["GenPart_pt"])
            etas = np.array(row["GenPart_eta"])
            pdgIds = np.array(row["GenPart_pdgId"])
            mother_indices = np.array(row["GenPart_genPartIdxMother"])
            
            for i in range(len(pts)):
                if abs(pdgIds[i]) == 5:  # b quark
                    mother_idx = mother_indices[i]
                    if mother_idx >= 0 and mother_idx < len(pdgIds) and abs(pdgIds[mother_idx]) == 25:
                        ht += pts[i]
                        count += 1
        except (KeyError, AttributeError, ValueError, IndexError):
            return (0, 0) if return_count else 0  # Return zero if generator info is missing

        return (ht, count) if return_count else ht

    elif level == 'offline':
        try:
            pts = np.array(row["Jet_pt"])
            etas = np.array(row["Jet_eta"])
        except (KeyError, AttributeError, ValueError, IndexError):
            return 0  # Return 0 HT if jet data is missing

        # Check if essential arrays are empty
        if len(pts) == 0 or len(etas) == 0:
            return 0  # Return 0 HT if jet data is missing

        valid = (pts > pt_min) & (np.abs(etas) < eta_max)
        valid_pts = pts[valid]
        n_valid_jets = len(valid_pts)
        
        # Handle exact jet count requirement
        if n_jets_exact is not None:
            # Compute HT only if the number of valid jets matches n_jets_exact
            if n_valid_jets == n_jets_exact:
                ht = np.sum(valid_pts)
            else:
                ht = 0  

        # Handle minimum jet count requirement
        elif n_jets_min is not None:
            # Compute HT only if there are at least n_jets_min valid jets
            if n_valid_jets >= n_jets_min:
                ht = np.sum(valid_pts)
            else:
                ht = 0  
        else:
            # Default behavior: sum HT for all valid jets
            ht = np.sum(valid_pts)
        
        # return_count is ignored for offline level
        return ht

    else:
        raise ValueError(f"Invalid level: {level}.")

def passes_offline_requirements(row, config):
    """
    Applies HLT trigger, HT, leading jet pT, b-tag, and matching requirements.
    """
    # Trigger
    if row[config["hlt_trigger"]] != 1:
        return False

    # HT
    if compute_HT(row, level='offline', pt_min=config["pt_min_for_ht"], eta_max=2.4) < config["ht_threshold"]:
        return False

    # Leading jets
    offline_pts = np.array(row["Jet_pt"])
    sorted_indices = np.argsort(-offline_pts)
    sorted_pts = offline_pts[sorted_indices]
    sorted_btags = np.array(row["Jet_btagDeepFlavB"])[sorted_indices]

    if len(sorted_pts) < len(config["leading_jet_pts"]):
        return False

    for i, thr in enumerate(config["leading_jet_pts"]):
        if sorted_pts[i] < thr:
            return False

    if np.count_nonzero(sorted_btags > config["btag_threshold"]) < config["btag_count_required"]:
        return False
        
    # Matching
    row_array = ak.Array([row])
    match_pass, match_total = jet_gen_matching(row_array, 0.4, get_offline_jets, 0, 999)
    if match_total == 0 or match_pass < 1:
        return False
    return True 

# -----------------------------------------------------------------------------
# Features
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

def delta_R(eta1, phi1, eta2, phi2):
    """
    Compute delta R, delta phi, and delta eta between two objects.
    """
    dphi = abs(phi1 - phi2)

    if dphi > np.pi:
        dphi = 2 * np.pi - dphi

    deta = abs(eta1 - eta2)
    dR = np.sqrt(dphi**2 + deta**2)
    return dR, dphi, deta

def boost_fourvector(fourvec, beta):
    """
    Boosts a four-vector (E,px,py,pz) by a velocity beta=(βx,βy,βz).
    """
    beta = np.array(beta)
    beta2 = np.dot(beta, beta)
    if beta2 >= 1:
        raise ValueError("Beta squared must be less than 1")
    gamma = 1.0 / np.sqrt(1 - beta2)
    E, px, py, pz = fourvec
    p = np.array([px, py, pz])
    bp = np.dot(beta, p)
    E_prime = gamma * (E - bp)

    # Decompose momentum into parallel and perpendicular components
    if beta2 > 0:
        p_parallel = (np.dot(p, beta) / beta2) * beta
    else:
        p_parallel = np.zeros_like(p)
    p_perp = p - p_parallel
    p_parallel_prime = gamma * (p_parallel - beta * E)
    p_prime = p_perp + p_parallel_prime

    return (E_prime, p_prime[0], p_prime[1], p_prime[2])

def compute_helicity_angle(candidate_fourvec, jet_fourvec):
    """
    Computes |cosθ| for a chosen decay product.
    The candidate_fourvec (E,px,py,pz) defines the boost from lab to candidate rest frame.
    The jet_fourvec is the four-vector of one of the daughter jets.
    """
    E, px, py, pz = candidate_fourvec
    if E == 0: return 0.0
    beta = (px/E, py/E, pz/E)  # boost velocity of candidate
    jet_boosted = boost_fourvector(jet_fourvec, beta)

    # Define the candidate-flight direction (from lab)
    p_candidate = np.array([px, py, pz])
    norm_candidate = np.linalg.norm(p_candidate)

    if norm_candidate == 0:
        return 0.0
    
    candidate_dir = p_candidate / norm_candidate
    jet_p = np.array([jet_boosted[1], jet_boosted[2], jet_boosted[3]])
    norm_jet = np.linalg.norm(jet_p)

    if norm_jet == 0:
        return 0.0
    cos_theta = np.dot(jet_p, candidate_dir) / norm_jet

    return abs(cos_theta)

def compute_four_vector(pt, eta, phi, mass):
    """
    Compute the four-vector components (E, px, py, pz) from kinematic variables.
    """
    E = np.sqrt(pt**2 + mass**2) * np.cosh(eta)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = np.sqrt(pt**2 + mass**2) * np.sinh(eta)
    return E, px, py, pz

def compute_kinematics_from_four_vector(fourvec):
    """
    Compute pt, eta, mass from a four-vector (E, px, py, pz).
    """
    E, px, py, pz = fourvec
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    eta = 0.5 * np.log((p + pz) / (p - pz)) if (p - pz) != 0 else 0.0
    mass = np.sqrt(max(E**2 - (px**2 + py**2 + pz**2), 0))
    return pt, eta, mass

def plot_feature(field, xlabel, title, filename, full_dataset, xlim=None, log_scale=False, flatten=False, smooth_sigma=None):
    signal_mask = full_dataset["label"] == 1
    
    # Get signal and background data
    signal_data_ak = full_dataset[signal_mask][field]
    bkg_data_ak = full_dataset[~signal_mask][field]
    signal_weights_ak = full_dataset[signal_mask]["weight"]
    bkg_weights_ak = full_dataset[~signal_mask]["weight"]

    if flatten:
        # Flatten data and broadcast weights 
        signal_data = ak.to_numpy(ak.flatten(signal_data_ak))
        bkg_data = ak.to_numpy(ak.flatten(bkg_data_ak))
        signal_weights = ak.to_numpy(ak.flatten(ak.broadcast_arrays(signal_weights_ak, signal_data_ak)[0]))
        bkg_weights = ak.to_numpy(ak.flatten(ak.broadcast_arrays(bkg_weights_ak, bkg_data_ak)[0]))

    else:
        signal_data = ak.to_numpy(signal_data_ak)
        bkg_data = ak.to_numpy(bkg_data_ak)
        signal_weights = ak.to_numpy(signal_weights_ak)
        bkg_weights = ak.to_numpy(bkg_weights_ak)

    # Remove NaN values and infinities
    signal_valid = np.isfinite(signal_data)
    signal_data = signal_data[signal_valid]
    signal_weights = signal_weights[signal_valid]

    bkg_valid = np.isfinite(bkg_data)
    bkg_data = bkg_data[bkg_valid]
    bkg_weights = bkg_weights[bkg_valid]
    
    sum_signal_weights = np.sum(signal_weights)
    sum_bkg_weights = np.sum(bkg_weights)

    # Normalise weights to unity
    if sum_signal_weights > 0:
        signal_weights = signal_weights / sum_signal_weights

    if sum_bkg_weights > 0:
        bkg_weights = bkg_weights / sum_bkg_weights
    
    if xlim is None:
        bins = 50
    else:
        bins = np.linspace(xlim[0], xlim[1], 25)
    
    # Precompute raw histogram counts and sum of squares for error bars
    if sum_bkg_weights > 0:
        bkg_raw_counts, bin_edges = np.histogram(bkg_data, bins=bins, weights=bkg_weights)
        bkg_raw_sumw2, _ = np.histogram(bkg_data, bins=bins, weights=bkg_weights**2)
        bkg_errors = np.sqrt(bkg_raw_sumw2)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if sum_signal_weights > 0:
        sig_raw_counts, _ = np.histogram(signal_data, bins=bins, weights=signal_weights)
        sig_raw_sumw2, _ = np.histogram(signal_data, bins=bins, weights=signal_weights**2)
        sig_errors = np.sqrt(sig_raw_sumw2)

    plt.figure(figsize=(8, 8))
    # Plot background, optionally smoothing counts to emphasize shape
    if sum_bkg_weights > 0:
        if smooth_sigma is not None:
            # compute and smooth histogram counts
            bkg_counts, bkg_edges = np.histogram(bkg_data, bins=bins, weights=bkg_weights)
            bkg_counts = gaussian_filter1d(bkg_counts, sigma=smooth_sigma)
            bkg_centers = (bkg_edges[:-1] + bkg_edges[1:]) / 2
            plt.plot(bkg_centers, bkg_counts, color='blue', label='Background (QCD)')
        else:
            plt.hist(bkg_data, bins=bins, histtype='stepfilled', color='blue', alpha=0.7,
                     weights=bkg_weights, label='Background (QCD)', density=False)

    # Plot signal, optionally smoothing (to compare shapes)
    if sum_signal_weights > 0:
        if smooth_sigma is not None:
            sig_counts, sig_edges = np.histogram(signal_data, bins=bins, weights=signal_weights)
            sig_counts = gaussian_filter1d(sig_counts, sigma=smooth_sigma)
            sig_centers = (sig_edges[:-1] + sig_edges[1:]) / 2
            plt.plot(sig_centers, sig_counts, color='red', label='Signal (ggf_sm)')
        else:
            plt.hist(signal_data, bins=bins, histtype='stepfilled', color='red', alpha=0.7,
                     weights=signal_weights, label='Signal (ggf_sm)', density=False)
    
    # Overlay error bars
    if sum_bkg_weights > 0:
        plt.errorbar(bin_centers, bkg_raw_counts, yerr=bkg_errors,
                     fmt='none', ecolor='black', alpha=0.7)
    if sum_signal_weights > 0:
        plt.errorbar(bin_centers, sig_raw_counts, yerr=sig_errors,
                     fmt='none', ecolor='black', alpha=0.7)
    
    plt.xlabel(xlabel)
    plt.ylabel("Normalised to Unity")
    plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)
    
    if log_scale:
        plt.yscale("log")
        current_ylim = plt.gca().get_ylim()

        if current_ylim[0] <= 0:
             plt.gca().set_ylim(bottom=min(1e-5, current_ylim[1] * 0.1)) # Set a small positive minimum

    plt.grid(True, alpha=0.3)
    if sum_bkg_weights > 0 and sum_signal_weights > 0:
       plt.legend()

    elif sum_bkg_weights > 0:
        plt.legend(handles=[plt.Line2D([0], [0], color='blue', lw=2, label='Background (QCD)')])

    elif sum_signal_weights > 0:
        plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=2, label='Signal (ggf_sm)')])

    plt.savefig(filename)
    plt.close()

def plot_weighted_matrix(mat, title, fname, features,
                         show_upper_triangle_only=True,
                         dpi=300, inches_per_feature=0.35):
    """
    Plot a covariance / correlation matrix with readable size.

    Parameters
    ----------
    mat : pd.DataFrame
        Square matrix (index and columns should be `features`).
    title : str
        Title for the figure.
    fname : str
        Where to save it (extension decides format).
    features : list[str]
        Ordered list of feature names (same order as `mat`).
    show_upper_triangle_only : bool, default True
        If True the lower triangle is masked so the colour bar range
        can be symmetric (-1…1 for correlations, full range for cov).
    """
    n_feat = len(features)
    side = max(6, inches_per_feature * n_feat)
    fig, ax = plt.subplots(figsize=(side, side))
    mask = np.triu(np.ones_like(mat, dtype=bool)) if show_upper_triangle_only else None
    sns.heatmap(mat, ax=ax, cmap='coolwarm', mask=mask, square=True,
                cbar_kws={'shrink': 0.7}, linewidths=0.4,
                xticklabels=features, yticklabels=features)
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
