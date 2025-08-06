import uproot

def root_file_preview(file_path):
    """
    Preview size and columns of a ROOT file

    Args:
        file_path (str): Path to the ROOT file to preview   
    """
    with uproot.open(file_path) as file:
        print(f"\nFile: {file_path} contents:")
        print(file.classnames())

        # Try to get the Events TTree - check both possible locations
        events = None
        events_key = None
        
        if "Delphes" in file:
            events = file["Delphes"]
            events_key = "Delphes"
        elif "scNtuplizer/Delphes" in file:
            events = file["scNtuplizer/Delphes"]
            events_key = "scNtuplizer/Delphes"
        
        if events is not None:
            print(f"\n{events_key} TTree raw representation:")
            print(events)

            print("\n")
            print("Events (rows):", getattr(events, "num_entries", "N/A"))
            print("Number of branches (columns):", len(events.keys()))
            print("\nBranches (columns):", list(events.keys()))
            print("\n")
        else:
            print("\nNo 'Delphes' or 'scNtuplizer/Delphes' TTree found\n")

root_file_preview("/vols/cms/us322/02b_final_events_14_new/signal_sm/batch_2/run_01_decayed_1/tag_1_pythia8_events_delphes.root")