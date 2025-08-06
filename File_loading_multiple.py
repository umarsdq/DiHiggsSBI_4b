import uproot
import awkward as ak
import numpy as np

class data:
    def __init__(self,
                 signal_directories,
                 signal_file_ranges,
                 signal_processes,
                 qcd_directories,
                 qcd_file_ranges,
                 qcd_processes,
                 cross_sections,
                 luminosity,
                 variables=None,
                 scale_factor=1):
        
        self.signal_directories = signal_directories
        self.signal_file_ranges = signal_file_ranges
        self.signal_processes = signal_processes
        self.qcd_directories = qcd_directories
        self.qcd_file_ranges = qcd_file_ranges
        self.qcd_processes = qcd_processes
        self.cross_sections = cross_sections
        self.luminosity = luminosity
        self.variables = variables
        self.scale_factor = scale_factor

        # Pre-calculate scaled file ranges
        self.scaled_signal_ranges = [n // self.scale_factor for n in self.signal_file_ranges]
        self.scaled_qcd_ranges = [n // self.scale_factor for n in self.qcd_file_ranges]

    def load_signal(self, library="ak"):
        sig_list = []
        for directory, n_files, process in zip(self.signal_directories, self.scaled_signal_ranges, self.signal_processes):
            files = {f"{directory}data_{i}.root": "Events" for i in range(n_files)}
            
            # If self.variables is None, don't specify variables to load all
            if self.variables is None:
                df = uproot.concatenate(files, library=library)
            else:
                varlist = self.variables.get(process, self.variables) if isinstance(self.variables, dict) else self.variables
                df = uproot.concatenate(
                    files,
                    expressions=None,
                    filter_name=varlist,
                    library=library
                )

            n_events = len(df)
            weight = self.luminosity * self.cross_sections[process] / n_events if n_events > 0 else 0

            df = ak.with_field(df, weight, "weight")
            df = ak.with_field(df, 1, "label")
            sig_list.append(df)
            print(f"\nProcessed {process}: Events={n_events}, Weight={weight}")
            
        return ak.concatenate(sig_list) if sig_list else ak.Array([])

    def load_background(self, library="ak"):
        bkg_list = []
        for directory, n_files, process in zip(self.qcd_directories, self.scaled_qcd_ranges, self.qcd_processes):
            files = {f"{directory}data_{i}.root": "Events" for i in range(n_files)}
            
            # If self.variables is None, don't specify variables to load all
            if self.variables is None:
                df = uproot.concatenate(files, library=library)
            else:
                varlist = self.variables.get(process, self.variables) if isinstance(self.variables, dict) else self.variables
                df = uproot.concatenate(
                    files,
                    expressions=None,
                    filter_name=varlist,
                    library=library
                )

            n_events = len(df)
            weight = self.luminosity * self.cross_sections[process] / n_events if n_events > 0 else 0

            df = ak.with_field(df, weight, "weight")
            df = ak.with_field(df, 0, "label")
            bkg_list.append(df)

            print(f"\nProcessed {process}: Events={n_events}, Weight={weight}")

        return ak.concatenate(bkg_list) if bkg_list else ak.Array([])

    def load_process(self, process_name, library="ak"):
        directory = None
        n_files_orig = None
        is_signal = None

        if process_name in self.signal_processes:
            idx = self.signal_processes.index(process_name)
            directory = self.signal_directories[idx]
            n_files_orig = self.signal_file_ranges[idx]
            is_signal = True
        elif process_name in self.qcd_processes:
            idx = self.qcd_processes.index(process_name)
            directory = self.qcd_directories[idx]
            n_files_orig = self.qcd_file_ranges[idx]
            is_signal = False
        else:
            raise ValueError(f"Process '{process_name}' not found in configuration.")

        n_files = n_files_orig // self.scale_factor
        files = {f"{directory}data_{i}.root": "Events" for i in range(n_files)}
        
        # If self.variables is None, don't specify variables to load all
        if self.variables is None:
            print(f"\nLoading process: {process_name} from {directory} ({n_files} files) with all available variables")
            df = uproot.concatenate(files, library=library)
        else:
            varlist = self.variables.get(process_name, self.variables) if isinstance(self.variables, dict) else self.variables
            print(f"\nLoading process: {process_name} from {directory} ({n_files} files)")
            df = uproot.concatenate(
                files,
                expressions=None,
                filter_name=varlist,
                library=library
            )

        n_events = len(df)
        weight = self.luminosity * self.cross_sections[process_name] / n_events if n_events > 0 else 0

        df = ak.with_field(df, weight, "weight")

        print(f"\nProcessed {process_name}: Events={n_events}, Weight={weight}")
        return df

    def load_full(self, library="ak"):
        signal_data = self.load_signal(library=library)
        background_data = self.load_background(library=library)

        if not isinstance(signal_data, ak.Array):
             signal_data = ak.Array([])
        if not isinstance(background_data, ak.Array):
             background_data = ak.Array([])

        if len(signal_data) == 0 and len(background_data) == 0:
            return ak.Array([])
        elif len(signal_data) == 0:
            return background_data
        elif len(background_data) == 0:
            return signal_data
        else:
            return ak.concatenate([signal_data, background_data])

# Sample configuration
def file_config(scale_factor=1):
    luminosity = 0.3  # pb^-1
    
    cross_sections = {
        "ggf_sm": 1.575e-02,
        "bbtt": 1.575e-02,
        "QCD_Pt-20To30": 4.329e+08,
        "QCD_Pt-30To50": 1.172e+08,
        "QCD_Pt-50To80": 1.749e+07,
        "QCD_Pt-80To120": 2.657e+06,
        "QCD_Pt-120To170": 4.678e+05,
        "QCD_Pt-170To300": 1.203e+05,
        "QCD_Pt-300To470": 8.157e+03,
        "QCD_Pt-470To600": 6.831e+02,
        "QCD_Pt-600ToInf": 2.416e+02
    }
    
    # Signal samples
    signal_directories = [
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/ggf_sm/cat_base/prod_250227/",
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/bbtt/cat_base/prod_250227/"
    ]
    
    signal_file_ranges = [
        50,
        170
    ]
    
    signal_processes = [
        "ggf_sm",
        "bbtt"
    ]
    
    # QCD samples
    qcd_directories = [
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/QCD_Pt-20To30/cat_base/prod_250123/",
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/QCD_Pt-30To50/cat_base/prod_250123/",
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/QCD_Pt-50To80/cat_base/prod_250123/",
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/QCD_Pt-80To120/cat_base/prod_250123/",
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/QCD_Pt-120To170/cat_base/prod_250123/",
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/QCD_Pt-170To300/cat_base/prod_250123/",
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/QCD_Pt-300To470/cat_base/prod_250123/",
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/QCD_Pt-470To600/cat_base/prod_250123/",
        "/vols/cms/jleonhol/cmt/PreprocessRDF/hh_phase2/QCD_Pt-600ToInf/cat_base/prod_250123/"
    ]
    
    qcd_file_ranges = [
        472,
        485,
        299,
        148,
        100,
        99,
        99,
        98,
        93
    ]
    
    qcd_processes = [
        "QCD_Pt-20To30",
        "QCD_Pt-30To50",
        "QCD_Pt-50To80",
        "QCD_Pt-80To120",
        "QCD_Pt-120To170",
        "QCD_Pt-170To300",
        "QCD_Pt-300To470",
        "QCD_Pt-470To600",
        "QCD_Pt-600ToInf"
    ]
    
    return {
        "signal_directories": signal_directories,
        "signal_file_ranges": signal_file_ranges,
        "signal_processes": signal_processes,
        "qcd_directories": qcd_directories,
        "qcd_file_ranges": qcd_file_ranges,
        "qcd_processes": qcd_processes,
        "cross_sections": cross_sections,
        "luminosity": luminosity
    } 
