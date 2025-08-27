## Steps to Install MadGraph 3.5.1
- First git clone and setup the working directory using README file.

To install MadGraph 3.5.1,

```
wget https://herwig.hepforge.org/downloads?f=mirror/MG5_aMC_v3.5.1.tar.gz
tar -zxf MG5_aMC_v3.5.1.tar.gz
cd mg5amcnlo
./bin/mg5_aMC
```

Install LHAPDF and the required SMEFT@NLO model,

```
import model SMEFTatNLO
install lhapdf6
```

Exit out and change restrict cards of SMEFT@NLO. This zeros out irrelevant values to improve computational efficiency.

```
exit
cd models/SMEFTatNLO
rm restrict_LO.dat restrict_default.dat
cp -r ../../dihiggs_4b/cards/restrict_LO.dat .
cp -r ../../dihiggs_4b/cards/restrict_default.dat .
```
Finally, install the required dependencies (Pythia8, Delphes 3.5.1. etc).

```
cd ../HEPTools/
```

```
wget https://pythia.org/download/pythia83/pythia8306.tgz
tar -xzf pythia8306.tgz
cd pythia8306
./configure --prefix=$PWD/../pythia8-install
make -j14
make install
```

```
cd ..
```

```
wget http://cp3.irmp.ucl.ac.be/downloads/Delphes-3.5.0.tar.gz
tar -zxf Delphes-3.5.0.tar.gz
cd Delphes-3.5.0
make -j14
ln -s DelphesHepMC2 DelphesHepMC
```

Add the following lines to mg5amcnlo/input/mg5_configuration.txt to overwrite default installation paths:

```
pythia8_path = ./HEPTools/pythia-install
delphes_path = ./HEPTools/Delphes-3.5.0
lhapdf = ./HEPTools/lhapdf6_py3/bin/lhapdf-config
```

The remaining dependencies will be installed during the first MadGraph run (usually takes 10-15 minutes).