"""
*General:*
------------
[1] Package title: Feature Extraction of Electricity Load (FEEL)
[2] Authors: Maomao Hu, Dongjiao Ge, David Wallom
[3] Organization: Oxford e-Research Center, Department of Engineering Science, University of Oxford
[4] Contact info: maomao.hu@eng.ox.ac.uk
[5] Development time: Oct 2020
[6] Acknowledgement: This work was financially supported by the UK Engineering and Physical Sciences Research Council (EPSRC) under grant (EP/S030131/1).

*About FEEL:*
-------------
[1] This Python package (i.e., FEEL) aims to help energy data analysts to easily extract interpretable features
    of daily electricity profiles for further machine learning purpose.
[2] Two PY files (.py) are included in the FEEL package, including ifeel_transformation.py and ifeel_extraction.py.
[3] Two types of features can be extracted by using this package: 13 global features (GF) and 8 peak-period features (PF).
[4] The 13 global features are extracted based on raw time-series data, while the 8 peak-period features are extracted
    based on symbolic representation of time series. The feature extraction process is performed by calling the functions in ifeel_extraction.py.
[5] For fast peak-period feature extraction, Symbolic Aggregate approXimation (SAX) representation is first used to transform
    the time-series numerical patterns into alphabetical words. The feature transformation process is performed by
    calling the functions in ifeel_transformation.py.

*Notes:*
-------------
[1] To successfully run the FEEL, the following Python data analysis libraries need to be installed in advance: Numpy, Scipy, and Pandas
[2] The demonstration case has been tested on Python 3.7.7

*References*
------------
[1] Hu M, Ge D, Telford R, Stephen B, Wallom, B. Classification and characterization of intra-day load curves of PV and
    Non-PV households using interpretable feature extraction and feature-based clustering. Energy.
[2] Lin J, Keogh E, Wei L, Lonardi S. Experiencing SAX: a novel symbolic representation of time series. Data Mining and Knowledge Discovery. 2007;15:107-44.
[3] Keogh E, Lin J, Fu A. HOT SAX: efficiently finding the most unusual time series subsequence.  Fifth IEEE International Conference on Data Mining (ICDM'05)2005. p. 8 pp.
"""

from . import Feel_extraction
from . import Feel_transformation