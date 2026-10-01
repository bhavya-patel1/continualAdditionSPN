# Continual Addition for Sum-Product Networks
All required dependencies are listed in requirements.txt. You'll also need to install the Einet repository separately.
`pip install git+https://github.com/braun-steven/simple-einet`

Lib folder has the insertion methodology. Specifically:  `insertion.py`
`conf/config.yaml` can be used for updating experiment configurations
The experiment scripts are in the root directory. Specifically:
`run_benchmarks.py` runs ContinualSPN against the naive baseline and full retraining only.
`run_einet_benchmarks.py` runs the full test suite, including all RAT-SPN/Einet insertion strategies.