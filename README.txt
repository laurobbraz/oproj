We recommend using a Python virtual environment.
To create one, use
$ python -m venv /path/to/new/virtual/env

To activate the virtual environment,
$ source /path/to/new/virtual/env/bin/activate

To install all the requirements run

$ pip install -r requirements.txt

The local projection .zip file has to be
placed inside the VASP simulation directory.

To run the code using NCORES cores and
saving outputs at output.dat, use

$ mpirun -np NCORES python3 local_projections_from_vasp.py >> output &
