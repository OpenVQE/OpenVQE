# mrh
GPL research code of Matthew R. Hermes

### WARNING!
MC-PDFT and MC-DCFT modules (other than a few experimental features) have been *moved to [pyscf-forge](https://github.com/pyscf/pyscf-forge)* and are currently in the process of being removed from this project.

### DEPENDENCIES:
- PySCF, including all header files in pyscf/lib
- [pyscf-forge](https://github.com/pyscf/pyscf-forge)
- see `pyscf_version.txt` and `pyscf-forge_version.txt` for last-tested versions or commits

### INSTALLATION:
- cd /path/to/mrh/lib
- mkdir build ; cd build
- cmake ..
- make
- Add /path/to/mrh to Python's search path somehow. Examples:
    * "sys.path.append ('/path/to')" in Python script files
    * "export PYTHONPATH=/path/to:$PYTHONPATH" in the shell
    * "ln -s /path/to/mrh /path/to/python/site-packages/mrh" in the shell
- If you installed PySCF from source and the compilation still fails, try setting the path to the PySCF library directory manually:
`cmake -DPYSCFLIB=/full/path/to/pyscf/lib ..`

### Notes
- The dev branch is continuously updated. The master branch is updated every time I pull PySCF and confirm that everything still works. If you have some issue and you think it may be related to PySCF version mismatch, try using the master branch and the precise PySCF commit indicated above.
- If you are using Intel MKL as the BLAS library, you may need to enable the corresponding cmake option:
`cmake -DBLA_VENDOR=Intel10_64lp_seq ..`

### ACKNOWLEDGMENTS:
- This work is supported by the U.S. Department of Energy, Office of Basic Energy Sciences, Division of Chemical Sciences, Geosciences and Biosciences through the Nanoporous Materials Genome Center under award DE-FG02-17ER16362.

