import pytest
from unittest.mock import patch
from openvqe.main_ucc import main
from openvqe.ucc_family.get_energy_ucc import EnergyUCC

class DummyEnergyUCC():
    def get_energies(self, *args, **kwargs):
        return (10, -1.137)

def test_main_output(mocker, capsys):
    mocker.patch.object(EnergyUCC, "__new__", return_value = DummyEnergyUCC())
    main()
    captured = capsys.readouterr()
    assert "Running in the non active case:" in captured.out
    assert "Pool size:  36" in captured.out
    assert "length of the cluster OP:  36" in captured.out
    assert "length of the cluster OPS:  36" in captured.out
    assert "iterations are: 10" in captured.out
    assert "results are: -1.137" in captured.out