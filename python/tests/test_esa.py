import os
import pytest
import set_path
import qupled.qupled as qp
from qupled.classic import ESA, ClassicScheme


@pytest.fixture
def esa_instance():
    assert issubclass(qp.Stls, qp.Rpa)
    return ESA(ESA.Input(1.0, 1.0))


def test_default(esa_instance):
    assert issubclass(ESA, ClassicScheme)
    assert issubclass(ESA, qp.ESA)
    assert esa_instance.hdfFileName == "rs1.000_theta1.000_ESA.h5"


def test_compute(esa_instance, mocker):
    mockMPITime = mocker.patch("qupled.util.MPI.timer", return_value=0)
    mockMPIBarrier = mocker.patch("qupled.util.MPI.barrier")
    mockCompute = mocker.patch("qupled.qupled.ESA.compute")
    mockCheckStatusAndClean = mocker.patch("qupled.classic.ESA._checkStatusAndClean")
    mockSave = mocker.patch("qupled.classic.ESA._save")
    esa_instance.compute()
    assert mockMPITime.call_count == 2
    assert mockMPIBarrier.call_count == 1
    assert mockCompute.call_count == 1
    assert mockCheckStatusAndClean.call_count == 1
    assert mockSave.call_count == 1
