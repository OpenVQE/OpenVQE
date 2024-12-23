def side_effect(*args, **kwargs):
    print("Mock qubit_adapt_vqe called")
    return (10, 10, -1.137, -1.137)

def test_main_output(mocker, capsys):
    mock_qubit_adapt_vqe = mocker.patch('openvqe.adapt.qubit_adapt_vqe.qubit_adapt_vqe', side_effect=side_effect)
    from openvqe.main_qubit_adapt import main
    main()
    captured = capsys.readouterr()
    assert "Running in the non active case:" in captured.out
    assert "Pool size:  70" in captured.out
    assert "length of the cluster OP:  70" in captured.out
    assert "length of the cluster OPS:  70" in captured.out
    assert "length of the pool 50" in captured.out
    assert "iterations are: 10" in captured.out
    assert "results are: -1.137" in captured.out
    mock_qubit_adapt_vqe.assert_called_once()