def side_effect(*args, **kwargs):
    print("Mock fermionic_adapt_vqe called")
    return (10, 10, -1.137, -1.137)

def test_main_output(mocker, capsys):
    mock_fermionic_adapt_vqe = mocker.patch('openvqe.adapt.fermionic_adapt_vqe.fermionic_adapt_vqe', side_effect=side_effect)
    from openvqe.main_fermionic_adapt import main
    main()
    captured = capsys.readouterr()
    assert "Running in the non active case:" in captured.out
    assert "Pool size:  175" in captured.out
    assert "length of the cluster OP:  175" in captured.out
    assert "length of the cluster OPS:  175" in captured.out
    assert "Running in the active case:" in captured.out
    assert "Pool size:  69" in captured.out
    assert "length of the cluster OP:  69" in captured.out
    assert "length of the cluster OPS:  69" in captured.out