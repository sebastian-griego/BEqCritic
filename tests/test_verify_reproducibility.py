from scripts.verify_reproducibility import verification_commands


def test_verification_commands_cover_reproducibility_gate():
    commands = verification_commands("python")

    assert commands[0] == ["python", "-m", "compileall", "-q", "beqcritic", "scripts", "tests"]
    assert any("--check" in command for command in commands)
    assert any("--verify-source-hashes" in command for command in commands)
    assert commands[-1] == ["python", "-m", "pytest", "-q"]


def test_verification_commands_can_skip_pytest():
    commands = verification_commands("python", include_tests=False)

    assert ["python", "-m", "pytest", "-q"] not in commands
