import os
import subprocess

import pytest


@pytest.fixture
def setup_and_cleanup():
    output_file_path = "./result/example01/result_C1.0.csv"
    if os.path.exists(output_file_path):
        print(f"[INFO] Old output file exists: {output_file_path}")
        os.remove(output_file_path)
        print("[INFO] Deleted the file.")
    print("[INFO] ChemTSv2 running...")
    yield
    if os.path.exists(output_file_path):
        print("[INFO] Cleanup the output file...")
        os.remove(output_file_path)
        print(f"[INFO] Deleted file: {output_file_path}")
    

def test_debug_check_success(setup_and_cleanup):
    result01 = subprocess.run(
        ["chemtsv2", "-c", "config/setting.yaml", "--debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result01.returncode == 0

    result02 = subprocess.run(
        ["chemtsv2-debug-check"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result02.returncode == 0, f"Expected return code 0 but got {result02.returncode}"
    assert "[INFO] Output validation successful" in result02.stdout, f"Expected '[INFO] Output validation successful' but got {result02.stdout}"


def test_debug_check_failure(setup_and_cleanup):
    result01 = subprocess.run(
        ["chemtsv2", "-c", "config/setting.yaml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result01.returncode == 0

    result02 = subprocess.run(
        ["chemtsv2-debug-check"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result02.returncode == 1, f"Expected return code 1 but got {result02.returncode}"
    assert "[ERROR] Output validation failed. Please review your changes." in result02.stdout, f"Expected error details but got {result02.stdout}"
