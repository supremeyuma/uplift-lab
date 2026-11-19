import uplift


def test_package_imports_and_reports_version() -> None:
    assert uplift.__version__ == "0.1.0"
