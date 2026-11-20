import uplift


def test_package_imports_and_reports_version() -> None:
    assert uplift.__version__ == "1.0.0"
