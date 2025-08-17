from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chemtsv2")
except PackageNotFoundError:
    __version__ = "unknown"
