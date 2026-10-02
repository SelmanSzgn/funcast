# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-10-02

### Fixed
- The `degree` parameter is now passed to the B-spline bases in `FunCast`
  and `select_h_rrss`. Previously it only set minimum sizes, and the bases
  were always cubic.
- `get_basis` now accepts a `degree` parameter.
- `funcast.__version__` is now consistent with the package version.

## [1.0.0]

### Added
- First release: `FunCast` model, B-spline and Fourier bases, and
  automatic selection of `h` with the RRSS criterion.
