# Contributing to diffindiff

Thank you for your interest in contributing to `diffindiff`.

`diffindiff` is an open-source Python package for convenient Difference-in-Differences (DiD) analyses. It is designed to support the entire workflow of DiD analyses, including data preparation, treatment and control group definition, model estimation, diagnostic testing, and visualization. The package supports conventional and staggered treatment adoption, as well as extensions such as two-way fixed effects, group- or individual-specific treatment effects, triple-difference estimation, and DiD models with a synthetic control unit. Particular attention is paid to extensive testing and diagnosis with regard to the data, treatments, and groups.

## Reporting issues

Bug reports, questions, and feature requests are welcome.

Please use the [GitHub Issues](https://github.com/geowieland/diffindiff_official/issues) page to report:

- bugs or unexpected behavior,
- documentation issues,
- feature requests,
- questions about the use of the package.

When reporting a bug, please provide a minimal reproducible example where possible, including the `diffindiff` version and relevant Python, dependency, and environment information. For issues related to statistical results, please also describe the data structure, model specification, and expected behavior.

## Contributing code

Contributions via pull requests are welcome.

Before submitting a pull request:

1. Fork the repository and create a separate branch for your changes.
2. Keep changes focused on a single feature, bug fix, or documentation improvement.
3. Follow the existing code structure and style.
4. Update the documentation, docstrings, or examples when appropriate.
5. Add or update tests where appropriate.
6. Make sure that existing functionality is not unintentionally changed.

Pull requests should include a short description of the changes and their motivation. For changes to statistical methods or model behavior, please also explain the methodological rationale and, where appropriate, provide references to the relevant literature.

## Documentation

Improvements to the documentation, examples, and methodological explanations are welcome.

Documentation changes should be consistent with the existing terminology and structure of the project. Examples should be reproducible and accurately reflect the package's functionality.

## Development

The package can be installed directly from the repository:

```bash
pip install git+https://github.com/geowieland/diffindiff_official.git
```

For development, clone the repository and install the package locally:

```bash
git clone https://github.com/geowieland/diffindiff_official.git
cd diffindiff_official
pip install -e .
```

The `tests/` directory contains usage examples for many of the included functions and can serve as a reference when developing or extending the package.

## Scientific contributions

Contributions related to the implementation or extension of Difference-in-Differences, staggered treatment adoption, treatment effect estimation, synthetic control units, triple-difference estimation, or diagnostic procedures should include appropriate references to the underlying scientific literature where relevant.

Please describe methodological changes clearly so that their scientific purpose, assumptions, and implementation can be reviewed. Where applicable, contributions should include tests or reproducible examples demonstrating the behavior of the proposed changes, particularly for treatment effect estimates, standard errors, confidence intervals, and diagnostic results.

If you use the `diffindiff` Python package in your research, please cite the software as described in the [README](https://github.com/geowieland/diffindiff_official#citation).

## Code of conduct

Please keep discussions and contributions respectful, constructive, and focused on improving the software.

## License

By contributing to this repository, you agree that your contributions will be licensed under the MIT License used by `diffindiff`.