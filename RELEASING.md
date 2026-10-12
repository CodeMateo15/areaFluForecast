# Releasing

A release happens when a version bump reaches `main`. On every push to
`main`, `.github/workflows/release.yml` reads `__version__`. It releases only
if both of these hold:
- no tag `v<version>` exists yet
- `CHANGELOG.md` gives that version a date, not "unreleased"

Otherwise it does nothing, so bumping the version early during development is
safe.

When it releases, it:
1. checks that `CITATION.cff` is at the same version
2. builds the sdist and wheel
3. installs the wheel on Linux and macOS (Python 3.10 and 3.13) and runs every
   offline test against it
4. publishes to PyPI with trusted publishing (no stored token)
5. tags the commit `v<version>` and creates the GitHub release, with the
   changelog section as its notes
6. leaves the rest to Zenodo, which archives the release and mints a DOI

Nothing is tagged or released unless PyPI accepted the upload. If a step
fails, fix the cause and push again, or rerun the workflow from the Actions
tab.

## Cutting a release

1. In `src/areafluforecast/__init__.py`, set `__version__` (the package
   metadata reads it from there).
2. In `CITATION.cff`, set `version:` and `date-released:`.
3. In `CHANGELOG.md`, change the version's heading from "unreleased" to the
   same date.
4. Commit and push to `main`. The tests check that the three files agree.

Versions ending in `rc1`, `b1` or `a1` become pre-releases on GitHub.
`pip install areaFluForecast` skips them on PyPI unless asked with `--pre`.

## After the first Zenodo release

Zenodo gives two DOIs:
- one for this version
- a *concept* DOI that always resolves to the latest version

Put the concept DOI in `CITATION.cff` so every later release cites it:

```yaml
doi: 10.5281/zenodo.NNNNNNN
```

Then add the badge Zenodo shows on the record page to the top of `README.md`.
These are documentation-only changes; they don't need a new version.

## One-time setup

- **PyPI trusted publisher.** On pypi.org, go to areaFluForecast → Manage →
  Publishing and add a GitHub publisher:
  - owner `CodeMateo15`
  - repository `areaFluForecast`
  - workflow `release.yml`
  - environment `pypi`
- **GitHub environment.** In the repository's Settings → Environments, create
  `pypi`. Adding yourself as a required reviewer makes every upload wait for
  your click.
- **Zenodo.** Log in to zenodo.org with GitHub, open GitHub in the account
  menu, and switch the repository on. Do this *before* pushing the release.
  Zenodo only archives releases published after it is switched on. It takes
  the record's title, authors, description, keywords and license from
  `CITATION.cff`.
