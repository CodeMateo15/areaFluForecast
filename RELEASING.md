# Releasing

Releases are cut by pushing a tag. `.github/workflows/release.yml` then:

1. checks that the tag, `__version__`, `CITATION.cff` and `CHANGELOG.md` agree
2. builds the sdist and wheel
3. installs the wheel on Linux and macOS (Python 3.10 and 3.13) and runs every
   offline test against it
4. publishes to PyPI with trusted publishing (no stored token)
5. creates the GitHub release, with the changelog section as its notes;
   Zenodo archives it and mints a DOI

## Cutting a release

1. Set the version in `src/areafluforecast/__init__.py` (`__version__`; the
   package metadata reads it from there) and in `CITATION.cff`.
2. In `CHANGELOG.md`, change the version's heading from "unreleased" to
   today's date.
3. Commit, then tag and push:

   ```bash
   git tag v1.0.0
   git push origin main v1.0.0
   ```

4. Watch the run under the repository's Actions tab. If any step fails before
   `publish`, nothing has been released: fix it, delete the tag (`git tag -d
   v1.0.0 && git push origin :v1.0.0`) and tag again.
5. After a release Zenodo archives, put the DOI in `CITATION.cff` (the `doi:`
   line) and in the README.

Versions ending in `rc1`, `b1` or `a1` are marked as pre-releases on GitHub.
`pip install areaFluForecast` skips them on PyPI unless asked with `--pre`.

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
  menu, and switch the repository on. Do this before the release you want a
  DOI for. Zenodo only sees releases published after it is switched on.
