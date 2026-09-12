"""Check manual publishing confirmation and the installed wheel's version."""

import os
from importlib.metadata import version

from packaging.version import Version


def main():
    """Allow build-only runs; reject unsafe publishing requests."""
    confirmation = os.environ.get("CONFIRM_REF", "")
    if not confirmation:
        print("Build only. Nothing will be published to PyPI.")
        return

    selected_ref = os.environ["RELEASE_REF"]
    if confirmation != selected_ref:
        raise SystemExit("Confirmation must exactly match the selected branch or tag.")

    release = Version(version("fatqat"))
    if release.is_devrelease or release.local is not None:
        raise SystemExit("Development and local versions cannot be published to PyPI.")

    print(f"Confirmed publishing fatqat {release} from {selected_ref}.")


if __name__ == "__main__":
    main()
