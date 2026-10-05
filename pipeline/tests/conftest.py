"""Tests import src first, so that its numeric pins apply before numpy loads (as in the CLI)."""
import src  # noqa: F401
