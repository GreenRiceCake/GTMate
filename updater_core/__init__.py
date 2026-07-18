"""Core services for the GTMate updater."""

from .manifest import UPDATER_PROTOCOL_VERSION, UpdateInfo, UpdatePackage

__all__ = [
    "UPDATER_PROTOCOL_VERSION",
    "UpdateInfo",
    "UpdatePackage",
]
