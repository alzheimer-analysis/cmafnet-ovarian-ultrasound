import os


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def artifact_dir():
    return os.path.join(project_root(), "artifacts")


def split_manifest_path():
    return os.path.join(artifact_dir(), "splits", "internal_v1.json")


def external_manifest_path():
    return os.path.join(artifact_dir(), "splits", "external_ids.json")


def normalizer_path():
    return os.path.join(artifact_dir(), "stats", "clinical_norm.json")


def checkpoint_path(name):
    return os.path.join(artifact_dir(), "checkpoints", name)


def results_path(name):
    return os.path.join(artifact_dir(), "results", name)
