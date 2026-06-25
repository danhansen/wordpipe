from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from typing import Callable, Literal
from urllib.request import urlretrieve
import zipfile


DEFAULT_MODEL_REPO = (
    "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
)
DEFAULT_MODEL_FILES = (
    "tokens.txt",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "README.md",
)
DEFAULT_NEMO_SOURCE_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
DEFAULT_NEMO_SOURCE_FILENAME = "nemotron-3.5-asr-streaming-0.6b.nemo"
DEFAULT_PREBUILT_PROFILE_REPO = "danhansen/wordpipe-nemotron-3.5-asr-streaming-0.6b"
ModelProfile = Literal["fast", "compact"]
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class DownloadPlan:
    repo_id: str
    output_dir: Path
    files: tuple[str, ...]

    @property
    def model_dir(self) -> Path:
        return self.output_dir / self.repo_id.split("/")[-1]


@dataclass(frozen=True)
class ModelProfileSpec:
    name: ModelProfile
    title: str
    description: str
    build_profile: str
    output_name: str
    prebuilt_filename: str
    emit_ort_format: bool = False

    def output_dir(self, model_root: Path) -> Path:
        return model_root.expanduser() / self.output_name

    def runtime_dir(self, model_root: Path) -> Path:
        output = self.output_dir(model_root)
        if self.emit_ort_format:
            return output.with_name(f"{output.name}-ort-format")
        return output


@dataclass(frozen=True)
class _PreparedProfileSource:
    path: Path
    cleanup_dir: Path | None = None


MODEL_PROFILES: dict[ModelProfile, ModelProfileSpec] = {
    "fast": ModelProfileSpec(
        name="fast",
        title="Fast",
        description="FP32 projected-cache model; fastest validated profile, largest footprint.",
        build_profile="fp32-projected",
        output_name="nemotron-wordpipe-fast-fp32-projected",
        prebuilt_filename="wordpipe-nemotron-fast-fp32-projected.tar.gz",
    ),
    "compact": ModelProfileSpec(
        name="compact",
        title="Compact",
        description="Dynamic-int8 projected-cache model with fixed shapes and ORT-format startup.",
        build_profile="compact-fixed-shape",
        output_name="nemotron-wordpipe-compact-fixed-shape",
        prebuilt_filename="wordpipe-nemotron-compact-fixed-shape.tar.gz",
        emit_ort_format=True,
    ),
}


def make_download_plan(
    output_dir: Path,
    repo_id: str = DEFAULT_MODEL_REPO,
    include_test_wavs: bool = False,
) -> DownloadPlan:
    files = list(DEFAULT_MODEL_FILES)
    if include_test_wavs:
        files.extend(["test_wavs/en.wav", "test_wavs/ja.wav"])
    return DownloadPlan(repo_id=repo_id, output_dir=output_dir.expanduser(), files=tuple(files))


def download_model(plan: DownloadPlan, force: bool = False) -> Path:
    model_dir = plan.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in plan.files:
        destination = model_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            print(f"skip {destination}", file=sys.stderr)
            continue
        url = model_file_url(plan.repo_id, filename)
        print(f"download {url}", file=sys.stderr)
        temporary = destination.with_name(destination.name + ".part")
        urlretrieve(url, temporary, _progress_reporter(destination))
        temporary.replace(destination)
    return model_dir


def model_file_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _progress_reporter(destination: Path):
    last_report = 0.0

    def report(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < 5 and block_count != 0:
            return
        last_report = now
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            print(
                f"  {destination.name}: {downloaded}/{total_size} bytes ({percent:.1f}%)",
                file=sys.stderr,
            )
        else:
            print(f"  {destination.name}: {downloaded} bytes", file=sys.stderr)

    return report


def default_model_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "wordpipe" / "models"
    return Path.home() / ".local" / "share" / "wordpipe" / "models"


def default_nemo_source_path(model_root: Path | None = None) -> Path:
    root = model_root.expanduser() if model_root is not None else default_model_root()
    return root / "sources" / DEFAULT_NEMO_SOURCE_FILENAME


def profile_spec(name: str) -> ModelProfileSpec:
    if name not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile: {name}")
    return MODEL_PROFILES[name]  # type: ignore[index]


def profile_runtime_dir(model_root: Path, profile: str) -> Path:
    return profile_spec(profile).runtime_dir(model_root)


def profile_installed(model_root: Path, profile: str) -> bool:
    runtime_dir = profile_runtime_dir(model_root, profile)
    return model_runtime_dir_valid(runtime_dir)


def model_runtime_dir_valid(runtime_dir: Path) -> bool:
    return (runtime_dir / "tokenizer.model").exists() and (
        (runtime_dir / "encoder.ort").exists() or (runtime_dir / "encoder.onnx").exists()
    ) and ((runtime_dir / "decoder_joint.ort").exists() or (runtime_dir / "decoder_joint.onnx").exists())


def source_is_built_profile(source: Path) -> bool:
    return source.is_dir() and model_runtime_dir_valid(source)


def source_may_be_built_profile_archive(source: Path) -> bool:
    name = source.name.lower()
    if name.endswith(".nemo"):
        return False
    return name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))


def install_built_profile(
    *,
    source: Path,
    model_root: Path,
    profile: str,
    force: bool = False,
) -> Path:
    prepared_source = _prepare_built_profile_source(source)
    destination = profile_runtime_dir(model_root, profile)
    if destination.exists():
        if not force:
            raise RuntimeError(f"profile {profile!r} is already installed at {destination}; pass --force to overwrite it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(prepared_source.path, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        if prepared_source.cleanup_dir is not None:
            shutil.rmtree(prepared_source.cleanup_dir, ignore_errors=True)
    return destination


def download_prebuilt_profile_archive(
    *,
    profile: str,
    model_root: Path,
    repo_id: str = DEFAULT_PREBUILT_PROFILE_REPO,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    spec = profile_spec(profile)
    output_dir = model_root.expanduser() / "downloads" / repo_id.replace("/", "--")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / spec.prebuilt_filename
    if destination.exists() and not force:
        _progress(progress, f"Using cached prebuilt profile: {destination}")
        return destination

    try:
        from huggingface_hub import hf_hub_download

        _progress(progress, f"Downloading {spec.title} profile from {repo_id}")
        env_value = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        try:
            import hf_transfer  # noqa: F401
        except ImportError:
            enable_hf_transfer = False
        else:
            enable_hf_transfer = True
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=spec.prebuilt_filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                force_download=force,
            )
        finally:
            if enable_hf_transfer:
                if env_value is None:
                    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
                else:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = env_value
        path = Path(downloaded)
        _progress(progress, f"Prebuilt profile archive ready: {path}")
        return path
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download prebuilt Wordpipe model profiles. "
            "Install it with hf_transfer support, or pass --source with a local profile archive."
        ) from exc


def install_prebuilt_profile(
    *,
    source: Path,
    model_root: Path,
    profile: str,
    python: Path = Path(sys.executable),
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    spec = profile_spec(profile)
    prepared_source = _prepare_built_profile_source(source)
    try:
        onnx_dir = spec.output_dir(model_root)
        _install_prepared_profile(prepared_source.path, onnx_dir, force=force)
    finally:
        if prepared_source.cleanup_dir is not None:
            shutil.rmtree(prepared_source.cleanup_dir, ignore_errors=True)

    if not spec.emit_ort_format:
        _progress(progress, f"Model profile ready: {onnx_dir}")
        return onnx_dir

    runtime_dir = spec.runtime_dir(model_root)
    if runtime_dir.exists() and not force and model_runtime_dir_valid(runtime_dir):
        _progress(progress, f"Using cached ORT runtime profile: {runtime_dir}")
        return runtime_dir

    command = [
        str(python.expanduser()),
        str(wordpipe_scripts_dir() / "convert_nemotron_to_ort_format.py"),
        str(onnx_dir),
        str(runtime_dir),
        "--force",
    ]
    _progress(progress, " ".join(command))
    _run_with_progress(command, progress)
    _progress(progress, f"Model profile ready: {runtime_dir}")
    return runtime_dir


def _install_prepared_profile(source: Path, destination: Path, *, force: bool) -> None:
    if destination.exists():
        if not force:
            raise RuntimeError(
                f"profile output already exists at {destination}; pass --force to overwrite it"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(source, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _prepare_built_profile_source(source: Path) -> _PreparedProfileSource:
    source = source.expanduser()
    if source_is_built_profile(source):
        return _PreparedProfileSource(source)
    if source.is_file() and source.name.lower().endswith(".nemo"):
        raise RuntimeError(f"{source} is a NeMo source model, not a built Wordpipe model profile.")
    if source.is_file() and source_may_be_built_profile_archive(source) and tarfile.is_tarfile(source):
        import tempfile

        tempdir = Path(tempfile.mkdtemp(prefix="wordpipe-profile-"))
        try:
            with tarfile.open(source, "r:*") as archive:
                _extract_tar_safely(archive, tempdir)
            return _PreparedProfileSource(_find_built_profile_dir(tempdir), tempdir)
        except Exception:
            shutil.rmtree(tempdir, ignore_errors=True)
            raise
    if source.is_file() and source_may_be_built_profile_archive(source) and zipfile.is_zipfile(source):
        import tempfile

        tempdir = Path(tempfile.mkdtemp(prefix="wordpipe-profile-"))
        try:
            with zipfile.ZipFile(source) as archive:
                _extract_zip_safely(archive, tempdir)
            return _PreparedProfileSource(_find_built_profile_dir(tempdir), tempdir)
        except Exception:
            shutil.rmtree(tempdir, ignore_errors=True)
            raise
    raise RuntimeError(
        f"{source} is not a built Wordpipe model profile. Expected a directory or archive "
        "containing tokenizer.model plus encoder/decoder_joint ONNX or ORT files."
    )


def _extract_zip_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for info in archive.infolist():
        target = (destination / info.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Refusing unsafe zip member: {info.filename}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise RuntimeError(f"Refusing unsupported zip member: {info.filename}")
        archive.extract(info, destination)


def _extract_tar_safely(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Refusing unsafe tar member: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"Refusing unsupported tar member: {member.name}")
    archive.extractall(destination)


def _find_built_profile_dir(root: Path) -> Path:
    if source_is_built_profile(root):
        return root
    matches = [path for path in root.rglob("tokenizer.model") if source_is_built_profile(path.parent)]
    if len(matches) == 1:
        return matches[0].parent
    if not matches:
        raise RuntimeError(f"archive did not contain a built Wordpipe model profile: {root}")
    raise RuntimeError(f"archive contained multiple built Wordpipe model profiles: {root}")


def download_nemo_source(
    source: str = DEFAULT_NEMO_SOURCE_REPO,
    output_path: Path | None = None,
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    candidate = Path(source).expanduser()
    if candidate.exists():
        _progress(progress, f"Using local source model: {candidate}")
        return candidate

    destination = output_path.expanduser() if output_path is not None else default_nemo_source_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        _progress(progress, f"Using cached source model: {destination}")
        return destination

    try:
        from huggingface_hub import hf_hub_download

        _progress(progress, f"Downloading source model from {source}")
        env_value = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        try:
            import hf_transfer  # noqa: F401
        except ImportError:
            enable_hf_transfer = False
        else:
            enable_hf_transfer = True
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        try:
            downloaded = hf_hub_download(
                repo_id=source,
                filename=DEFAULT_NEMO_SOURCE_FILENAME,
                local_dir=destination.parent,
                local_dir_use_symlinks=False,
                force_download=force,
            )
        finally:
            if enable_hf_transfer:
                if env_value is None:
                    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
                else:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = env_value
        path = Path(downloaded)
        if path != destination:
            if destination.exists():
                destination.unlink()
            path.replace(destination)
        _progress(progress, f"Source model ready: {destination}")
        return destination
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the NeMo source model. "
            "Install it with hf_transfer support or pass a local .nemo path."
        ) from exc


def build_profile_command(
    *,
    source: Path,
    model_root: Path,
    profile: str,
    python: Path,
    force: bool = False,
) -> list[str]:
    spec = profile_spec(profile)
    output_dir = spec.output_dir(model_root)
    work_dir = profile_build_dir(model_root, profile)
    return [
        str(python),
        str(wordpipe_scripts_dir() / "build_nemotron_wordpipe_model.py"),
        str(source.expanduser()),
        str(output_dir),
        "--work-dir",
        str(work_dir),
        "--profile",
        spec.build_profile,
        *(["--emit-ort-format"] if spec.emit_ort_format else []),
        *(["--force"] if force else []),
    ]


def profile_build_dir(model_root: Path, profile: str) -> Path:
    spec = profile_spec(profile)
    return model_root.expanduser() / "build" / spec.name


def wordpipe_scripts_dir() -> Path:
    override = os.environ.get("WORDPIPE_SCRIPTS_DIR")
    if override:
        return Path(override).expanduser()

    repo_scripts = Path(__file__).resolve().parents[2] / "scripts"
    if (repo_scripts / "build_nemotron_wordpipe_model.py").exists():
        return repo_scripts

    installed_scripts = Path(sys.prefix) / "share" / "wordpipe" / "scripts"
    if (installed_scripts / "build_nemotron_wordpipe_model.py").exists():
        return installed_scripts

    app_scripts = Path("/app/share/wordpipe/scripts")
    if (app_scripts / "build_nemotron_wordpipe_model.py").exists():
        return app_scripts

    return repo_scripts


def build_model_profile(
    *,
    source: Path,
    model_root: Path,
    profile: str,
    python: Path = Path(sys.executable),
    force: bool = False,
    dry_run: bool = False,
    keep_build_dir: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    command = build_profile_command(
        source=source,
        model_root=model_root,
        profile=profile,
        python=python,
        force=force,
    )
    rendered_command = " ".join(command)
    print(rendered_command, file=sys.stderr)
    _progress(progress, rendered_command)
    build_dir = profile_build_dir(model_root, profile)
    if not dry_run:
        _run_with_progress(command, progress)
        if not keep_build_dir and build_dir.exists():
            _progress(progress, f"Removing build intermediates: {build_dir}")
            shutil.rmtree(build_dir)
    runtime_dir = profile_runtime_dir(model_root, profile)
    _progress(progress, f"Model profile ready: {runtime_dir}")
    return runtime_dir


def _run_with_progress(command: list[str], progress: ProgressCallback | None) -> None:
    if progress is None:
        subprocess.run(command, check=True)
        return

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        message = line.rstrip()
        if message:
            _progress(progress, message)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
