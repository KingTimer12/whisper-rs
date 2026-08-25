//! Locating libonnxruntime at runtime.
//!
//! `ort` runs in `load-dynamic` mode because static linking collides with
//! CTranslate2's `protobuf` and crashes the process with `SIGBUS`. The
//! trade-off is that the dylib has to be found at runtime, and a failure to
//! find it must explain itself: a bare `ort` initialisation error tells a
//! Python caller nothing actionable.

use crate::error::{Error, Result};
use std::path::PathBuf;
use std::sync::OnceLock;

static INIT: OnceLock<bool> = OnceLock::new();

/// Where the `onnxruntime` pip package lives, asked of the interpreter that
/// actually loaded this extension.
///
/// Shelling out to whichever `python3` is on `PATH` finds the WRONG
/// interpreter whenever the caller is in a virtualenv -- which is the normal
/// case. Observed directly: onnxruntime installed in the project venv, the
/// extension imported from that venv, and the probe still reporting "not
/// found", because `python3` resolved to a Homebrew interpreter that has no
/// onnxruntime. The user has done exactly what the error message told them to
/// do and the error message does not go away.
///
/// We are hosted BY a Python interpreter, so we can just ask it.
fn onnxruntime_dir() -> Option<PathBuf> {
    pyo3::Python::attach(|py| {
        use pyo3::types::PyAnyMethods;
        let module = py.import("onnxruntime").ok()?;
        let file: String = module.getattr("__file__").ok()?.extract().ok()?;
        PathBuf::from(file).parent().map(PathBuf::from)
    })
}

/// Candidate paths inside an installed `onnxruntime` pip package.
fn pip_candidates() -> Vec<PathBuf> {
    let Some(dir) = onnxruntime_dir() else {
        return Vec::new();
    };
    if dir.as_os_str().is_empty() {
        return Vec::new();
    }

    let capi = dir.join("capi");
    let mut found = Vec::new();
    // The wheel ships a version-stamped filename (libonnxruntime.1.29.0.dylib),
    // so the directory is scanned rather than a fixed name being guessed.
    if let Ok(entries) = std::fs::read_dir(&capi) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("libonnxruntime") || name.starts_with("onnxruntime") {
                found.push(entry.path());
            }
        }
    }
    found
}

/// Testable core: the first existing path among `explicit` then `candidates`.
fn locate_in(explicit: Option<PathBuf>, candidates: &[PathBuf]) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(path);
        }
        return Err(Error::OnnxRuntimeMissing {
            message: format!(
                "ORT_DYLIB_PATH points to {}, which does not exist. Unset it to \
                 use the onnxruntime pip package, or install one with \
                 `pip install whisper-rs[diarization]`.",
                path.display()
            ),
        });
    }

    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }

    Err(Error::OnnxRuntimeMissing {
        message: "diarization needs the onnxruntime shared library, which was \
                  not found. Install it with `pip install whisper-rs[diarization]`, \
                  or point ORT_DYLIB_PATH at an existing libonnxruntime."
            .into(),
    })
}

/// Locate libonnxruntime: `ORT_DYLIB_PATH` first, then the pip package.
pub fn locate() -> Result<PathBuf> {
    let explicit = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from);
    locate_in(explicit, &pip_candidates())
}

/// Initialise `ort` once per process.
///
/// `OnceLock` rather than repeated `init`: `ort`'s environment is global, and
/// a second `commit()` on an already-initialised environment returns `false`,
/// which would otherwise surface as a spurious error on the second
/// `transcribe(diarize=True)` call in a process.
pub fn init_ort() -> Result<()> {
    let path = locate()?;
    let committed = INIT.get_or_init(|| {
        ort::init_from(path.to_string_lossy().as_ref())
            .map(|env| env.commit())
            .unwrap_or(false)
    });

    if *committed {
        Ok(())
    } else {
        Err(Error::OnnxRuntimeMissing {
            message: format!(
                "found libonnxruntime at {} but could not initialise it. It may \
                 be older than the required 1.28, or built for another \
                 architecture.",
                path.display()
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_dylib_names_the_install_command() {
        // `locate` consults ORT_DYLIB_PATH first. Pointing it somewhere that
        // does not exist exercises the error path without depending on
        // whether onnxruntime happens to be installed here.
        let err = locate_in(Some("/nonexistent/libonnxruntime.dylib".into()), &[])
            .expect_err("a nonexistent path must not resolve");
        let message = err.to_string();
        assert!(
            message.contains("whisper-rs[diarization]"),
            "the error must name the fix, got: {message}"
        );
    }

    #[test]
    fn an_explicit_path_that_exists_is_used_verbatim() {
        // Any existing file stands in for the dylib: `locate` checks presence,
        // it does not validate the file's contents. `ort` reports a bad
        // library far more precisely than a guess here could.
        let this_file = std::path::PathBuf::from(file!());
        let found = locate_in(Some(this_file.clone()), &[]).expect("an existing path resolves");
        assert_eq!(found, this_file);
    }

    #[test]
    fn a_site_packages_candidate_is_found() {
        let this_file = std::path::PathBuf::from(file!());
        let found =
            locate_in(None, std::slice::from_ref(&this_file)).expect("a present candidate resolves");
        assert_eq!(found, this_file);
    }
}
