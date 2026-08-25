//! The only module allowed to use pyo3 types.

pub mod iter;
pub mod model;
pub mod segment;

use crate::error::Error;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Map a crate error onto the Python exception a caller would expect.
pub fn to_pyerr(err: Error) -> PyErr {
    let message = err.to_string();
    match err {
        Error::AudioRead { .. } | Error::AudioFormat { .. } | Error::AudioEmpty { .. } => {
            PyValueError::new_err(message)
        }
        Error::ModelNotFound { .. } | Error::Download { .. } => PyOSError::new_err(message),
        Error::Resample(_) | Error::Vad(_) | Error::Ct2(_) | Error::Diarize(_) => {
            PyRuntimeError::new_err(message)
        }
        Error::OnnxRuntimeMissing { .. } => PyOSError::new_err(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;
    use std::path::PathBuf;

    #[test]
    fn audio_errors_become_value_errors() {
        Python::initialize();
        Python::attach(|py| {
            let err = to_pyerr(Error::AudioEmpty {
                path: PathBuf::from("/tmp/x.wav"),
            });
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn model_errors_become_os_errors() {
        Python::initialize();
        Python::attach(|py| {
            let err = to_pyerr(Error::Download {
                name: "tiny".into(),
                message: "offline".into(),
            });
            assert!(err.is_instance_of::<PyOSError>(py));
        });
    }

    #[test]
    fn backend_errors_become_runtime_errors() {
        Python::initialize();
        Python::attach(|py| {
            let err = to_pyerr(Error::Ct2("boom".into()));
            assert!(err.is_instance_of::<PyRuntimeError>(py));
        });
    }

    #[test]
    fn messages_keep_the_path_and_model_name() {
        Python::initialize();
        let msg = to_pyerr(Error::ModelNotFound {
            name: "large-v3".into(),
            path: PathBuf::from("/cache/models"),
            message: "missing required file(s): model.bin".into(),
        })
        .to_string();

        assert!(msg.contains("large-v3"), "got {msg}");
        assert!(msg.contains("/cache/models"), "got {msg}");
        assert!(msg.contains("model.bin"), "got {msg}");
    }
}
